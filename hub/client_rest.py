"""
Dev Vector DB Hub REST Client
Provides REST API integration for the central knowledge hub.
"""

import os
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sentence_transformers
from rich.console import Console

console = Console()

@dataclass
class HubItem:
    """Represents an item to be stored in the hub."""
    content: str
    content_type: str  # pattern, concept, documentation, idea, code
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    id: Optional[str] = None

class HubRestClient:
    """REST client for Dev Vector DB Hub integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the REST client."""
        self.config = config
        self.base_url = config.get("base_url", "http://localhost:8003")
        self.api_key = os.getenv("HUB_API_KEY", config.get("api_key", ""))
        self.project_id = config.get("project_id", "")
        
        # Initialize embedding model
        self.embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedding_dimension = config.get("embedding_dimension", 384)
        self.embedding_model = sentence_transformers.SentenceTransformer(self.embedding_model_name)
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=config.get("rest", {}).get("max_retries", 3),
            backoff_factor=config.get("rest", {}).get("retry_backoff", 2.0),
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
        }
        
        # Cache for embeddings
        self.cache = {}
        self.cache_enabled = config.get("cache", {}).get("enabled", True)
        self.cache_ttl = config.get("cache", {}).get("ttl_seconds", 3600)
        self.cache_max_size = config.get("cache", {}).get("max_size", 1000)
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make a request to the hub API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, headers=self.headers, timeout=30)
            elif method.upper() == "POST":
                response = self.session.post(url, headers=self.headers, json=data, timeout=30)
            elif method.upper() == "PUT":
                response = self.session.put(url, headers=self.headers, json=data, timeout=30)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, headers=self.headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Hub API request failed: {e}[/red]")
            raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using the configured model."""
        if self.cache_enabled and text in self.cache:
            return self.cache[text]
        
        embedding = self.embedding_model.encode(text).tolist()
        
        if self.cache_enabled:
            if len(self.cache) >= self.cache_max_size:
                # Simple LRU: remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[text] = embedding
        
        return embedding
    
    def _anonymize_content(self, content: str, metadata: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Anonymize content and metadata for privacy."""
        # Simple PII removal - in production, use a proper PII detection library
        anonymized_content = content
        
        # Remove common PII patterns
        import re
        anonymized_content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', anonymized_content)
        anonymized_content = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', anonymized_content)
        anonymized_content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', anonymized_content)
        
        # Anonymize metadata
        anonymized_metadata = metadata.copy()
        sensitive_keys = ['user_id', 'email', 'phone', 'address', 'ip_address']
        for key in sensitive_keys:
            if key in anonymized_metadata:
                anonymized_metadata[key] = '[REDACTED]'
        
        return anonymized_content, anonymized_metadata
    
    def register_project(self, project_name: str, description: str = "") -> Dict[str, Any]:
        """Register a new project with the hub."""
        data = {
            "name": project_name,
            "description": description,
            "access_level": self.config.get("projects", {}).get("default_access_level", "read_write")
        }
        
        response = self._make_request("POST", "/projects/register", data)
        self.project_id = response.get("project_id")
        console.print(f"[green]✓[/green] Registered project: {project_name} (ID: {self.project_id})")
        return response
    
    def store_item(self, item: HubItem) -> Dict[str, Any]:
        """Store an item in the hub."""
        # Anonymize if configured
        if self.config.get("sync_policies", {}).get("anonymize_pii", True):
            content, metadata = self._anonymize_content(item.content, item.metadata)
        else:
            content, metadata = item.content, item.metadata
        
        # Get embedding
        embedding = item.embedding or self._get_embedding(content)
        
        # Prepare payload
        payload = {
            "content": content,
            "content_type": item.content_type,
            "metadata": {
                **metadata,
                "project_id": self.project_id,
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": self.embedding_dimension,
                "created_at": time.time(),
                "source": "gola_pipeline"
            },
            "embedding": embedding
        }
        
        if item.id:
            payload["id"] = item.id
        
        response = self._make_request("POST", "/store", payload)
        console.print(f"[green]✓[/green] Stored {item.content_type}: {item.id or response.get('id')}")
        return response
    
    def search(self, query: str, content_types: Optional[List[str]] = None, 
               limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search the hub for relevant content."""
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        data = {
            "query": query,
            "embedding": query_embedding,
            "limit": limit,
            "threshold": threshold,
            "project_id": self.project_id
        }
        
        if content_types:
            data["content_types"] = content_types
        
        response = self._make_request("POST", "/search", data)
        results = response.get("results", [])
        
        console.print(f"[blue]Found {len(results)} results for query: {query}[/blue]")
        return results
    
    def get_project_context(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get context and patterns for a project."""
        project_id = project_id or self.project_id
        if not project_id:
            raise ValueError("No project ID specified")
        
        response = self._make_request("GET", f"/projects/{project_id}/context")
        return response
    
    def get_stats(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a project or global stats."""
        endpoint = f"/stats/project/{project_id}" if project_id else "/stats"
        response = self._make_request("GET", endpoint)
        return response
    
    def sync_dataset_insights(self, dataset_slug: str, dataset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync dataset insights to the hub."""
        insights = []
        
        # Extract patterns from successful runs
        if "runs" in dataset_data:
            successful_runs = [run for run in dataset_data["runs"] if run.get("status") == "ok"]
            
            # Pattern: Successful model combinations
            model_combinations = {}
            for run in successful_runs:
                key = f"{run.get('provider')}/{run.get('model_id')}"
                model_combinations[key] = model_combinations.get(key, 0) + 1
            
            for combo, count in model_combinations.items():
                if count >= 3:  # Only patterns with multiple successes
                    insights.append(HubItem(
                        content=f"Model combination {combo} worked well for {dataset_slug}",
                        content_type="pattern",
                        metadata={
                            "dataset": dataset_slug,
                            "model_combo": combo,
                            "success_count": count,
                            "task_types": list(set(run.get("task_type") for run in successful_runs if run.get("provider") + "/" + run.get("model_id") == combo))
                        }
                    ))
        
        # Extract concepts from entity/triple analysis
        if "outputs" in dataset_data:
            entities = dataset_data["outputs"].get("entities", [])
            if entities:
                # Concept: Common entity types
                entity_types = {}
                for entity in entities:
                    entity_type = entity.get("entity_type", "unknown")
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                
                for entity_type, count in entity_types.items():
                    if count >= 5:  # Only significant patterns
                        insights.append(HubItem(
                            content=f"Entity type '{entity_type}' appears frequently in {dataset_slug}",
                            content_type="concept",
                            metadata={
                                "dataset": dataset_slug,
                                "entity_type": entity_type,
                                "frequency": count,
                                "total_entities": len(entities)
                            }
                        ))
        
        # Store insights
        results = []
        for insight in insights:
            try:
                result = self.store_item(insight)
                results.append(result)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to store insight: {e}[/yellow]")
        
        console.print(f"[green]✓[/green] Synced {len(results)} insights from dataset {dataset_slug}")
        return {"synced_count": len(results), "insights": results}
    
    def get_development_coordinates(self, query: str) -> Dict[str, Any]:
        """Get development coordinates for a query."""
        data = {
            "query": query,
            "project_id": self.project_id
        }
        
        response = self._make_request("POST", "/development/coordinates", data)
        return response 
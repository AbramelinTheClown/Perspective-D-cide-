"""
Main ETX framework for emergent taxonomy discovery.

Provides the core functionality for ingesting content, discovering categories,
and exporting results using embedding and clustering techniques.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from ..core.schemas import ContentItem, CategoryProposal, AnalysisResult
from ..core.config import get_config
from .engines import EmbeddingEngine, ClusteringEngine
from .builders import CategorizationBuilder

class ETXFramework:
    """
    Emergent TaXonomy framework for dynamic content categorization.
    
    Uses embedding and clustering to discover categories from content
    without predefined taxonomies.
    """
    
    def __init__(self, project: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ETX framework.
        
        Args:
            project: Project name for organizing data
            config: Framework configuration
        """
        self.project = project
        self.config = config or get_config().etx_config
        
        # Initialize engines
        self.embedding_engine = self._create_embedding_engine()
        self.clustering_engine = self._create_clustering_engine()
        
        # Initialize builder
        self.builder = CategorizationBuilder(self.config)
        
        # Storage for content and results
        self.content_items: List[ContentItem] = []
        self.categories: List[CategoryProposal] = []
        self.analysis_results: List[AnalysisResult] = []
        
        # Processing state
        self._embeddings: Optional[List[List[float]]] = None
        self._clusters: Optional[List[int]] = None
    
    def _create_embedding_engine(self) -> EmbeddingEngine:
        """Create embedding engine based on configuration."""
        model_name = self.config.get("embedding_model", "bge-small-en")
        
        try:
            from .engines import FastEmbedEngine
            return FastEmbedEngine(model_name=model_name)
        except ImportError:
            # Fallback to simple engine
            from .engines import SimpleEmbeddingEngine
            return SimpleEmbeddingEngine()
    
    def _create_clustering_engine(self) -> ClusteringEngine:
        """Create clustering engine based on configuration."""
        clustering_type = self.config.get("clustering_type", "minibatch_kmeans")
        
        try:
            from .engines import MiniBatchKMeansEngine
            return MiniBatchKMeansEngine()
        except ImportError:
            # Fallback to simple engine
            from .engines import SimpleClusteringEngine
            return SimpleClusteringEngine()
    
    def ingest(self, source_path: Union[str, Path], modality: str = "text") -> None:
        """
        Ingest content from a source file.
        
        Args:
            source_path: Path to source file (JSONL format)
            modality: Content modality (text, image, etc.)
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Read JSONL file
        with open(source_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        
                        # Create content item
                        content_item = ContentItem(
                            id=data.get('id', f"item_{line_num}"),
                            content=data.get('text', data.get('content', '')),
                            metadata={
                                'source_file': str(source_path),
                                'line_number': line_num,
                                'modality': modality,
                                **data.get('metadata', {})
                            }
                        )
                        
                        self.content_items.append(content_item)
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON at line {line_num}: {e}")
        
        print(f"Ingested {len(self.content_items)} content items from {source_path}")
    
    def discover(self) -> None:
        """
        Discover categories using embedding and clustering.
        """
        if not self.content_items:
            raise ValueError("No content items to process. Call ingest() first.")
        
        print("Starting category discovery...")
        
        # Extract text content
        texts = [item.content for item in self.content_items]
        
        # Generate embeddings
        print("Generating embeddings...")
        self._embeddings = self.embedding_engine.embed(texts)
        
        # Perform clustering
        print("Performing clustering...")
        self._clusters = self.clustering_engine.cluster(self._embeddings)
        
        # Build categories
        print("Building categories...")
        self.categories = self.builder.build_categories(
            content_items=self.content_items,
            embeddings=self._embeddings,
            clusters=self._clusters
        )
        
        # Create analysis results
        self.analysis_results = []
        for i, item in enumerate(self.content_items):
            cluster_id = self._clusters[i] if self._clusters else 0
            
            # Find category for this cluster
            category = next((cat for cat in self.categories if cat.metadata.get('cluster_id') == cluster_id), None)
            
            result = AnalysisResult(
                content_id=item.id,
                analysis_type="categorization",
                results={
                    'cluster_id': cluster_id,
                    'category': category.category_name if category else 'unknown',
                    'confidence': category.confidence if category else 0.0
                },
                confidence=category.confidence if category else 0.0,
                metadata={
                    'cluster_id': cluster_id,
                    'embedding_dim': len(self._embeddings[i]) if self._embeddings else 0
                }
            )
            
            self.analysis_results.append(result)
        
        print(f"Discovery complete. Found {len(self.categories)} categories.")
    
    def export(self, topic: str = "any", as_df: bool = True) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Export analysis results.
        
        Args:
            topic: Topic filter (not used in current implementation)
            as_df: Whether to return as pandas DataFrame
            
        Returns:
            Analysis results as DataFrame or list of dictionaries
        """
        if not self.analysis_results:
            raise ValueError("No analysis results to export. Call discover() first.")
        
        # Convert to list of dictionaries
        results = []
        for result in self.analysis_results:
            result_dict = {
                'content_id': result.content_id,
                'category': result.results.get('category', 'unknown'),
                'confidence': result.confidence,
                'cluster_id': result.results.get('cluster_id', 0),
                'analysis_type': result.analysis_type,
                'created_at': result.created_at.isoformat()
            }
            results.append(result_dict)
        
        if as_df:
            return pd.DataFrame(results)
        else:
            return results
    
    def get_categories(self) -> List[CategoryProposal]:
        """Get discovered categories."""
        return self.categories
    
    def get_content_by_category(self, category_name: str) -> List[ContentItem]:
        """Get content items for a specific category."""
        category_ids = [cat.category_name for cat in self.categories if cat.category_name == category_name]
        
        if not category_ids:
            return []
        
        # Find content items in this category
        content_items = []
        for i, result in enumerate(self.analysis_results):
            if result.results.get('category') == category_name:
                content_items.append(self.content_items[i])
        
        return content_items 
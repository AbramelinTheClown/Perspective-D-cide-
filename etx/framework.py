"""
ETX Framework for Perspective D<cide>.

Main framework for Emergent Taxonomy analysis and content categorization.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ETXConfig:
    """Configuration for ETX framework."""
    
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    clustering_type: str = "minibatch_kmeans"
    n_clusters: int = 10
    min_confidence: float = 0.7
    min_keywords: int = 3
    batch_size: int = 1000
    enable_plugins: bool = True

@dataclass
class ETXResult:
    """Result of ETX analysis."""
    
    categories: List[Dict[str, Any]]
    embeddings: Optional[np.ndarray] = None
    clusters: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ETXFramework:
    """Main ETX framework for content analysis and categorization."""
    
    def __init__(self, config: Optional[ETXConfig] = None):
        """
        Initialize ETX framework.
        
        Args:
            config: Framework configuration
        """
        self.config = config or ETXConfig()
        self.embedding_engine = None
        self.clustering_engine = None
        self.categorization_builder = None
        self.plugin_manager = None
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize framework components."""
        try:
            # Initialize embedding engine
            from .engines import FastEmbedEngine
            self.embedding_engine = FastEmbedEngine(self.config.embedding_model)
            logger.info(f"Initialized embedding engine: {self.config.embedding_model}")
        except ImportError as e:
            logger.warning(f"Could not initialize embedding engine: {e}")
        
        try:
            # Initialize clustering engine
            if self.config.clustering_type == "minibatch_kmeans":
                from .engines import MiniBatchKMeansEngine
                self.clustering_engine = MiniBatchKMeansEngine(
                    n_clusters=self.config.n_clusters,
                    batch_size=self.config.batch_size
                )
            elif self.config.clustering_type == "hdbscan":
                from .engines import HDBSCANEngine
                self.clustering_engine = HDBSCANEngine()
            else:
                from .engines import MiniBatchKMeansEngine
                self.clustering_engine = MiniBatchKMeansEngine(
                    n_clusters=self.config.n_clusters
                )
            logger.info(f"Initialized clustering engine: {self.config.clustering_type}")
        except ImportError as e:
            logger.warning(f"Could not initialize clustering engine: {e}")
        
        try:
            # Initialize categorization builder
            from .builders import CategorizationBuilder
            self.categorization_builder = CategorizationBuilder(
                min_confidence=self.config.min_confidence,
                min_keywords=self.config.min_keywords
            )
            logger.info("Initialized categorization builder")
        except ImportError as e:
            logger.warning(f"Could not initialize categorization builder: {e}")
        
        if self.config.enable_plugins:
            try:
                # Initialize plugin manager
                from .plugins import PluginManager
                self.plugin_manager = PluginManager()
                logger.info("Initialized plugin manager")
            except ImportError as e:
                logger.warning(f"Could not initialize plugin manager: {e}")
    
    def analyze_content(self, content: List[str]) -> ETXResult:
        """
        Analyze content using the ETX framework.
        
        Args:
            content: List of text content to analyze
            
        Returns:
            ETXResult with categories and analysis
        """
        if not content:
            return ETXResult(categories=[], metadata={"error": "No content provided"})
        
        try:
            # Step 1: Generate embeddings
            embeddings = None
            if self.embedding_engine:
                embedding_result = self.embedding_engine.embed(content)
                embeddings = embedding_result.embeddings
                logger.info(f"Generated embeddings: {embeddings.shape}")
            
            # Step 2: Perform clustering
            clusters = None
            if self.clustering_engine and embeddings is not None:
                clustering_result = self.clustering_engine.cluster(embeddings)
                clusters = clustering_result.labels
                logger.info(f"Generated clusters: {len(set(clusters))} unique clusters")
            
            # Step 3: Build taxonomy
            categories = []
            if self.categorization_builder and embeddings is not None and clusters is not None:
                taxonomy_result = self.categorization_builder.build_taxonomy(
                    embeddings, clusters, content, clustering_result.centroids
                )
                categories = [
                    {
                        "id": cat.id,
                        "name": cat.name,
                        "description": cat.description,
                        "keywords": cat.keywords,
                        "confidence": cat.confidence,
                        "parent_id": cat.parent_id,
                        "children": cat.children,
                        "metadata": cat.metadata
                    }
                    for cat in taxonomy_result.categories
                ]
                logger.info(f"Built taxonomy: {len(categories)} categories")
            
            # Step 4: Run plugins (optional)
            plugin_results = {}
            if self.plugin_manager:
                plugin_results = self.plugin_manager.process_with_plugins(content)
                logger.info(f"Processed with plugins: {list(plugin_results.keys())}")
            
            metadata = {
                "content_count": len(content),
                "embedding_model": self.config.embedding_model,
                "clustering_type": self.config.clustering_type,
                "n_clusters": self.config.n_clusters,
                "plugin_results": plugin_results
            }
            
            return ETXResult(
                categories=categories,
                embeddings=embeddings,
                clusters=clusters,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"ETX analysis failed: {e}")
            return ETXResult(
                categories=[],
                metadata={"error": str(e)}
            )
    
    def get_categories(self, content: List[str]) -> List[Dict[str, Any]]:
        """
        Get categories for content.
        
        Args:
            content: List of text content
            
        Returns:
            List of category dictionaries
        """
        result = self.analyze_content(content)
        return result.categories
    
    def get_embeddings(self, content: List[str]) -> Optional[np.ndarray]:
        """
        Get embeddings for content.
        
        Args:
            content: List of text content
            
        Returns:
            Embeddings array or None
        """
        if not self.embedding_engine:
            return None
        
        try:
            embedding_result = self.embedding_engine.embed(content)
            return embedding_result.embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def get_clusters(self, content: List[str]) -> Optional[np.ndarray]:
        """
        Get clusters for content.
        
        Args:
            content: List of text content
            
        Returns:
            Cluster labels array or None
        """
        if not self.clustering_engine:
            return None
        
        embeddings = self.get_embeddings(content)
        if embeddings is None:
            return None
        
        try:
            clustering_result = self.clustering_engine.cluster(embeddings)
            return clustering_result.labels
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return None
    
    def export_results(self, result: ETXResult, format: str = "json") -> str:
        """
        Export ETX results to various formats.
        
        Args:
            result: ETX analysis result
            format: Export format ("json", "csv")
            
        Returns:
            Exported data as string
        """
        if format == "json":
            return self._export_json(result)
        elif format == "csv":
            return self._export_csv(result)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, result: ETXResult) -> str:
        """Export results as JSON."""
        import json
        
        data = {
            "categories": result.categories,
            "metadata": result.metadata,
            "has_embeddings": result.embeddings is not None,
            "has_clusters": result.clusters is not None
        }
        
        return json.dumps(data, indent=2)
    
    def _export_csv(self, result: ETXResult) -> str:
        """Export results as CSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(["ID", "Name", "Description", "Keywords", "Confidence", "Parent", "Children"])
        
        # Data
        for category in result.categories:
            writer.writerow([
                category["id"],
                category["name"],
                category["description"],
                "; ".join(category["keywords"]),
                f"{category['confidence']:.3f}",
                category["parent_id"] or "",
                "; ".join(category["children"])
            ])
        
        return output.getvalue() 
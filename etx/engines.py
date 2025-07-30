"""
ETX Engines for Perspective D<cide>.

Provides embedding and clustering engines for the Emergent Taxonomy framework.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    
    embeddings: np.ndarray
    metadata: Dict[str, Any]
    model_name: str
    dimensions: int

@dataclass
class ClusteringResult:
    """Result of clustering operation."""
    
    labels: np.ndarray
    centroids: np.ndarray
    n_clusters: int
    metadata: Dict[str, Any]

class BaseEmbeddingEngine:
    """Base class for embedding engines."""
    
    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self.dimensions = 384  # Default dimensions
        
    def embed(self, texts: List[str]) -> EmbeddingResult:
        """Embed a list of texts."""
        raise NotImplementedError("Subclasses must implement embed()")
    
    def get_dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self.dimensions

class FastEmbedEngine(BaseEmbeddingEngine):
    """Fast embedding engine using sentence-transformers."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        super().__init__(model_name)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimensions = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {self.model_name} ({self.dimensions} dimensions)")
        except ImportError:
            logger.warning("sentence-transformers not available, using dummy embeddings")
            self.model = None
    
    def embed(self, texts: List[str]) -> EmbeddingResult:
        """Embed texts using sentence-transformers."""
        if self.model is None:
            # Fallback to dummy embeddings
            embeddings = np.random.rand(len(texts), self.dimensions)
            return EmbeddingResult(
                embeddings=embeddings,
                metadata={"method": "dummy", "model": "random"},
                model_name="dummy",
                dimensions=self.dimensions
            )
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return EmbeddingResult(
                embeddings=embeddings,
                metadata={"method": "sentence_transformers", "model": self.model_name},
                model_name=self.model_name,
                dimensions=self.dimensions
            )
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # Fallback to dummy embeddings
            embeddings = np.random.rand(len(texts), self.dimensions)
            return EmbeddingResult(
                embeddings=embeddings,
                metadata={"method": "dummy", "error": str(e)},
                model_name="dummy",
                dimensions=self.dimensions
            )

class BaseClusteringEngine:
    """Base class for clustering engines."""
    
    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters
    
    def cluster(self, embeddings: np.ndarray) -> ClusteringResult:
        """Cluster embeddings."""
        raise NotImplementedError("Subclasses must implement cluster()")

class MiniBatchKMeansEngine(BaseClusteringEngine):
    """Mini-batch K-means clustering engine."""
    
    def __init__(self, n_clusters: int = 10, batch_size: int = 1000):
        super().__init__(n_clusters)
        self.batch_size = batch_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the clustering model."""
        try:
            from sklearn.cluster import MiniBatchKMeans
            self.model = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=self.batch_size,
                random_state=42
            )
            logger.info(f"Loaded MiniBatchKMeans with {self.n_clusters} clusters")
        except ImportError:
            logger.warning("scikit-learn not available, using dummy clustering")
            self.model = None
    
    def cluster(self, embeddings: np.ndarray) -> ClusteringResult:
        """Cluster embeddings using MiniBatchKMeans."""
        if self.model is None:
            # Fallback to dummy clustering
            n_samples = embeddings.shape[0]
            labels = np.random.randint(0, self.n_clusters, n_samples)
            centroids = np.random.rand(self.n_clusters, embeddings.shape[1])
            return ClusteringResult(
                labels=labels,
                centroids=centroids,
                n_clusters=self.n_clusters,
                metadata={"method": "dummy", "model": "random"}
            )
        
        try:
            labels = self.model.fit_predict(embeddings)
            centroids = self.model.cluster_centers_
            return ClusteringResult(
                labels=labels,
                centroids=centroids,
                n_clusters=self.n_clusters,
                metadata={"method": "minibatch_kmeans", "batch_size": self.batch_size}
            )
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            # Fallback to dummy clustering
            n_samples = embeddings.shape[0]
            labels = np.random.randint(0, self.n_clusters, n_samples)
            centroids = np.random.rand(self.n_clusters, embeddings.shape[1])
            return ClusteringResult(
                labels=labels,
                centroids=centroids,
                n_clusters=self.n_clusters,
                metadata={"method": "dummy", "error": str(e)}
            )

class HDBSCANEngine(BaseClusteringEngine):
    """HDBSCAN clustering engine for density-based clustering."""
    
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 5):
        super().__init__(n_clusters=0)  # HDBSCAN determines number of clusters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the HDBSCAN model."""
        try:
            import hdbscan
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples
            )
            logger.info("Loaded HDBSCAN clustering model")
        except ImportError:
            logger.warning("hdbscan not available, using dummy clustering")
            self.model = None
    
    def cluster(self, embeddings: np.ndarray) -> ClusteringResult:
        """Cluster embeddings using HDBSCAN."""
        if self.model is None:
            # Fallback to dummy clustering
            n_samples = embeddings.shape[0]
            n_clusters = max(1, n_samples // 10)  # Rough estimate
            labels = np.random.randint(0, n_clusters, n_samples)
            centroids = np.random.rand(n_clusters, embeddings.shape[1])
            return ClusteringResult(
                labels=labels,
                centroids=centroids,
                n_clusters=n_clusters,
                metadata={"method": "dummy", "model": "random"}
            )
        
        try:
            labels = self.model.fit_predict(embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise (-1)
            
            # Calculate centroids for non-noise clusters
            centroids = []
            for i in range(n_clusters):
                cluster_mask = labels == i
                if np.any(cluster_mask):
                    centroid = np.mean(embeddings[cluster_mask], axis=0)
                    centroids.append(centroid)
            
            if centroids:
                centroids = np.array(centroids)
            else:
                centroids = np.random.rand(1, embeddings.shape[1])
            
            return ClusteringResult(
                labels=labels,
                centroids=centroids,
                n_clusters=n_clusters,
                metadata={
                    "method": "hdbscan",
                    "min_cluster_size": self.min_cluster_size,
                    "min_samples": self.min_samples,
                    "noise_points": np.sum(labels == -1)
                }
            )
        except Exception as e:
            logger.error(f"HDBSCAN clustering failed: {e}")
            # Fallback to dummy clustering
            n_samples = embeddings.shape[0]
            n_clusters = max(1, n_samples // 10)
            labels = np.random.randint(0, n_clusters, n_samples)
            centroids = np.random.rand(n_clusters, embeddings.shape[1])
            return ClusteringResult(
                labels=labels,
                centroids=centroids,
                n_clusters=n_clusters,
                metadata={"method": "dummy", "error": str(e)}
            ) 
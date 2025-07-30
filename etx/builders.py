"""
ETX Builders for Perspective D<cide>.

Provides categorization building functionality for the Emergent Taxonomy framework.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CategoryNode:
    """Represents a category in the taxonomy."""
    
    id: str
    name: str
    description: str
    keywords: List[str]
    confidence: float
    parent_id: Optional[str] = None
    children: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TaxonomyResult:
    """Result of taxonomy building operation."""
    
    categories: List[CategoryNode]
    hierarchy: Dict[str, List[str]]
    metadata: Dict[str, Any]
    confidence_scores: Dict[str, float]

class CategorizationBuilder:
    """Builds emergent taxonomies from clustered content."""
    
    def __init__(self, min_confidence: float = 0.7, min_keywords: int = 3):
        """
        Initialize the categorization builder.
        
        Args:
            min_confidence: Minimum confidence threshold for categories
            min_keywords: Minimum number of keywords required
        """
        self.min_confidence = min_confidence
        self.min_keywords = min_keywords
        self.category_templates = self._load_category_templates()
    
    def _load_category_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load category templates for different content types."""
        return {
            "technical": {
                "keywords": ["code", "programming", "software", "development", "technology"],
                "description_template": "Technical content related to {keywords}"
            },
            "business": {
                "keywords": ["business", "management", "strategy", "marketing", "finance"],
                "description_template": "Business content covering {keywords}"
            },
            "academic": {
                "keywords": ["research", "study", "analysis", "academic", "scholarly"],
                "description_template": "Academic content including {keywords}"
            },
            "creative": {
                "keywords": ["creative", "art", "design", "writing", "media"],
                "description_template": "Creative content featuring {keywords}"
            },
            "general": {
                "keywords": ["general", "information", "overview", "introduction"],
                "description_template": "General content about {keywords}"
            }
        }
    
    def build_taxonomy(self, 
                      embeddings: np.ndarray, 
                      labels: np.ndarray,
                      texts: List[str],
                      centroids: np.ndarray) -> TaxonomyResult:
        """
        Build a taxonomy from clustering results.
        
        Args:
            embeddings: Content embeddings
            labels: Cluster labels
            texts: Original text content
            centroids: Cluster centroids
            
        Returns:
            TaxonomyResult with categories and hierarchy
        """
        try:
            # Create category nodes from clusters
            categories = []
            hierarchy = {}
            confidence_scores = {}
            
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                
                # Get texts for this cluster
                cluster_mask = labels == label
                cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
                
                if len(cluster_texts) < 2:  # Skip very small clusters
                    continue
                
                # Analyze cluster content
                category_info = self._analyze_cluster_content(cluster_texts)
                
                # Calculate confidence
                confidence = self._calculate_cluster_confidence(
                    embeddings[cluster_mask], 
                    centroids[label] if label < len(centroids) else None
                )
                
                if confidence < self.min_confidence:
                    continue
                
                # Create category node
                category = CategoryNode(
                    id=f"category_{label}",
                    name=category_info["name"],
                    description=category_info["description"],
                    keywords=category_info["keywords"],
                    confidence=confidence,
                    metadata={
                        "cluster_id": label,
                        "text_count": len(cluster_texts),
                        "content_type": category_info["content_type"]
                    }
                )
                
                categories.append(category)
                confidence_scores[category.id] = confidence
                hierarchy[category.id] = []
            
            # Build hierarchy (simple approach - could be enhanced)
            self._build_hierarchy(categories, embeddings, labels, centroids)
            
            return TaxonomyResult(
                categories=categories,
                hierarchy=hierarchy,
                metadata={
                    "total_categories": len(categories),
                    "min_confidence": self.min_confidence,
                    "min_keywords": self.min_keywords
                },
                confidence_scores=confidence_scores
            )
            
        except Exception as e:
            logger.error(f"Taxonomy building failed: {e}")
            # Return empty taxonomy
            return TaxonomyResult(
                categories=[],
                hierarchy={},
                metadata={"error": str(e)},
                confidence_scores={}
            )
    
    def _analyze_cluster_content(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze content to determine category information."""
        # Simple keyword extraction (could be enhanced with NLP)
        all_text = " ".join(texts).lower()
        
        # Count common words
        words = all_text.split()
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top keywords
        keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        keyword_list = [word for word, count in keywords if count > 1]
        
        # Determine content type
        content_type = self._classify_content_type(keyword_list)
        
        # Generate name and description
        name = self._generate_category_name(keyword_list, content_type)
        description = self._generate_category_description(keyword_list, content_type)
        
        return {
            "name": name,
            "description": description,
            "keywords": keyword_list[:self.min_keywords],
            "content_type": content_type
        }
    
    def _classify_content_type(self, keywords: List[str]) -> str:
        """Classify content type based on keywords."""
        for content_type, template in self.category_templates.items():
            template_keywords = template["keywords"]
            matches = sum(1 for kw in keywords if any(tk in kw for tk in template_keywords))
            if matches >= 2:
                return content_type
        return "general"
    
    def _generate_category_name(self, keywords: List[str], content_type: str) -> str:
        """Generate a category name from keywords."""
        if not keywords:
            return f"{content_type.title()} Content"
        
        # Use the most prominent keyword
        main_keyword = keywords[0].title()
        return f"{main_keyword} {content_type.title()}"
    
    def _generate_category_description(self, keywords: List[str], content_type: str) -> str:
        """Generate a category description."""
        template = self.category_templates[content_type]["description_template"]
        keyword_str = ", ".join(keywords[:3])
        return template.format(keywords=keyword_str)
    
    def _calculate_cluster_confidence(self, 
                                    cluster_embeddings: np.ndarray, 
                                    centroid: Optional[np.ndarray]) -> float:
        """Calculate confidence score for a cluster."""
        if centroid is None or len(cluster_embeddings) == 0:
            return 0.0
        
        # Calculate average distance to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        avg_distance = np.mean(distances)
        
        # Convert to confidence (lower distance = higher confidence)
        # Normalize to 0-1 range
        max_expected_distance = 2.0  # Empirical threshold
        confidence = max(0.0, 1.0 - (avg_distance / max_expected_distance))
        
        return min(1.0, confidence)
    
    def _build_hierarchy(self, 
                        categories: List[CategoryNode],
                        embeddings: np.ndarray,
                        labels: np.ndarray,
                        centroids: np.ndarray) -> None:
        """Build hierarchical relationships between categories."""
        if len(categories) < 2:
            return
        
        # Simple hierarchical clustering based on centroid similarity
        for i, category1 in enumerate(categories):
            for j, category2 in enumerate(categories[i+1:], i+1):
                # Get centroids
                centroid1_idx = category1.metadata.get("cluster_id", 0)
                centroid2_idx = category2.metadata.get("cluster_id", 0)
                
                if centroid1_idx < len(centroids) and centroid2_idx < len(centroids):
                    centroid1 = centroids[centroid1_idx]
                    centroid2 = centroids[centroid2_idx]
                    
                    # Calculate similarity
                    similarity = self._calculate_centroid_similarity(centroid1, centroid2)
                    
                    # If similar enough, create parent-child relationship
                    if similarity > 0.8:  # High similarity threshold
                        # Determine which should be parent (larger cluster)
                        if category1.metadata["text_count"] > category2.metadata["text_count"]:
                            category1.children.append(category2.id)
                            category2.parent_id = category1.id
                        else:
                            category2.children.append(category1.id)
                            category1.parent_id = category2.id
    
    def _calculate_centroid_similarity(self, centroid1: np.ndarray, centroid2: np.ndarray) -> float:
        """Calculate similarity between two centroids."""
        # Cosine similarity
        dot_product = np.dot(centroid1, centroid2)
        norm1 = np.linalg.norm(centroid1)
        norm2 = np.linalg.norm(centroid2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def export_taxonomy(self, taxonomy: TaxonomyResult, format: str = "json") -> str:
        """Export taxonomy to various formats."""
        if format == "json":
            return self._export_json(taxonomy)
        elif format == "csv":
            return self._export_csv(taxonomy)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, taxonomy: TaxonomyResult) -> str:
        """Export taxonomy as JSON."""
        import json
        
        data = {
            "categories": [
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
                for cat in taxonomy.categories
            ],
            "hierarchy": taxonomy.hierarchy,
            "metadata": taxonomy.metadata,
            "confidence_scores": taxonomy.confidence_scores
        }
        
        return json.dumps(data, indent=2)
    
    def _export_csv(self, taxonomy: TaxonomyResult) -> str:
        """Export taxonomy as CSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(["ID", "Name", "Description", "Keywords", "Confidence", "Parent", "Children"])
        
        # Data
        for category in taxonomy.categories:
            writer.writerow([
                category.id,
                category.name,
                category.description,
                "; ".join(category.keywords),
                f"{category.confidence:.3f}",
                category.parent_id or "",
                "; ".join(category.children)
            ])
        
        return output.getvalue() 
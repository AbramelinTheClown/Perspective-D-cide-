"""
Symbolic system integration with the main framework.

Provides integration between the glyph/tarot symbolic system and the
main content analysis framework.
"""

from typing import Dict, List, Optional, Any
from ..core.schemas import ContentItem, SymbolicAnalysisResult
from ..core.config import Config

# Import with error handling
try:
    from .glyphs import GlyphLookup
except ImportError:
    GlyphLookup = None

try:
    from .logic import SymbolicCollapse, TarotMapping
except ImportError:
    SymbolicCollapse = None
    TarotMapping = None

def initialize_symbolic_system(config: Config) -> None:
    """
    Initialize the symbolic system with framework configuration.
    
    Args:
        config: Framework configuration
    """
    if not all([GlyphLookup, SymbolicCollapse, TarotMapping]):
        print("Warning: Some symbolic components are not available, skipping initialization")
        return
    
    symbolic_config = config.symbolic_config
    
    # Initialize components
    glyph_lookup = GlyphLookup(symbolic_config.get("glyphs_path"))
    collapse_engine = SymbolicCollapse(symbolic_config.get("glyphs_path"))
    tarot_mapping = TarotMapping(
        symbolic_system_path=symbolic_config.get("symbolic_system_path"),
        glyphs_path=symbolic_config.get("glyphs_path")
    )
    
    # Register with component registry
    try:
        from ..core.registry import get_registry
        registry = get_registry()
        
        registry.register_component("glyph_lookup", glyph_lookup)
        registry.register_component("collapse_engine", collapse_engine)
        registry.register_component("tarot_mapping", tarot_mapping)
        
        print("Symbolic system initialized successfully")
    except Exception as e:
        print(f"Warning: Could not register symbolic components: {e}")

class SymbolicAnalyzer:
    """
    Analyzer that combines glyph and tarot symbolic reasoning with content analysis.
    
    Integrates the symbolic system with the main framework for enhanced
    content understanding and categorization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize symbolic analyzer.
        
        Args:
            config: Configuration for symbolic analysis
        """
        self.config = config or {}
        
        # Initialize components with error handling
        try:
            from ..core.registry import get_registry
            registry = get_registry()
            
            self.glyph_lookup = registry.get_component("glyph_lookup")
            self.collapse_engine = registry.get_component("collapse_engine")
            self.tarot_mapping = registry.get_component("tarot_mapping")
        except Exception as e:
            print(f"Warning: Could not initialize symbolic analyzer components: {e}")
            self.glyph_lookup = None
            self.collapse_engine = None
            self.tarot_mapping = None
    
    def analyze_content(self, content_item: ContentItem) -> SymbolicAnalysisResult:
        """
        Analyze content using symbolic reasoning.
        
        Args:
            content_item: Content to analyze
            
        Returns:
            Symbolic analysis result
        """
        if not all([self.glyph_lookup, self.collapse_engine, self.tarot_mapping]):
            # Return empty result if components not available
            return SymbolicAnalysisResult(
                content_id=content_item.id,
                analysis_type="symbolic_reasoning",
                results={"error": "Symbolic components not available"},
                confidence=0.0
            )
        
        content_text = content_item.content.lower()
        
        # Extract keywords and concepts
        keywords = self._extract_keywords(content_text)
        
        # Find relevant glyphs
        relevant_glyphs = []
        for keyword in keywords:
            glyphs = self.glyph_lookup.search_by_name(keyword)
            relevant_glyphs.extend(glyphs)
        
        # Perform symbolic collapse
        if keywords:
            collapse_result = self.collapse_engine.collapse_by_tags(keywords)
        else:
            collapse_result = {"confidence": 0.0, "recommendations": []}
        
        # Find tarot correspondences
        tarot_cards = []
        archetypal_theme = None
        
        if relevant_glyphs:
            # Use first relevant glyph for tarot mapping
            glyph = relevant_glyphs[0]
            tarot_path = glyph.tarot
            if tarot_path:
                tarot_cards.append(tarot_path)
                archetypal_theme = glyph.archetype
        
        # Build results
        results = {
            "keywords": keywords,
            "relevant_glyphs": [g.name for g in relevant_glyphs],
            "collapse_result": collapse_result,
            "archetypal_theme": archetypal_theme
        }
        
        return SymbolicAnalysisResult(
            content_id=content_item.id,
            analysis_type="symbolic_reasoning",
            results=results,
            confidence=collapse_result.get("confidence", 0.0),
            glyphs_used=[g.name for g in relevant_glyphs],
            tarot_cards=tarot_cards,
            collapse_state=collapse_result,
            archetypal_theme=archetypal_theme
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction - could be enhanced with NLP
        words = text.split()
        # Filter out common words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 3 and word.lower() not in stop_words]
        return keywords[:10]  # Limit to top 10 keywords
    
    def get_symbolic_categories(self, content_items: List[ContentItem]) -> List[Dict[str, Any]]:
        """
        Get symbolic categories for multiple content items.
        
        Args:
            content_items: List of content items to categorize
            
        Returns:
            List of symbolic categories
        """
        categories = []
        
        for item in content_items:
            analysis = self.analyze_content(item)
            categories.append({
                "content_id": item.id,
                "archetypal_theme": analysis.archetypal_theme,
                "glyphs": analysis.glyphs_used,
                "tarot_cards": analysis.tarot_cards,
                "confidence": analysis.confidence
            })
        
        return categories 
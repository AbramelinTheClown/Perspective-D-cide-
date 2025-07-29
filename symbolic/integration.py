"""
Symbolic system integration with the main framework.

Provides integration between the glyph/tarot symbolic system and the
main content analysis framework.
"""

from typing import Dict, List, Optional, Any
from ..core.schemas import ContentItem, SymbolicAnalysisResult
from ..core.config import Config
from .glyphs import GlyphLookup
from .logic import SymbolicCollapse, TarotMapping

def initialize_symbolic_system(config: Config) -> None:
    """
    Initialize the symbolic system with framework configuration.
    
    Args:
        config: Framework configuration
    """
    symbolic_config = config.symbolic_config
    
    # Initialize components
    glyph_lookup = GlyphLookup(symbolic_config.get("glyphs_path"))
    collapse_engine = SymbolicCollapse(symbolic_config.get("glyphs_path"))
    tarot_mapping = TarotMapping(
        symbolic_system_path=symbolic_config.get("symbolic_system_path"),
        glyphs_path=symbolic_config.get("glyphs_path")
    )
    
    # Register with component registry
    from ..core.registry import ComponentRegistry
    registry = ComponentRegistry.get_instance()
    
    registry.register_component("glyph_lookup", glyph_lookup)
    registry.register_component("collapse_engine", collapse_engine)
    registry.register_component("tarot_mapping", tarot_mapping)
    
    print("Symbolic system initialized successfully")

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
        from ..core.registry import ComponentRegistry
        registry = ComponentRegistry.get_instance()
        
        self.glyph_lookup = registry.get_component("glyph_lookup")
        self.collapse_engine = registry.get_component("collapse_engine")
        self.tarot_mapping = registry.get_component("tarot_mapping")
        
        self.config = config or {}
    
    def analyze_content(self, content_item: ContentItem) -> SymbolicAnalysisResult:
        """
        Analyze content using symbolic reasoning.
        
        Args:
            content_item: Content to analyze
            
        Returns:
            Symbolic analysis result
        """
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
            tarot_path = glyph.get('tarot')
            if tarot_path:
                card_data = self.tarot_mapping.get_card_by_number(tarot_path.split()[0])
                if card_data:
                    tarot_cards.append(card_data.get('title', ''))
                    archetypal_theme = self.tarot_mapping._extract_archetypal_theme(card_data)
        
        # Create symbolic analysis result
        result = SymbolicAnalysisResult(
            content_id=content_item.id,
            analysis_type="symbolic_reasoning",
            results={
                'keywords': keywords,
                'relevant_glyphs': [g['glyph_id'] for g in relevant_glyphs],
                'collapse_confidence': collapse_result.get('confidence', 0.0),
                'recommendations': collapse_result.get('recommendations', [])
            },
            confidence=collapse_result.get('confidence', 0.0),
            glyphs_used=[g['glyph_id'] for g in relevant_glyphs],
            tarot_cards=tarot_cards,
            collapse_state=collapse_result,
            archetypal_theme=archetypal_theme
        )
        
        return result
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for symbolic analysis.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - could be enhanced with NLP
        import re
        
        # Remove common words and punctuation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Return unique keywords (limit to top 10)
        return list(set(keywords))[:10]
    
    def get_symbolic_categories(self, content_items: List[ContentItem]) -> List[Dict[str, Any]]:
        """
        Get symbolic categories for a list of content items.
        
        Args:
            content_items: List of content items to categorize
            
        Returns:
            List of symbolic categories
        """
        categories = []
        
        for item in content_items:
            analysis = self.analyze_content(item)
            
            if analysis.archetypal_theme:
                category = {
                    'content_id': item.id,
                    'archetypal_theme': analysis.archetypal_theme,
                    'tarot_cards': analysis.tarot_cards,
                    'confidence': analysis.confidence,
                    'glyphs_used': analysis.glyphs_used
                }
                categories.append(category)
        
        return categories 
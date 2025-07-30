"""
Symbolic logic and reasoning for Perspective D<cide>.

Provides symbolic collapse and tarot mapping functionality.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CollapseResult:
    """Result of symbolic collapse operation."""
    
    confidence: float
    recommendations: List[str]
    archetypal_theme: Optional[str] = None
    energy_balance: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class SymbolicCollapse:
    """Manages symbolic collapse operations."""
    
    def __init__(self, glyphs_path: Optional[str] = None):
        """
        Initialize symbolic collapse system.
        
        Args:
            glyphs_path: Path to glyphs data
        """
        self.glyphs_path = glyphs_path
        self.collapse_patterns = self._load_collapse_patterns()
    
    def _load_collapse_patterns(self) -> Dict[str, Any]:
        """Load collapse patterns from configuration."""
        # Default collapse patterns
        return {
            "elemental_balance": {
                "fire": ["transformation", "passion", "energy"],
                "water": ["emotion", "flow", "intuition"],
                "earth": ["grounding", "stability", "material"],
                "air": ["intellect", "communication", "freedom"]
            },
            "archetypal_themes": {
                "hero": ["courage", "strength", "leadership"],
                "sage": ["wisdom", "knowledge", "guidance"],
                "lover": ["passion", "connection", "beauty"],
                "warrior": ["protection", "action", "discipline"],
                "mystic": ["intuition", "spirituality", "mystery"]
            }
        }
    
    def collapse_by_tags(self, tags: List[str]) -> CollapseResult:
        """
        Perform symbolic collapse based on tags.
        
        Args:
            tags: List of tags to collapse
            
        Returns:
            Collapse result with recommendations
        """
        if not tags:
            return CollapseResult(confidence=0.0, recommendations=[])
        
        # Analyze elemental balance
        elemental_scores = self._analyze_elemental_balance(tags)
        
        # Analyze archetypal themes
        archetypal_scores = self._analyze_archetypal_themes(tags)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(elemental_scores, archetypal_scores)
        
        # Calculate confidence
        confidence = self._calculate_confidence(elemental_scores, archetypal_scores)
        
        # Determine archetypal theme
        archetypal_theme = max(archetypal_scores.items(), key=lambda x: x[1])[0] if archetypal_scores else None
        
        return CollapseResult(
            confidence=confidence,
            recommendations=recommendations,
            archetypal_theme=archetypal_theme,
            energy_balance=self._determine_energy_balance(elemental_scores)
        )
    
    def _analyze_elemental_balance(self, tags: List[str]) -> Dict[str, float]:
        """Analyze elemental balance from tags."""
        scores = {"fire": 0.0, "water": 0.0, "earth": 0.0, "air": 0.0}
        
        for tag in tags:
            tag_lower = tag.lower()
            for element, keywords in self.collapse_patterns["elemental_balance"].items():
                if any(keyword in tag_lower for keyword in keywords):
                    scores[element] += 1.0
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def _analyze_archetypal_themes(self, tags: List[str]) -> Dict[str, float]:
        """Analyze archetypal themes from tags."""
        scores = {}
        
        for tag in tags:
            tag_lower = tag.lower()
            for archetype, keywords in self.collapse_patterns["archetypal_themes"].items():
                if any(keyword in tag_lower for keyword in keywords):
                    scores[archetype] = scores.get(archetype, 0.0) + 1.0
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def _generate_recommendations(self, elemental_scores: Dict[str, float], 
                                archetypal_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Elemental recommendations
        dominant_element = max(elemental_scores.items(), key=lambda x: x[1])
        if dominant_element[1] > 0.3:
            recommendations.append(f"Strong {dominant_element[0]} energy detected")
        
        # Archetypal recommendations
        if archetypal_scores:
            dominant_archetype = max(archetypal_scores.items(), key=lambda x: x[1])
            if dominant_archetype[1] > 0.3:
                recommendations.append(f"Archetypal theme: {dominant_archetype[0]}")
        
        # Balance recommendations
        if len(recommendations) == 0:
            recommendations.append("Neutral symbolic energy detected")
        
        return recommendations
    
    def _calculate_confidence(self, elemental_scores: Dict[str, float], 
                            archetypal_scores: Dict[str, float]) -> float:
        """Calculate confidence in the collapse result."""
        # Base confidence on how strongly patterns are detected
        elemental_confidence = max(elemental_scores.values()) if elemental_scores else 0.0
        archetypal_confidence = max(archetypal_scores.values()) if archetypal_scores else 0.0
        
        return (elemental_confidence + archetypal_confidence) / 2.0
    
    def _determine_energy_balance(self, elemental_scores: Dict[str, float]) -> str:
        """Determine overall energy balance."""
        yang_elements = elemental_scores.get("fire", 0.0) + elemental_scores.get("air", 0.0)
        yin_elements = elemental_scores.get("water", 0.0) + elemental_scores.get("earth", 0.0)
        
        if yang_elements > yin_elements + 0.2:
            return "yang"
        elif yin_elements > yang_elements + 0.2:
            return "yin"
        else:
            return "balanced"

class TarotMapping:
    """Manages tarot card mapping and correspondences."""
    
    def __init__(self, symbolic_system_path: Optional[str] = None, 
                 glyphs_path: Optional[str] = None):
        """
        Initialize tarot mapping system.
        
        Args:
            symbolic_system_path: Path to symbolic system data
            glyphs_path: Path to glyphs data
        """
        self.symbolic_system_path = symbolic_system_path
        self.glyphs_path = glyphs_path
        self.tarot_cards = self._load_tarot_cards()
        self.correspondences = self._load_correspondences()
    
    def _load_tarot_cards(self) -> Dict[str, Dict[str, Any]]:
        """Load tarot card definitions."""
        # Default tarot cards
        return {
            "The Fool": {
                "number": 0,
                "element": "air",
                "planet": "uranus",
                "keywords": ["innocence", "new beginnings", "spontaneity"],
                "description": "New beginnings, innocence, spontaneity"
            },
            "The Magician": {
                "number": 1,
                "element": "air",
                "planet": "mercury",
                "keywords": ["manifestation", "skill", "power"],
                "description": "Manifestation, skill, power"
            },
            "The High Priestess": {
                "number": 2,
                "element": "water",
                "planet": "moon",
                "keywords": ["intuition", "mystery", "subconscious"],
                "description": "Intuition, mystery, subconscious"
            },
            "The Sun": {
                "number": 19,
                "element": "fire",
                "planet": "sun",
                "keywords": ["vitality", "joy", "consciousness"],
                "description": "Vitality, joy, consciousness"
            },
            "The Moon": {
                "number": 18,
                "element": "water",
                "planet": "moon",
                "keywords": ["intuition", "illusion", "subconscious"],
                "description": "Intuition, illusion, subconscious"
            }
        }
    
    def _load_correspondences(self) -> Dict[str, List[str]]:
        """Load glyph-to-tarot correspondences."""
        return {
            "sun": ["The Sun"],
            "moon": ["The Moon"],
            "earth": ["The World"],
            "water": ["The Star"],
            "fire": ["The Tower"]
        }
    
    def get_tarot_card(self, card_name: str) -> Optional[Dict[str, Any]]:
        """Get tarot card information."""
        return self.tarot_cards.get(card_name)
    
    def map_glyph_to_tarot(self, glyph_name: str) -> List[str]:
        """Map a glyph to corresponding tarot cards."""
        return self.correspondences.get(glyph_name.lower(), [])
    
    def get_all_cards(self) -> List[str]:
        """Get all available tarot cards."""
        return list(self.tarot_cards.keys())
    
    def search_cards_by_keyword(self, keyword: str) -> List[str]:
        """Search tarot cards by keyword."""
        matching_cards = []
        keyword_lower = keyword.lower()
        
        for card_name, card_data in self.tarot_cards.items():
            if (keyword_lower in card_name.lower() or 
                any(keyword_lower in kw.lower() for kw in card_data.get("keywords", []))):
                matching_cards.append(card_name)
        
        return matching_cards 
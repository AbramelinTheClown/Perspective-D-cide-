"""
Map glyphs → cards → archetypes → actions.

This module provides tarot-based symbolic reasoning and archetypal mapping
for glyph interpretation and decision-making.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from ..glyphs.lookup import GlyphLookup

class TarotMapping:
    """
    Maps glyphs to tarot cards and archetypal meanings.
    
    Provides methods for tarot-based symbolic reasoning and interpretation.
    """
    
    def __init__(self, symbolic_system_path: Union[str, Path] = None, glyphs_path: Union[str, Path] = None):
        """
        Initialize the tarot mapping system.
        
        Args:
            symbolic_system_path: Path to symbolic_sytem.json
            glyphs_path: Path to glyphs.jsonl file
        """
        if symbolic_system_path is None:
            self.symbolic_system_path = Path(__file__).parent.parent / "symbolic_sytem.json"
        else:
            self.symbolic_system_path = Path(symbolic_system_path)
            
        self.glyph_lookup = GlyphLookup(glyphs_path)
        self.tarot_data = self._load_tarot_data()
    
    def _load_tarot_data(self) -> Dict[str, Any]:
        """Load tarot data from symbolic system file."""
        if not self.symbolic_system_path.exists():
            return {}
            
        with open(self.symbolic_system_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get('structure', {}).get('Major Arcana', {})
    
    def get_card_by_number(self, number: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Get tarot card data by number.
        
        Args:
            number: Card number (0-21)
            
        Returns:
            Card data or None
        """
        return self.tarot_data.get(str(number))
    
    def get_card_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Get tarot card data by title.
        
        Args:
            title: Card title (e.g., "The Fool", "The Magician")
            
        Returns:
            Card data or None
        """
        for card_data in self.tarot_data.values():
            if card_data.get('title') == title:
                return card_data
        return None
    
    def find_glyphs_for_card(self, card_number: Union[int, str]) -> List[Dict[str, Any]]:
        """
        Find glyphs associated with a tarot card.
        
        Args:
            card_number: Tarot card number
            
        Returns:
            List of associated glyphs
        """
        card = self.get_card_by_number(card_number)
        if not card:
            return []
        
        # Find glyphs by tarot path
        path = card.get('path', '')
        if path:
            return self.glyph_lookup.find_by_tarot(path)
        
        return []
    
    def get_archetypal_meaning(self, card_number: Union[int, str]) -> Dict[str, Any]:
        """
        Get archetypal meaning for a tarot card.
        
        Args:
            card_number: Tarot card number
            
        Returns:
            Dictionary with archetypal meaning data
        """
        card = self.get_card_by_number(card_number)
        if not card:
            return {}
        
        return {
            'title': card.get('title', ''),
            'planet': card.get('planet', ''),
            'zodiac': card.get('zodiac', ''),
            'modality': card.get('modality', ''),
            'path': card.get('path', ''),
            'keywords': card.get('keywords', []),
            'hebrew': card.get('hebrew', []),
            'angel_numbers': card.get('angel_numbers', []),
            'archetypal_theme': self._extract_archetypal_theme(card)
        }
    
    def _extract_archetypal_theme(self, card: Dict[str, Any]) -> str:
        """Extract archetypal theme from card data."""
        title = card.get('title', '').lower()
        
        # Map titles to archetypal themes
        theme_mapping = {
            'the fool': 'innocence and new beginnings',
            'the magician': 'manifestation and willpower',
            'the high priestess': 'intuition and mystery',
            'the empress': 'fertility and abundance',
            'the emperor': 'authority and structure',
            'the hierophant': 'tradition and conformity',
            'the lovers': 'partnership and choice',
            'the chariot': 'control and determination',
            'strength': 'courage and inner strength',
            'the hermit': 'introspection and solitude',
            'wheel of fortune': 'change and cycles',
            'justice': 'balance and fairness',
            'the hanged man': 'surrender and perspective',
            'death': 'transformation and endings',
            'temperance': 'moderation and patience',
            'the devil': 'shadow and temptation',
            'the tower': 'sudden change and revelation',
            'the star': 'hope and inspiration',
            'the moon': 'illusion and intuition',
            'the sun': 'joy and vitality',
            'judgement': 'rebirth and inner calling',
            'the world': 'completion and integration'
        }
        
        return theme_mapping.get(title, 'unknown archetype')
    
    def suggest_action_for_card(self, card_number: Union[int, str], context: str = '') -> Dict[str, Any]:
        """
        Suggest actions based on tarot card meaning.
        
        Args:
            card_number: Tarot card number
            context: Additional context for action suggestion
            
        Returns:
            Dictionary with suggested actions
        """
        card = self.get_card_by_number(card_number)
        if not card:
            return {}
        
        archetypal_meaning = self.get_archetypal_meaning(card_number)
        keywords = archetypal_meaning.get('keywords', [])
        
        # Generate action suggestions based on keywords
        actions = []
        for keyword in keywords:
            action = self._keyword_to_action(keyword, context)
            if action:
                actions.append(action)
        
        return {
            'card': card.get('title', ''),
            'archetypal_theme': archetypal_meaning.get('archetypal_theme', ''),
            'suggested_actions': actions,
            'confidence': min(1.0, len(actions) / 3.0)
        }
    
    def _keyword_to_action(self, keyword: str, context: str) -> Optional[str]:
        """Convert keyword to actionable suggestion."""
        keyword_lower = keyword.lower()
        
        action_mapping = {
            'divine vision': 'meditate on your higher purpose',
            'intuition': 'trust your gut feeling',
            'stability': 'ground yourself in the present moment',
            'clarity': 'seek clear communication',
            'strength': 'draw on your inner resources',
            'leadership': 'take charge of the situation',
            'respect': 'show respect to others and yourself',
            'wisdom': 'make thoughtful decisions',
            'truth': 'seek honest answers',
            'trust': 'have faith in the process',
            'timing': 'be patient with divine timing',
            'control': 'maintain focus and determination',
            'chaos': 'embrace the unknown',
            'intention': 'set clear intentions',
            'balance': 'find equilibrium in your life',
            'transformation': 'embrace change and growth',
            'hope': 'maintain optimism',
            'inspiration': 'follow your creative impulses',
            'joy': 'celebrate life\'s pleasures',
            'completion': 'finish what you started'
        }
        
        for key, action in action_mapping.items():
            if key in keyword_lower:
                return action
        
        return None
    
    def get_card_combination_meaning(self, card_numbers: List[Union[int, str]]) -> Dict[str, Any]:
        """
        Get meaning for a combination of tarot cards.
        
        Args:
            card_numbers: List of card numbers
            
        Returns:
            Combined meaning and interpretation
        """
        if len(card_numbers) < 2:
            return {}
        
        cards = []
        themes = []
        keywords = []
        
        for number in card_numbers:
            card = self.get_card_by_number(number)
            if card:
                cards.append(card.get('title', ''))
                archetypal = self.get_archetypal_meaning(number)
                themes.append(archetypal.get('archetypal_theme', ''))
                keywords.extend(archetypal.get('keywords', []))
        
        # Analyze combination
        combination_theme = self._analyze_combination_themes(themes)
        combination_keywords = list(set(keywords))  # Remove duplicates
        
        return {
            'cards': cards,
            'combination_theme': combination_theme,
            'shared_keywords': combination_keywords,
            'interpretation': self._generate_combination_interpretation(cards, combination_theme)
        }
    
    def _analyze_combination_themes(self, themes: List[str]) -> str:
        """Analyze themes to find common patterns."""
        if not themes:
            return "unknown combination"
        
        # Simple theme analysis - could be enhanced with more sophisticated NLP
        theme_text = ' '.join(themes).lower()
        
        if 'transformation' in theme_text and 'change' in theme_text:
            return "major transformation and renewal"
        elif 'intuition' in theme_text and 'wisdom' in theme_text:
            return "deep intuitive wisdom"
        elif 'strength' in theme_text and 'courage' in theme_text:
            return "inner strength and courage"
        elif 'balance' in theme_text and 'harmony' in theme_text:
            return "seeking balance and harmony"
        else:
            return f"combination of {len(themes)} archetypal energies"
    
    def _generate_combination_interpretation(self, cards: List[str], theme: str) -> str:
        """Generate interpretation for card combination."""
        if len(cards) == 2:
            return f"The combination of {cards[0]} and {cards[1]} suggests {theme}."
        elif len(cards) == 3:
            return f"The triad of {', '.join(cards[:-1])}, and {cards[-1]} indicates {theme}."
        else:
            return f"This complex combination of {len(cards)} cards represents {theme}." 
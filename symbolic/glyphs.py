"""
Glyph lookup and management system for Perspective D<cide>.

Provides glyph search, categorization, and symbolic mapping functionality.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Glyph:
    """Represents a symbolic glyph."""
    
    id: str
    name: str
    symbol: str
    category: str
    description: str
    tags: List[str]
    tarot: Optional[str] = None
    archetype: Optional[str] = None
    energy: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class GlyphLookup:
    """Manages glyph lookup and search functionality."""
    
    def __init__(self, glyphs_path: Optional[str] = None):
        """
        Initialize glyph lookup system.
        
        Args:
            glyphs_path: Path to glyphs JSONL file
        """
        self.glyphs_path = glyphs_path
        self.glyphs: Dict[str, Glyph] = {}
        self.name_index: Dict[str, List[str]] = {}
        self.category_index: Dict[str, List[str]] = {}
        self.tag_index: Dict[str, List[str]] = {}
        
        if glyphs_path:
            self.load_glyphs()
    
    def load_glyphs(self) -> None:
        """Load glyphs from JSONL file."""
        if not self.glyphs_path:
            logger.warning("No glyphs path specified, using default glyphs")
            self._load_default_glyphs()
            return
        
        try:
            glyphs_file = Path(self.glyphs_path)
            if not glyphs_file.exists():
                logger.warning(f"Glyphs file not found: {self.glyphs_path}")
                self._load_default_glyphs()
                return
            
            with open(glyphs_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        glyph = Glyph(
                            id=data.get('id', f"glyph_{line_num}"),
                            name=data.get('name', ''),
                            symbol=data.get('symbol', ''),
                            category=data.get('category', ''),
                            description=data.get('description', ''),
                            tags=data.get('tags', []),
                            tarot=data.get('tarot'),
                            archetype=data.get('archetype'),
                            energy=data.get('energy'),
                            metadata=data.get('metadata', {})
                        )
                        self.add_glyph(glyph)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON at line {line_num}: {e}")
                    except Exception as e:
                        logger.error(f"Error loading glyph at line {line_num}: {e}")
            
            logger.info(f"Loaded {len(self.glyphs)} glyphs from {self.glyphs_path}")
            
        except Exception as e:
            logger.error(f"Error loading glyphs: {e}")
            self._load_default_glyphs()
    
    def _load_default_glyphs(self) -> None:
        """Load a minimal set of default glyphs."""
        default_glyphs = [
            {
                'id': 'sun',
                'name': 'Sun',
                'symbol': '☀️',
                'category': 'celestial',
                'description': 'Solar energy, vitality, consciousness',
                'tags': ['light', 'energy', 'consciousness', 'vitality'],
                'tarot': 'The Sun',
                'archetype': 'Hero',
                'energy': 'yang'
            },
            {
                'id': 'moon',
                'name': 'Moon',
                'symbol': '🌙',
                'category': 'celestial',
                'description': 'Lunar energy, intuition, subconscious',
                'tags': ['intuition', 'subconscious', 'mystery', 'emotion'],
                'tarot': 'The Moon',
                'archetype': 'Mystic',
                'energy': 'yin'
            },
            {
                'id': 'earth',
                'name': 'Earth',
                'symbol': '🌍',
                'category': 'elemental',
                'description': 'Grounding, stability, material world',
                'tags': ['grounding', 'stability', 'material', 'practical'],
                'tarot': 'The World',
                'archetype': 'Sage',
                'energy': 'yin'
            },
            {
                'id': 'water',
                'name': 'Water',
                'symbol': '💧',
                'category': 'elemental',
                'description': 'Emotion, flow, adaptability',
                'tags': ['emotion', 'flow', 'adaptability', 'intuition'],
                'tarot': 'The Star',
                'archetype': 'Lover',
                'energy': 'yin'
            },
            {
                'id': 'fire',
                'name': 'Fire',
                'symbol': '🔥',
                'category': 'elemental',
                'description': 'Transformation, passion, energy',
                'tags': ['transformation', 'passion', 'energy', 'creativity'],
                'tarot': 'The Tower',
                'archetype': 'Warrior',
                'energy': 'yang'
            }
        ]
        
        for data in default_glyphs:
            glyph = Glyph(**data)
            self.add_glyph(glyph)
        
        logger.info(f"Loaded {len(self.glyphs)} default glyphs")
    
    def add_glyph(self, glyph: Glyph) -> None:
        """Add a glyph to the lookup system."""
        self.glyphs[glyph.id] = glyph
        
        # Index by name
        name_lower = glyph.name.lower()
        if name_lower not in self.name_index:
            self.name_index[name_lower] = []
        self.name_index[name_lower].append(glyph.id)
        
        # Index by category
        if glyph.category not in self.category_index:
            self.category_index[glyph.category] = []
        self.category_index[glyph.category].append(glyph.id)
        
        # Index by tags
        for tag in glyph.tags:
            tag_lower = tag.lower()
            if tag_lower not in self.tag_index:
                self.tag_index[tag_lower] = []
            self.tag_index[tag_lower].append(glyph.id)
    
    def get_glyph(self, glyph_id: str) -> Optional[Glyph]:
        """Get a glyph by ID."""
        return self.glyphs.get(glyph_id)
    
    def search_by_name(self, name: str) -> List[Glyph]:
        """Search glyphs by name (partial match)."""
        name_lower = name.lower()
        matching_ids = []
        
        for indexed_name, glyph_ids in self.name_index.items():
            if name_lower in indexed_name or indexed_name in name_lower:
                matching_ids.extend(glyph_ids)
        
        return [self.glyphs[glyph_id] for glyph_id in set(matching_ids)]
    
    def search_by_category(self, category: str) -> List[Glyph]:
        """Search glyphs by category."""
        glyph_ids = self.category_index.get(category, [])
        return [self.glyphs[glyph_id] for glyph_id in glyph_ids]
    
    def search_by_tags(self, tags: List[str]) -> List[Glyph]:
        """Search glyphs by tags."""
        matching_ids = set()
        
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in self.tag_index:
                matching_ids.update(self.tag_index[tag_lower])
        
        return [self.glyphs[glyph_id] for glyph_id in matching_ids]
    
    def get_all_glyphs(self) -> List[Glyph]:
        """Get all glyphs."""
        return list(self.glyphs.values())
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(self.category_index.keys())
    
    def get_tags(self) -> List[str]:
        """Get all available tags."""
        return list(self.tag_index.keys()) 
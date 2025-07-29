"""
Unified interface to search, fetch, and describe glyphs using tag filters, regex, or tarot mapping.

This module provides a high-level API for glyph discovery and symbolic reasoning.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
from ..utils.lazy_loader import stream_jsonl, find_by_tag, find_by_id, search_by_regex, find_by_multiple_tags

class GlyphLookup:
    """
    Unified glyph lookup and search interface.
    
    Provides methods for finding glyphs by various criteria including tags,
    symbolic intent, tarot correspondences, and accessibility labels.
    """
    
    def __init__(self, glyphs_path: Union[str, Path] = None):
        """
        Initialize the glyph lookup system.
        
        Args:
            glyphs_path: Path to glyphs.jsonl file. If None, uses default location.
        """
        if glyphs_path is None:
            self.glyphs_path = Path(__file__).parent.parent / "assets" / "glyphs.jsonl"
        else:
            self.glyphs_path = Path(glyphs_path)
            
        if not self.glyphs_path.exists():
            raise FileNotFoundError(f"Glyphs file not found: {self.glyphs_path}")
    
    def find_by_tags(self, tags: List[str], match_all: bool = True) -> List[Dict[str, Any]]:
        """
        Find glyphs by tag criteria.
        
        Args:
            tags: List of tags to search for
            match_all: If True, glyph must have all tags. If False, any tag.
            
        Returns:
            List of matching glyph records
        """
        return find_by_multiple_tags(self.glyphs_path, tags, match_all)
    
    def find_by_intent(self, intent: str) -> List[Dict[str, Any]]:
        """
        Find glyphs by symbolic intent.
        
        Args:
            intent: Intent to search for (e.g., 'shape', 'process', 'decision')
            
        Returns:
            List of matching glyph records
        """
        return [glyph for glyph in stream_jsonl(self.glyphs_path) 
                if glyph.get('intent', '').lower() == intent.lower()]
    
    def find_by_tarot(self, tarot_path: str) -> List[Dict[str, Any]]:
        """
        Find glyphs by tarot path correspondence.
        
        Args:
            tarot_path: Tarot path to search for
            
        Returns:
            List of matching glyph records
        """
        return [glyph for glyph in stream_jsonl(self.glyphs_path) 
                if glyph.get('tarot') == tarot_path]
    
    def find_by_symbolic_id(self, symbolic_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a glyph by its symbolic ID.
        
        Args:
            symbolic_id: Symbolic identifier
            
        Returns:
            Matching glyph record or None
        """
        return find_by_id(self.glyphs_path, 'symbolic_id', symbolic_id)
    
    def find_by_glyph_id(self, glyph_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a glyph by its glyph ID (codepoint).
        
        Args:
            glyph_id: Glyph ID (e.g., 'U+E200')
            
        Returns:
            Matching glyph record or None
        """
        return find_by_id(self.glyphs_path, 'glyph_id', glyph_id)
    
    def search_by_name(self, name_pattern: str) -> List[Dict[str, Any]]:
        """
        Search glyphs by name using regex pattern.
        
        Args:
            name_pattern: Regex pattern for name search
            
        Returns:
            List of matching glyph records
        """
        return search_by_regex(self.glyphs_path, 'name', name_pattern)
    
    def get_all_intents(self) -> Set[str]:
        """
        Get all unique intents in the glyph system.
        
        Returns:
            Set of unique intents
        """
        intents = set()
        for glyph in stream_jsonl(self.glyphs_path):
            intent = glyph.get('intent')
            if intent:
                intents.add(intent)
        return intents
    
    def get_all_symbolic_ids(self) -> Set[str]:
        """
        Get all unique symbolic IDs in the glyph system.
        
        Returns:
            Set of unique symbolic IDs
        """
        symbolic_ids = set()
        for glyph in stream_jsonl(self.glyphs_path):
            symbolic_id = glyph.get('symbolic_id')
            if symbolic_id:
                symbolic_ids.add(symbolic_id)
        return symbolic_ids
    
    def describe_glyph(self, glyph_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a comprehensive description of a glyph.
        
        Args:
            glyph_id: Glyph ID to describe
            
        Returns:
            Dictionary with glyph description and metadata
        """
        glyph = self.find_by_glyph_id(glyph_id)
        if not glyph:
            return None
            
        description = {
            'glyph_id': glyph['glyph_id'],
            'name': glyph.get('name', ''),
            'description': glyph.get('description', ''),
            'intent': glyph.get('intent', ''),
            'symbolic_id': glyph.get('symbolic_id', ''),
            'tags': glyph.get('tags', []),
            'accessibility_label': glyph.get('accessibility_label', ''),
            'font': glyph.get('font', ''),
            'tarot_path': glyph.get('tarot', ''),
            'unicode_fallback': glyph.get('unicode', ''),
            'svg_available': bool(glyph.get('svg_path')),
        }
        
        return description
    
    def get_glyph_hierarchy(self) -> Dict[str, Any]:
        """
        Get the hierarchical structure of glyphs by intent and tags.
        
        Returns:
            Dictionary representing the glyph hierarchy
        """
        hierarchy = {}
        
        for glyph in stream_jsonl(self.glyphs_path):
            intent = glyph.get('intent', 'unknown')
            if intent not in hierarchy:
                hierarchy[intent] = {'glyphs': [], 'tags': set()}
            
            hierarchy[intent]['glyphs'].append({
                'glyph_id': glyph['glyph_id'],
                'name': glyph.get('name', ''),
                'symbolic_id': glyph.get('symbolic_id', ''),
                'tags': glyph.get('tags', [])
            })
            
            hierarchy[intent]['tags'].update(glyph.get('tags', []))
        
        # Convert sets to lists for JSON serialization
        for intent_data in hierarchy.values():
            intent_data['tags'] = list(intent_data['tags'])
        
        return hierarchy

# Convenience functions
def find_glyphs_by_intent(intent: str, glyphs_path: Union[str, Path] = None) -> List[Dict[str, Any]]:
    """Convenience function to find glyphs by intent."""
    lookup = GlyphLookup(glyphs_path)
    return lookup.find_by_intent(intent)

def get_glyph_hierarchy(glyphs_path: Union[str, Path] = None) -> Dict[str, Any]:
    """Convenience function to get glyph hierarchy."""
    lookup = GlyphLookup(glyphs_path)
    return lookup.get_glyph_hierarchy() 
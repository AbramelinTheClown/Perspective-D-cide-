"""
Implements symbolic decision/collapse engine using glyph + tag state.

This module provides the core logic for symbolic reasoning and decision-making
based on glyph combinations and their semantic relationships.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union
from pathlib import Path
from ..glyphs.lookup import GlyphLookup

@dataclass
class CollapseState:
    """
    Represents the current state of a symbolic collapse operation.
    
    Tracks which glyphs are active, their relationships, and the decision context.
    """
    
    active_glyphs: List[str] = field(default_factory=list)
    active_tags: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    
    def add_glyph(self, glyph_id: str, metadata: Dict[str, Any] = None) -> None:
        """Add a glyph to the active state."""
        if glyph_id not in self.active_glyphs:
            self.active_glyphs.append(glyph_id)
            if metadata:
                self.context[f"glyph_{glyph_id}"] = metadata
    
    def remove_glyph(self, glyph_id: str) -> None:
        """Remove a glyph from the active state."""
        if glyph_id in self.active_glyphs:
            self.active_glyphs.remove(glyph_id)
            self.context.pop(f"glyph_{glyph_id}", None)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the active state."""
        self.active_tags.add(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the active state."""
        self.active_tags.discard(tag)
    
    def record_decision(self, decision: Dict[str, Any]) -> None:
        """Record a decision in the history."""
        decision['timestamp'] = json.dumps({'active_glyphs': self.active_glyphs.copy(),
                                          'active_tags': list(self.active_tags.copy())})
        self.history.append(decision)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'active_glyphs': self.active_glyphs,
            'active_tags': list(self.active_tags),
            'context': self.context,
            'history': self.history,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollapseState':
        """Create state from dictionary."""
        state = cls()
        state.active_glyphs = data.get('active_glyphs', [])
        state.active_tags = set(data.get('active_tags', []))
        state.context = data.get('context', {})
        state.history = data.get('history', [])
        state.confidence = data.get('confidence', 0.0)
        return state

class SymbolicCollapse:
    """
    Symbolic decision and collapse engine.
    
    Processes glyph combinations and their semantic relationships to make
    symbolic decisions and collapse complex states into simpler representations.
    """
    
    def __init__(self, glyphs_path: Union[str, Path] = None):
        """
        Initialize the symbolic collapse engine.
        
        Args:
            glyphs_path: Path to glyphs.jsonl file
        """
        self.glyph_lookup = GlyphLookup(glyphs_path)
        self.state = CollapseState()
    
    def load_state(self, state_data: Dict[str, Any]) -> None:
        """Load a collapse state from dictionary."""
        self.state = CollapseState.from_dict(state_data)
    
    def save_state(self) -> Dict[str, Any]:
        """Save current state to dictionary."""
        return self.state.to_dict()
    
    def collapse_by_tags(self, tags: List[str], strategy: str = 'intersection') -> Dict[str, Any]:
        """
        Collapse glyphs based on tag criteria.
        
        Args:
            tags: Tags to use for collapse
            strategy: 'intersection' (all tags) or 'union' (any tag)
            
        Returns:
            Collapse result with recommendations
        """
        match_all = strategy == 'intersection'
        matching_glyphs = self.glyph_lookup.find_by_tags(tags, match_all)
        
        # Update state
        for glyph in matching_glyphs:
            self.state.add_glyph(glyph['glyph_id'], glyph)
            self.state.active_tags.update(glyph.get('tags', []))
        
        # Calculate confidence based on match quality
        confidence = min(1.0, len(matching_glyphs) / max(len(tags), 1))
        self.state.confidence = confidence
        
        # Generate collapse result
        result = {
            'strategy': strategy,
            'input_tags': tags,
            'matching_glyphs': len(matching_glyphs),
            'confidence': confidence,
            'recommendations': self._generate_recommendations(matching_glyphs),
            'next_steps': self._suggest_next_steps(matching_glyphs)
        }
        
        self.state.record_decision(result)
        return result
    
    def collapse_by_intent(self, intent: str) -> Dict[str, Any]:
        """
        Collapse glyphs based on symbolic intent.
        
        Args:
            intent: Intent to collapse by
            
        Returns:
            Collapse result with recommendations
        """
        matching_glyphs = self.glyph_lookup.find_by_intent(intent)
        
        # Update state
        for glyph in matching_glyphs:
            self.state.add_glyph(glyph['glyph_id'], glyph)
            self.state.active_tags.update(glyph.get('tags', []))
        
        # Calculate confidence
        confidence = min(1.0, len(matching_glyphs) / 10.0)  # Normalize to 0-1
        self.state.confidence = confidence
        
        result = {
            'strategy': 'intent',
            'input_intent': intent,
            'matching_glyphs': len(matching_glyphs),
            'confidence': confidence,
            'recommendations': self._generate_recommendations(matching_glyphs),
            'next_steps': self._suggest_next_steps(matching_glyphs)
        }
        
        self.state.record_decision(result)
        return result
    
    def collapse_by_tarot(self, tarot_path: str) -> Dict[str, Any]:
        """
        Collapse glyphs based on tarot path correspondence.
        
        Args:
            tarot_path: Tarot path to collapse by
            
        Returns:
            Collapse result with recommendations
        """
        matching_glyphs = self.glyph_lookup.find_by_tarot(tarot_path)
        
        # Update state
        for glyph in matching_glyphs:
            self.state.add_glyph(glyph['glyph_id'], glyph)
            self.state.active_tags.update(glyph.get('tags', []))
        
        # Calculate confidence
        confidence = min(1.0, len(matching_glyphs) / 5.0)  # Tarot paths are more specific
        self.state.confidence = confidence
        
        result = {
            'strategy': 'tarot',
            'input_tarot_path': tarot_path,
            'matching_glyphs': len(matching_glyphs),
            'confidence': confidence,
            'recommendations': self._generate_recommendations(matching_glyphs),
            'next_steps': self._suggest_next_steps(matching_glyphs)
        }
        
        self.state.record_decision(result)
        return result
    
    def _generate_recommendations(self, glyphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on matching glyphs."""
        recommendations = []
        
        for glyph in glyphs[:5]:  # Limit to top 5
            rec = {
                'glyph_id': glyph['glyph_id'],
                'name': glyph.get('name', ''),
                'intent': glyph.get('intent', ''),
                'description': glyph.get('description', ''),
                'confidence_boost': self._calculate_confidence_boost(glyph)
            }
            recommendations.append(rec)
        
        return recommendations
    
    def _suggest_next_steps(self, glyphs: List[Dict[str, Any]]) -> List[str]:
        """Suggest next steps based on current glyphs."""
        steps = []
        
        if len(glyphs) == 0:
            steps.append("No matching glyphs found. Try broadening your search criteria.")
        elif len(glyphs) == 1:
            steps.append("Single glyph found. Consider adding complementary glyphs.")
        elif len(glyphs) < 5:
            steps.append("Few glyphs found. Consider refining your search.")
        else:
            steps.append("Multiple glyphs found. Consider filtering by additional criteria.")
        
        # Add intent-specific suggestions
        intents = set(glyph.get('intent') for glyph in glyphs if glyph.get('intent'))
        if 'shape' in intents:
            steps.append("Shape glyphs detected. Consider adding process or decision glyphs.")
        if 'process' in intents:
            steps.append("Process glyphs detected. Consider adding outcome or result glyphs.")
        
        return steps
    
    def _calculate_confidence_boost(self, glyph: Dict[str, Any]) -> float:
        """Calculate confidence boost for a glyph based on state context."""
        boost = 0.0
        
        # Boost if glyph tags match active tags
        glyph_tags = set(glyph.get('tags', []))
        overlap = len(glyph_tags & self.state.active_tags)
        boost += overlap * 0.1
        
        # Boost if glyph intent matches context
        if glyph.get('intent') in self.state.context.get('preferred_intents', []):
            boost += 0.2
        
        return min(boost, 0.5)  # Cap at 0.5
    
    def get_collapse_summary(self) -> Dict[str, Any]:
        """Get a summary of the current collapse state."""
        return {
            'active_glyphs_count': len(self.state.active_glyphs),
            'active_tags_count': len(self.state.active_tags),
            'confidence': self.state.confidence,
            'history_length': len(self.state.history),
            'recent_decisions': self.state.history[-3:] if self.state.history else []
        }

# Convenience function
def collapse_symbols(criteria: Dict[str, Any], glyphs_path: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Convenience function to collapse symbols based on criteria.
    
    Args:
        criteria: Dictionary with 'type' and 'value' keys
        glyphs_path: Path to glyphs.jsonl file
        
    Returns:
        Collapse result
    """
    collapse = SymbolicCollapse(glyphs_path)
    
    collapse_type = criteria.get('type', 'tags')
    value = criteria.get('value')
    
    if collapse_type == 'tags':
        return collapse.collapse_by_tags(value)
    elif collapse_type == 'intent':
        return collapse.collapse_by_intent(value)
    elif collapse_type == 'tarot':
        return collapse.collapse_by_tarot(value)
    else:
        raise ValueError(f"Unknown collapse type: {collapse_type}") 
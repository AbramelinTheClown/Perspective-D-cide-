"""
Face parsing and mapping for UI scaffold generation.

Converts ASCII face glyphs into structured scaffold data using the
five-layer architecture (Bone, Blob, Biz, Leaf, Spirit).
"""

import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import the existing face parsing logic
from ..acsii_face import parse_glyph, ear_pairs, cheek_pairs, eyes, noses

@dataclass
class FaceStructure:
    """Structured representation of an ASCII face."""
    
    glyph: str
    header: str
    aside_left: str
    widget_left: str
    content_main: str
    widget_right: str
    aside_right: str
    footer: str
    
    @classmethod
    def from_parsed(cls, parsed: Dict[str, Any]) -> 'FaceStructure':
        """Create FaceStructure from parsed glyph data."""
        return cls(
            glyph=parsed.get('glyph', ''),
            header=parsed.get('header', ''),
            aside_left=parsed.get('aside_left', ''),
            widget_left=parsed.get('widget_left', ''),
            content_main=parsed.get('content_main', ''),
            widget_right=parsed.get('widget_right', ''),
            aside_right=parsed.get('aside_right', ''),
            footer=parsed.get('footer', '')
        )

class FaceParser:
    """
    Parser for ASCII faces that converts them into structured scaffold data.
    
    Uses the existing acsii_face.py parsing logic and extends it with
    scaffold-specific mapping and analysis.
    """
    
    def __init__(self):
        """Initialize the face parser."""
        self.ear_pairs = ear_pairs
        self.cheek_pairs = cheek_pairs
        self.eyes = eyes
        self.noses = noses
    
    def parse_face(self, glyph: str) -> FaceStructure:
        """
        Parse an ASCII face glyph into structured data.
        
        Args:
            glyph: ASCII face glyph string
            
        Returns:
            FaceStructure containing parsed components
        """
        parsed = parse_glyph(glyph)
        return FaceStructure.from_parsed(parsed)
    
    def analyze_face_complexity(self, face: FaceStructure) -> Dict[str, Any]:
        """
        Analyze the complexity and characteristics of a face.
        
        Args:
            face: Parsed face structure
            
        Returns:
            Dictionary containing complexity analysis
        """
        # Count components
        eye_count = sum(1 for eye in self.eyes if eye in face.content_main)
        nose_count = sum(1 for nose in self.noses if nose in face.content_main)
        
        # Determine layout type
        layout_type = self._determine_layout_type(face)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(face)
        
        return {
            'eye_count': eye_count,
            'nose_count': nose_count,
            'layout_type': layout_type,
            'complexity_score': complexity_score,
            'has_widgets': bool(face.widget_left or face.widget_right),
            'has_header': bool(face.header),
            'has_footer': bool(face.footer)
        }
    
    def _determine_layout_type(self, face: FaceStructure) -> str:
        """Determine the layout type based on face structure."""
        
        if face.widget_left and face.widget_right:
            return "three_column"
        elif face.widget_left or face.widget_right:
            return "two_column"
        elif face.header and face.footer:
            return "header_footer"
        else:
            return "simple"
    
    def _calculate_complexity_score(self, face: FaceStructure) -> float:
        """Calculate a complexity score for the face (0.0 to 1.0)."""
        
        score = 0.0
        
        # Base score for having content
        if face.content_main:
            score += 0.2
        
        # Add score for each component
        if face.header:
            score += 0.1
        if face.footer:
            score += 0.1
        if face.widget_left:
            score += 0.15
        if face.widget_right:
            score += 0.15
        
        # Add score for content complexity
        content_length = len(face.content_main)
        if content_length > 5:
            score += 0.2
        elif content_length > 3:
            score += 0.1
        
        return min(score, 1.0)
    
    def get_face_templates(self) -> Dict[str, str]:
        """Get common face templates for different UI patterns."""
        
        return {
            "dashboard": "ʕᵒ☯ϖ☯ᵒʔ",
            "form": "乁﴾ovo乁﴿",
            "gallery": "୧ˇ☯-☯ˇ୨", 
            "chat": "ʢᵒᴗ.ᴗᵒʡ",
            "settings": "༼⌐■∇■¬༽",
            "simple": "◉V◉",
            "complex": "ᕳ>人<ᕲ",
            "minimal": "◕ᴥ◕"
        }
    
    def select_face_by_content(self, content_analysis: Dict[str, Any]) -> str:
        """
        Select appropriate face based on content analysis.
        
        Args:
            content_analysis: Analysis of content characteristics
            
        Returns:
            Appropriate face glyph
        """
        templates = self.get_face_templates()
        
        # Determine face based on content characteristics
        archetypal_theme = content_analysis.get('archetypal_theme', '')
        keywords = content_analysis.get('keywords', [])
        
        if archetypal_theme == "communication":
            return templates["chat"]
        elif archetypal_theme == "analysis":
            return templates["dashboard"]
        elif archetypal_theme == "input":
            return templates["form"]
        elif archetypal_theme == "display":
            return templates["gallery"]
        elif archetypal_theme == "configuration":
            return templates["settings"]
        elif len(keywords) > 10:
            return templates["complex"]
        elif len(keywords) < 3:
            return templates["minimal"]
        else:
            return templates["simple"]

def parse_face_to_scaffold(glyph: str) -> List[Dict[str, Any]]:
    """
    Parse ASCII face into UI scaffold JSONL.
    
    Args:
        glyph: ASCII face glyph string
        
    Returns:
        List of scaffold layer dictionaries
    """
    
    parser = FaceParser()
    face = parser.parse_face(glyph)
    
    scaffold = []
    
    # 1. Shell layer (Bone)
    shell_id = face.aside_left + face.aside_right
    scaffold.append({
        "type": "Shell",
        "id": shell_id,
        "props": {
            "viewport": "responsive",
            "theme": "auto",
            "layout": "flex"
        },
        "metadata": {
            "glyph_source": glyph,
            "layer": "bone",
            "description": "Global container shell"
        }
    })
    
    # 2. Container layers (Blob)
    if face.header:
        scaffold.append({
            "type": "Header",
            "id": face.aside_left,
            "props": {"position": "sticky"},
            "metadata": {"layer": "blob", "glyph_source": face.aside_left}
        })
    
    if face.widget_left:
        scaffold.append({
            "type": "Aside",
            "class": "left",
            "id": face.widget_left,
            "props": {"width": "250px"},
            "metadata": {"layer": "blob", "glyph_source": face.widget_left}
        })
    
    scaffold.append({
        "type": "Main",
        "id": face.content_main,
        "props": {"flex": "1"},
        "metadata": {"layer": "blob", "glyph_source": face.content_main}
    })
    
    if face.widget_right:
        scaffold.append({
            "type": "Aside",
            "class": "right",
            "id": face.widget_right,
            "props": {"width": "250px"},
            "metadata": {"layer": "blob", "glyph_source": face.widget_right}
        })
    
    if face.footer:
        scaffold.append({
            "type": "Footer",
            "id": face.aside_right,
            "props": {"position": "sticky"},
            "metadata": {"layer": "blob", "glyph_source": face.aside_right}
        })
    
    # 3. Component layers (Leaf) - from eyes and nose
    components = parse_components(face.content_main)
    for comp in components:
        scaffold.append({
            "type": "Card",
            "id": comp["id"],
            "props": {
                "variant": comp["variant"],
                "size": comp["size"]
            },
            "metadata": {"layer": "leaf", "glyph_source": comp["id"]}
        })
    
    # 4. Overlay layer (Spirit)
    scaffold.append({
        "type": "Overlay",
        "id": "overlay-root",
        "props": {"zIndex": 1000},
        "metadata": {"layer": "spirit", "glyph_source": "spirit"}
    })
    
    return scaffold

def parse_components(content_main: str) -> List[Dict[str, Any]]:
    """
    Parse the main content area into individual components.
    
    Args:
        content_main: Main content string (eyes + nose + eyes)
        
    Returns:
        List of component definitions
    """
    
    components = []
    
    # Split content into individual characters
    chars = list(content_main)
    
    for i, char in enumerate(chars):
        # Determine component type and variant
        if char in eyes:
            variant = "default"
            size = "large"
        elif char in noses:
            variant = "highlight"
            size = "medium"
        else:
            variant = "secondary"
            size = "small"
        
        components.append({
            "id": char,
            "variant": variant,
            "size": size,
            "position": i
        })
    
    return components 
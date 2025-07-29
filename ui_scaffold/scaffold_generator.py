"""
Scaffold generation and management for UI scaffolds.

Provides high-level functionality for generating, enhancing, and managing
UI scaffolds from ASCII faces and content analysis.
"""

import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .face_parser import FaceParser, parse_face_to_scaffold
from ..core.schemas import ContentItem
from ..symbolic import SymbolicAnalyzer

class ScaffoldGenerator:
    """
    High-level scaffold generator that combines face parsing with content analysis.
    
    Provides methods for generating scaffolds from faces, content, and templates,
    with support for enhancement and customization.
    """
    
    def __init__(self):
        """Initialize the scaffold generator."""
        self.face_parser = FaceParser()
        self.symbolic_analyzer = SymbolicAnalyzer()
    
    def generate_from_face(self, face_glyph: str, enhance: bool = False) -> List[Dict[str, Any]]:
        """
        Generate scaffold from a face glyph.
        
        Args:
            face_glyph: ASCII face glyph string
            enhance: Whether to enhance the scaffold with additional features
            
        Returns:
            List of scaffold layer dictionaries
        """
        scaffold = parse_face_to_scaffold(face_glyph)
        
        if enhance:
            scaffold = self.enhance_scaffold(scaffold, "standard")
        
        return scaffold
    
    def generate_from_content(self, content_item: ContentItem) -> List[Dict[str, Any]]:
        """
        Generate scaffold from content analysis.
        
        Args:
            content_item: Content item to analyze
            
        Returns:
            List of scaffold layer dictionaries
        """
        # Analyze content to determine appropriate face
        content_analysis = self.symbolic_analyzer.analyze_content(content_item)
        
        # Select appropriate face based on content characteristics
        appropriate_face = self.face_parser.select_face_by_content(
            content_analysis.results
        )
        
        # Generate scaffold from face
        scaffold = self.generate_from_face(appropriate_face, enhance=True)
        
        # Customize scaffold based on content analysis
        scaffold = self.customize_scaffold_for_content(scaffold, content_analysis)
        
        return scaffold
    
    def generate_from_template(self, template_name: str) -> List[Dict[str, Any]]:
        """
        Generate scaffold from a predefined template.
        
        Args:
            template_name: Name of the template to use
            
        Returns:
            List of scaffold layer dictionaries
        """
        templates = self.face_parser.get_face_templates()
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        face_glyph = templates[template_name]
        return self.generate_from_face(face_glyph, enhance=True)
    
    def enhance_scaffold(self, scaffold: List[Dict[str, Any]], enhancement_type: str) -> List[Dict[str, Any]]:
        """
        Enhance a scaffold with additional features.
        
        Args:
            scaffold: Base scaffold to enhance
            enhancement_type: Type of enhancement to apply
            
        Returns:
            Enhanced scaffold
        """
        enhanced = scaffold.copy()
        
        if enhancement_type == "complex":
            # Add more component layers
            enhanced.extend(self._generate_additional_components())
        elif enhancement_type == "responsive":
            # Add responsive breakpoints
            enhanced = self._add_responsive_features(enhanced)
        elif enhancement_type == "accessible":
            # Add accessibility features
            enhanced = self._add_accessibility_features(enhanced)
        elif enhancement_type == "interactive":
            # Add interactive features
            enhanced = self._add_interactive_features(enhanced)
        
        return enhanced
    
    def customize_scaffold_for_content(self, scaffold: List[Dict[str, Any]], content_analysis: Any) -> List[Dict[str, Any]]:
        """
        Customize scaffold based on content analysis.
        
        Args:
            scaffold: Base scaffold to customize
            content_analysis: Analysis result from symbolic analyzer
            
        Returns:
            Customized scaffold
        """
        customized = scaffold.copy()
        
        # Add content-specific metadata
        for layer in customized:
            if 'metadata' not in layer:
                layer['metadata'] = {}
            
            layer['metadata']['content_analysis'] = {
                'archetypal_theme': content_analysis.archetypal_theme,
                'confidence': content_analysis.confidence,
                'glyphs_used': content_analysis.glyphs_used
            }
        
        # Customize theme based on archetypal theme
        archetypal_theme = content_analysis.archetypal_theme
        if archetypal_theme:
            shell_layer = next((s for s in customized if s['type'] == 'Shell'), None)
            if shell_layer:
                shell_layer['props']['theme'] = self._map_theme_to_archetype(archetypal_theme)
        
        return customized
    
    def export_scaffold(self, scaffold: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
        """
        Export scaffold to JSONL file.
        
        Args:
            scaffold: Scaffold to export
            output_path: Path to output file
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for layer in scaffold:
                f.write(json.dumps(layer, ensure_ascii=False) + '\n')
    
    def import_scaffold(self, input_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Import scaffold from JSONL file.
        
        Args:
            input_path: Path to input file
            
        Returns:
            List of scaffold layer dictionaries
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Scaffold file not found: {input_path}")
        
        scaffold = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    layer = json.loads(line)
                    scaffold.append(layer)
        
        return scaffold
    
    def _generate_additional_components(self) -> List[Dict[str, Any]]:
        """Generate additional component layers for complex scaffolds."""
        
        return [
            {
                "type": "Section",
                "id": "additional-section",
                "props": {"padding": "2rem"},
                "metadata": {"layer": "biz", "description": "Additional content section"}
            },
            {
                "type": "Card",
                "id": "additional-card",
                "props": {"variant": "secondary", "size": "medium"},
                "metadata": {"layer": "leaf", "description": "Additional content card"}
            }
        ]
    
    def _add_responsive_features(self, scaffold: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add responsive design features to scaffold."""
        
        # Find shell layer and add responsive breakpoints
        for layer in scaffold:
            if layer['type'] == 'Shell':
                layer['props']['breakpoints'] = {
                    "mobile": "320px",
                    "tablet": "768px",
                    "desktop": "1024px"
                }
                break
        
        return scaffold
    
    def _add_accessibility_features(self, scaffold: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add accessibility features to scaffold."""
        
        for layer in scaffold:
            if 'props' not in layer:
                layer['props'] = {}
            
            # Add ARIA labels and roles
            if layer['type'] in ['Header', 'Main', 'Footer', 'Aside']:
                layer['props']['role'] = layer['type'].lower()
                layer['props']['aria-label'] = f"{layer['type']} section"
            
            elif layer['type'] == 'Card':
                layer['props']['role'] = 'article'
                layer['props']['aria-label'] = 'Content card'
        
        return scaffold
    
    def _add_interactive_features(self, scaffold: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add interactive features to scaffold."""
        
        # Add interactive overlay components
        interactive_overlays = [
            {
                "type": "Tooltip",
                "target": "◉",
                "props": {"text": "Information", "position": "top"},
                "metadata": {"layer": "spirit", "description": "Interactive tooltip"}
            },
            {
                "type": "Modal",
                "id": "interactive-modal",
                "props": {"trigger": "click", "backdrop": True},
                "metadata": {"layer": "spirit", "description": "Interactive modal"}
            }
        ]
        
        scaffold.extend(interactive_overlays)
        return scaffold
    
    def _map_theme_to_archetype(self, archetypal_theme: str) -> str:
        """Map archetypal theme to UI theme."""
        
        theme_mapping = {
            "communication": "blue",
            "analysis": "green", 
            "input": "orange",
            "display": "purple",
            "configuration": "gray",
            "creation": "yellow",
            "destruction": "red"
        }
        
        return theme_mapping.get(archetypal_theme, "auto")

class FaceToScaffold:
    """
    Convenience class for face-to-scaffold conversion.
    
    Provides a simple interface for converting ASCII faces to UI scaffolds
    with common enhancement patterns.
    """
    
    def __init__(self):
        """Initialize the face-to-scaffold converter."""
        self.generator = ScaffoldGenerator()
    
    def convert(self, face_glyph: str, enhancement: str = "standard") -> List[Dict[str, Any]]:
        """
        Convert face glyph to scaffold.
        
        Args:
            face_glyph: ASCII face glyph string
            enhancement: Enhancement type to apply
            
        Returns:
            List of scaffold layer dictionaries
        """
        return self.generator.generate_from_face(face_glyph, enhance=True)
    
    def template_to_scaffold(self, template_name: str) -> List[Dict[str, Any]]:
        """
        Convert template to scaffold.
        
        Args:
            template_name: Name of the template
            
        Returns:
            List of scaffold layer dictionaries
        """
        return self.generator.generate_from_template(template_name)
    
    def enhance_scaffold(self, face_glyph: str, enhancement_type: str) -> List[Dict[str, Any]]:
        """
        Generate enhanced scaffold from face.
        
        Args:
            face_glyph: ASCII face glyph string
            enhancement_type: Type of enhancement
            
        Returns:
            Enhanced scaffold
        """
        base_scaffold = self.generator.generate_from_face(face_glyph, enhance=False)
        return self.generator.enhance_scaffold(base_scaffold, enhancement_type) 
"""UI Scaffold System for Perspective D<cide>.

Transforms ASCII face glyphs into complete, responsive web interface scaffolds
using a five-layer architecture and JSONL streaming format.
"""

from .face_parser import FaceParser, parse_face_to_scaffold
from .scaffold_generator import ScaffoldGenerator, FaceToScaffold
from .renderers import (
    ReactScaffoldRenderer,
    VueScaffoldRenderer,
    ScaffoldRenderer
)
from .validators import ScaffoldValidator, ValidationResult
from .templates import ScaffoldTemplateManager
from .manager import UIScaffoldManager

__all__ = [
    # Core parsing and generation
    'FaceParser',
    'parse_face_to_scaffold',
    'ScaffoldGenerator',
    'FaceToScaffold',
    
    # Renderers
    'ReactScaffoldRenderer',
    'VueScaffoldRenderer', 
    'ScaffoldRenderer',
    
    # Validation and testing
    'ScaffoldValidator',
    'ValidationResult',
    
    # Templates and management
    'ScaffoldTemplateManager',
    'UIScaffoldManager'
] 
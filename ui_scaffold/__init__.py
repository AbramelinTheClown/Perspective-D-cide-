"""
UI Scaffold System for Perspective D<cide>.

Provides tools for converting ASCII faces and symbolic representations into
functional UI components and layouts.
"""

# Core scaffold components
from .face_parser import FaceParser, parse_face_to_scaffold
from .scaffold_generator import ScaffoldGenerator

# Renderers
try:
    from .renderers import ScaffoldRenderer, RendererConfig, ReactScaffoldRenderer, VueScaffoldRenderer
except ImportError as e:
    print(f"Warning: UI scaffold renderers not available: {e}")
    ScaffoldRenderer = None
    RendererConfig = None
    ReactScaffoldRenderer = None
    VueScaffoldRenderer = None

# Legacy exports for backward compatibility
try:
    from .face_parser import FaceToScaffold
except ImportError:
    FaceToScaffold = None

__all__ = [
    # Core components
    'FaceParser',
    'parse_face_to_scaffold',
    'ScaffoldGenerator',
    
    # Renderers
    'ScaffoldRenderer',
    'RendererConfig',
    'ReactScaffoldRenderer',
    'VueScaffoldRenderer',
    
    # Legacy exports
    'FaceToScaffold'
] 
"""Perspective D<cide> - Modular Symbolic Computing Library with Emergent Intelligence.

This package provides a comprehensive system for symbolic reasoning, content analysis,
and emergent taxonomy discovery through a modular, streaming-capable architecture.
"""

# Core framework
from .core import (
    Config, 
    initialize_framework,
    StorageBackend,
    ComponentRegistry,
    ContentItem,
    CategoryProposal,
    AnalysisResult
)

# Symbolic system (existing glyph/tarot functionality)
from .symbolic import (
    SymbolicAnalyzer,
    GlyphLookup,
    SymbolicCollapse,
    TarotMapping
)

# ETX system (new emergent taxonomy)
try:
    from .etx import (
        ETXFramework,
        CategorizationBuilder,
        FastEmbedEngine,
        MiniBatchKMeansEngine
    )
    ETX_AVAILABLE = True
except ImportError:
    ETX_AVAILABLE = False

# Legacy utilities (preserved for backward compatibility)
from .utils import (
    stream_jsonl,
    find_by_tag,
    find_by_accessibility_label,
    find_by_id
)

# UI Scaffold system (new feature)
try:
    from .ui_scaffold import (
        FaceToScaffold,
        ScaffoldGenerator,
        parse_face_to_scaffold,
        ReactScaffoldRenderer,
        VueScaffoldRenderer
    )
    UI_SCAFFOLD_AVAILABLE = True
except ImportError:
    UI_SCAFFOLD_AVAILABLE = False

__version__ = "0.2.0"

__all__ = [
    # Core framework
    'Config',
    'initialize_framework',
    'StorageBackend', 
    'ComponentRegistry',
    'ContentItem',
    'CategoryProposal',
    'AnalysisResult',
    
    # Symbolic system
    'SymbolicAnalyzer',
    'GlyphLookup',
    'SymbolicCollapse',
    'TarotMapping',
    
    # ETX system (if available)
    'ETX_AVAILABLE',
    
    # UI Scaffold system (if available)
    'UI_SCAFFOLD_AVAILABLE',
    
    # Legacy utilities
    'stream_jsonl',
    'find_by_tag',
    'find_by_accessibility_label', 
    'find_by_id'
]

# Add ETX components to __all__ if available
if ETX_AVAILABLE:
    __all__.extend([
        'ETXFramework',
        'CategorizationBuilder',
        'FastEmbedEngine', 
        'MiniBatchKMeansEngine'
    ])

# Add UI Scaffold components to __all__ if available
if UI_SCAFFOLD_AVAILABLE:
    __all__.extend([
        'FaceToScaffold',
        'ScaffoldGenerator',
        'parse_face_to_scaffold',
        'ReactScaffoldRenderer',
        'VueScaffoldRenderer'
    ]) 
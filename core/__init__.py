"""Core framework components for Perspective D<cide>."""

from .config import Config, initialize_framework

# Import optional components with error handling
try:
    from .storage import StorageBackend
except ImportError:
    StorageBackend = None

try:
    from .registry import ComponentRegistry
except ImportError:
    ComponentRegistry = None

from .schemas import ContentItem, CategoryProposal, AnalysisResult

try:
    from .logging import setup_logging
except ImportError:
    setup_logging = None

__all__ = [
    'Config',
    'initialize_framework', 
    'StorageBackend',
    'ComponentRegistry',
    'ContentItem',
    'CategoryProposal', 
    'AnalysisResult',
    'setup_logging'
] 
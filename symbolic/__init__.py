"""Symbolic system integration for Perspective D<cide>."""

# Import with error handling for missing modules
try:
    from .integration import SymbolicAnalyzer, initialize_symbolic_system
except ImportError as e:
    print(f"Warning: Could not import symbolic integration: {e}")
    SymbolicAnalyzer = None
    initialize_symbolic_system = None

try:
    from .glyphs import GlyphLookup
except ImportError as e:
    print(f"Warning: Could not import glyphs module: {e}")
    GlyphLookup = None

try:
    from .logic import SymbolicCollapse, TarotMapping
except ImportError as e:
    print(f"Warning: Could not import logic module: {e}")
    SymbolicCollapse = None
    TarotMapping = None

__all__ = [
    'SymbolicAnalyzer',
    'initialize_symbolic_system',
    'GlyphLookup',
    'SymbolicCollapse', 
    'TarotMapping'
] 
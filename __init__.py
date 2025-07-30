"""
Perspective D<cide> - LLM-Oriented Symbolic Computing Framework

A comprehensive framework for analyzing asymmetrical data and generating
usable tables based on user prompts, with advanced symbolic reasoning capabilities.
"""

__version__ = "0.2.0"
__author__ = "Perspective D<cide> Team"

# Core imports - these should always work
try:
    from .core.config import Config
    from .core import initialize_framework
except ImportError as e:
    print(f"Warning: Core modules not available: {e}")
    Config = None
    initialize_framework = None

# Symbolic system imports with error handling
try:
    from .symbolic import (
        SymbolicAnalyzer,
        initialize_symbolic_system,
        GlyphLookup,
        SymbolicCollapse,
        TarotMapping
    )
except ImportError as e:
    print(f"Warning: Symbolic system not available: {e}")
    SymbolicAnalyzer = None
    initialize_symbolic_system = None
    GlyphLookup = None
    SymbolicCollapse = None
    TarotMapping = None

# ETX framework imports with error handling
try:
    from .etx import ETXFramework, CategorizationBuilder
except ImportError as e:
    print(f"Warning: ETX framework not available: {e}")
    ETXFramework = None
    CategorizationBuilder = None

# UI scaffold imports with error handling
try:
    from .ui_scaffold import UIScaffoldManager, FaceToScaffold
except ImportError as e:
    print(f"Warning: UI scaffold not available: {e}")
    UIScaffoldManager = None
    FaceToScaffold = None

__all__ = [
    # Core
    'Config',
    'initialize_framework',
    
    # Symbolic system
    'SymbolicAnalyzer',
    'initialize_symbolic_system',
    'GlyphLookup',
    'SymbolicCollapse',
    'TarotMapping',
    
    # ETX framework
    'ETXFramework',
    'CategorizationBuilder',
    
    # UI scaffold
    'UIScaffoldManager',
    'FaceToScaffold'
]

def initialize_framework_with_symbolic(config=None):
    """
    Initialize the framework with symbolic system support.
    
    Args:
        config: Configuration object. If None, loads from environment.
    """
    if initialize_framework is None:
        print("Warning: initialize_framework not available")
        return
    
    # Initialize core framework
    initialize_framework(config)
    
    # Initialize symbolic system if available
    if initialize_symbolic_system and config:
        try:
            initialize_symbolic_system(config)
        except Exception as e:
            print(f"Warning: Could not initialize symbolic system: {e}")

# Convenience function for quick setup
def quick_setup():
    """
    Quick setup for basic framework usage.
    
    Returns:
        Configured framework instance
    """
    if Config is None:
        raise RuntimeError("Config class not available")
    
    config = Config()
    initialize_framework_with_symbolic(config)
    return config 
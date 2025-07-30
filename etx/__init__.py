"""
ETX Framework for Perspective D<cide>.

Provides Emergent Taxonomy framework for content categorization and analysis.
"""

# Core ETX components
try:
    from .framework import ETXFramework
except ImportError as e:
    print(f"Warning: ETX framework not available: {e}")
    ETXFramework = None

# Builders
try:
    from .builders import CategorizationBuilder, CategoryNode, TaxonomyResult
except ImportError as e:
    print(f"Warning: ETX builders not available: {e}")
    CategorizationBuilder = None
    CategoryNode = None
    TaxonomyResult = None

# Engines
try:
    from .engines import (
        FastEmbedEngine, 
        MiniBatchKMeansEngine, 
        HDBSCANEngine,
        EmbeddingResult,
        ClusteringResult
    )
except ImportError as e:
    print(f"Warning: ETX engines not available: {e}")
    FastEmbedEngine = None
    MiniBatchKMeansEngine = None
    HDBSCANEngine = None
    EmbeddingResult = None
    ClusteringResult = None

# Plugins
try:
    from .plugins import (
        PluginManager,
        BaseETXPlugin,
        KeywordExtractionPlugin,
        SentimentAnalysisPlugin,
        LanguageDetectionPlugin
    )
except ImportError as e:
    print(f"Warning: ETX plugins not available: {e}")
    PluginManager = None
    BaseETXPlugin = None
    KeywordExtractionPlugin = None
    SentimentAnalysisPlugin = None
    LanguageDetectionPlugin = None

__all__ = [
    # Core framework
    'ETXFramework',
    
    # Builders
    'CategorizationBuilder',
    'CategoryNode',
    'TaxonomyResult',
    
    # Engines
    'FastEmbedEngine',
    'MiniBatchKMeansEngine',
    'HDBSCANEngine',
    'EmbeddingResult',
    'ClusteringResult',
    
    # Plugins
    'PluginManager',
    'BaseETXPlugin',
    'KeywordExtractionPlugin',
    'SentimentAnalysisPlugin',
    'LanguageDetectionPlugin'
] 
"""
Configuration management for the Perspective D<cide> framework.

Provides centralized configuration for all framework components including
storage, logging, processing, and component enablement.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

@dataclass
class Config:
    """Framework configuration."""
    
    # Basic settings
    framework_name: str = "perspective-dcide"
    version: str = "0.2.0"
    
    # Storage configuration
    storage_backend: str = "sqlite"
    storage_path: str = "~/.perspective_dcide"
    storage_config: Dict[str, Any] = field(default_factory=dict)
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Processing configuration
    enable_async: bool = True
    max_workers: int = 4
    batch_size: int = 1000
    
    # Component enablement
    enable_etx: bool = True
    enable_ml: bool = False
    enable_pipeline: bool = True
    enable_symbolic: bool = True  # Our glyph system
    
    # Symbolic system configuration (our existing glyph system)
    symbolic_config: Dict[str, Any] = field(default_factory=lambda: {
        "glyphs_path": "assets/glyphs.jsonl",
        "animations_path": "assets/animations.jsonl", 
        "icons_path": "assets/icons.jsonl",
        "enable_tarot_mapping": True,
        "enable_collapse_engine": True
    })
    
    # ETX-specific settings
    etx_config: Dict[str, Any] = field(default_factory=lambda: {
        "embedding_model": "bge-small-en",
        "clustering_type": "minibatch_kmeans",
        "consensus_threshold": 0.6,
        "min_confidence": 0.7
    })
    
    # ML-specific settings
    ml_config: Dict[str, Any] = field(default_factory=lambda: {
        "model_cache_dir": "~/.cache/perspective_dcide/models",
        "enable_gpu": False,
        "precision": "float32"
    })
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Expand paths
        self.storage_path = os.path.expanduser(self.storage_path)
        
        # Set up symbolic paths relative to package
        if self.enable_symbolic:
            package_root = Path(__file__).parent.parent
            self.symbolic_config["glyphs_path"] = str(package_root / "assets" / "glyphs.jsonl")
            self.symbolic_config["animations_path"] = str(package_root / "assets" / "animations.jsonl")
            self.symbolic_config["icons_path"] = str(package_root / "assets" / "icons.jsonl")
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        config = cls()
        
        # Framework settings
        config.framework_name = os.getenv("PERSPECTIVE_DCIDE_FRAMEWORK_NAME", config.framework_name)
        config.storage_backend = os.getenv("PERSPECTIVE_DCIDE_STORAGE_BACKEND", config.storage_backend)
        config.storage_path = os.getenv("PERSPECTIVE_DCIDE_STORAGE_PATH", config.storage_path)
        config.log_level = os.getenv("PERSPECTIVE_DCIDE_LOG_LEVEL", config.log_level)
        
        # Component enablement
        config.enable_etx = os.getenv("PERSPECTIVE_DCIDE_ENABLE_ETX", "true").lower() == "true"
        config.enable_ml = os.getenv("PERSPECTIVE_DCIDE_ENABLE_ML", "false").lower() == "true"
        config.enable_symbolic = os.getenv("PERSPECTIVE_DCIDE_ENABLE_SYMBOLIC", "true").lower() == "true"
        
        # Processing settings
        config.enable_async = os.getenv("PERSPECTIVE_DCIDE_ENABLE_ASYNC", "true").lower() == "true"
        config.max_workers = int(os.getenv("PERSPECTIVE_DCIDE_MAX_WORKERS", str(config.max_workers)))
        config.batch_size = int(os.getenv("PERSPECTIVE_DCIDE_BATCH_SIZE", str(config.batch_size)))
        
        return config

# Global configuration instance
_global_config: Optional[Config] = None

def initialize_framework(config: Optional[Config] = None) -> None:
    """
    Initialize the Perspective D<cide> framework.
    
    Args:
        config: Configuration object. If None, loads from environment.
    """
    global _global_config
    
    if config is None:
        config = Config.from_env()
    
    _global_config = config
    
    # Set up logging
    from .logging import setup_logging
    setup_logging({
        'level': config.log_level,
        'file': config.log_file
    })
    
    # Initialize storage
    from .storage import initialize as init_storage
    init_storage({
        'storage_backend': config.storage_backend,
        'storage_path': config.storage_path
    })
    
    # Initialize component registry
    from .registry import initialize as init_registry
    init_registry({
        'enable_etx': config.enable_etx,
        'enable_ml': config.enable_ml,
        'enable_symbolic': config.enable_symbolic
    })
    
    # Initialize symbolic system if enabled
    if config.enable_symbolic:
        try:
            from ..symbolic import initialize_symbolic_system
            initialize_symbolic_system(config)
        except ImportError:
            # Symbolic system not available, skip initialization
            pass

def get_config() -> Config:
    """Get the global configuration instance."""
    if _global_config is None:
        raise RuntimeError("Framework not initialized. Call initialize_framework() first.")
    return _global_config 
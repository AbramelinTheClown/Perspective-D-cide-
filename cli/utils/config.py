"""
Configuration loading utilities for the Gola CLI.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ConfigError(Exception):
    """Configuration error."""
    pass

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigError: If configuration file cannot be loaded
    """
    try:
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config:
            raise ConfigError(f"Empty configuration file: {config_path}")
        
        return config
        
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in configuration file {config_path}: {e}")
    except Exception as e:
        raise ConfigError(f"Error loading configuration from {config_path}: {e}")

def load_mode_config(mode: str, config_dir: Path) -> Dict[str, Any]:
    """
    Load mode-specific configuration.
    
    Args:
        mode: Mode name (general, fiction, technical, legal)
        config_dir: Configuration directory path
        
    Returns:
        Mode configuration dictionary
    """
    mode_config_path = config_dir / "modes" / f"{mode}.yaml"
    
    if not mode_config_path.exists():
        raise ConfigError(f"Mode configuration not found: {mode_config_path}")
    
    return load_config(mode_config_path)

def merge_configs(base_config: Dict[str, Any], mode_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge base configuration with mode-specific configuration.
    
    Args:
        base_config: Base configuration
        mode_config: Mode-specific configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    # Merge mode-specific settings
    if "tasks" in mode_config:
        merged["tasks"] = mode_config["tasks"]
    
    if "chunking" in mode_config:
        merged["chunking"] = mode_config["chunking"]
    
    if "deduplication" in mode_config:
        merged["deduplication"] = mode_config["deduplication"]
    
    if "validation" in mode_config:
        merged["validation"] = mode_config["validation"]
    
    if "routing" in mode_config:
        merged["routing"] = mode_config["routing"]
    
    if "schemas" in mode_config:
        merged["schemas"] = mode_config["schemas"]
    
    if "quality_metrics" in mode_config:
        merged["quality_metrics"] = mode_config["quality_metrics"]
    
    if "exports" in mode_config:
        merged["exports"] = mode_config["exports"]
    
    return merged

def get_project_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Get project directory paths from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of project paths
    """
    paths_config = config.get("paths", {})
    project_root = Path.cwd()
    
    paths = {}
    for key, path_str in paths_config.items():
        if path_str.startswith("./"):
            paths[key] = project_root / path_str[2:]
        else:
            paths[key] = Path(path_str)
    
    return paths

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ConfigError: If configuration is invalid
    """
    required_sections = ["providers", "paths", "vector_db"]
    
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required configuration section: {section}")
    
    # Validate providers
    providers = config.get("providers", {})
    if not providers:
        raise ConfigError("No providers configured")
    
    # Validate paths
    paths = config.get("paths", {})
    required_paths = ["data", "logs"]
    for path_key in required_paths:
        if path_key not in paths:
            raise ConfigError(f"Missing required path: {path_key}")
    
    # Validate vector database
    vector_db = config.get("vector_db", {})
    if "type" not in vector_db:
        raise ConfigError("Vector database type not specified")

def get_env_config() -> Dict[str, str]:
    """
    Get configuration from environment variables.
    
    Returns:
        Dictionary of environment-based configuration
    """
    env_config = {}
    
    # API Keys
    api_keys = [
        "LLAMA_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY", 
        "ANTHROPIC_API_KEY_1", "COHERE_API_KEY", "DEEPSEEK_API_KEY_1",
        "GROK_API_KEY_0", "HF_TOKEN", "HUB_API_KEY"
    ]
    
    for key in api_keys:
        value = os.getenv(key)
        if value:
            env_config[key] = value
    
    # Database configuration
    db_keys = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"]
    for key in db_keys:
        value = os.getenv(key)
        if value:
            env_config[key] = value
    
    # Other configuration
    other_keys = [
        "LM_STUDIO_API_LOCAL", "GITHUB_BASE_URL", "GROK_API_ENDPOINT",
        "GROK_MODEL", "XAI_SDK_ENABLED"
    ]
    
    for key in other_keys:
        value = os.getenv(key)
        if value:
            env_config[key] = value
    
    return env_config 
"""
Logging configuration for the Perspective D<cide> framework.

Provides centralized logging setup and configuration for all framework components.
"""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Set up logging configuration for the framework.
    
    Args:
        config: Logging configuration dictionary. If None, uses default configuration.
    """
    
    if config is None:
        config = {
            'level': os.getenv('PERSPECTIVE_DCIDE_LOG_LEVEL', 'INFO'),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': os.getenv('PERSPECTIVE_DCIDE_LOG_FILE'),
            'console': True
        }
    
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent.parent / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': config.get('level', 'INFO'),
                'formatter': 'standard',
                'stream': sys.stdout
            }
        },
        'loggers': {
            'perspective_dcide': {
                'level': config.get('level', 'INFO'),
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': 'WARNING',
            'handlers': ['console']
        }
    }
    
    # Add file handler if specified
    if config.get('file'):
        log_config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'level': config.get('level', 'INFO'),
            'formatter': 'detailed',
            'filename': config['file'],
            'mode': 'a'
        }
        log_config['loggers']['perspective_dcide']['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(log_config)
    
    # Set up framework logger
    logger = logging.getLogger('perspective_dcide')
    logger.info(f"Logging initialized with level: {config.get('level', 'INFO')}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f'perspective_dcide.{name}') 
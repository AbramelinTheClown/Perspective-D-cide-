"""
Logging utilities for the Gola CLI.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Install rich traceback handler
install()

console = Console()

class GolaLogger:
    """Custom logger for Gola system."""
    
    def __init__(self, name: str = "gola"):
        self.logger = logging.getLogger(name)
        self.console = console
    
    def setup(self, level: str = "INFO", log_file: Optional[Path] = None, 
              log_format: Optional[str] = None) -> None:
        """
        Setup logging configuration.
        
        Args:
            level: Logging level
            log_file: Path to log file
            log_format: Log format string
        """
        # Set log level
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with rich formatting
        console_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True
        )
        console_handler.setLevel(log_level)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            if log_format:
                formatter = logging.Formatter(log_format)
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
    
    def exception(self, message: str) -> None:
        """Log exception with traceback."""
        self.logger.exception(message)

def setup_logging(level: str = "INFO", config: Optional[Dict[str, Any]] = None) -> GolaLogger:
    """
    Setup logging for the Gola system.
    
    Args:
        level: Logging level
        config: Logging configuration
        
    Returns:
        Configured logger
    """
    logger = GolaLogger()
    
    if config:
        log_file = config.get("file")
        if log_file:
            log_file = Path(log_file)
        
        log_format = config.get("format")
        logger.setup(level, log_file, log_format)
    else:
        logger.setup(level)
    
    return logger

def get_logger(name: str = "gola") -> GolaLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return GolaLogger(name)

# Global logger instance
gola_logger = get_logger()

def log_function_call(func_name: str, args: Dict[str, Any] = None, 
                     kwargs: Dict[str, Any] = None) -> None:
    """
    Log function call for debugging.
    
    Args:
        func_name: Function name
        args: Function arguments
        kwargs: Function keyword arguments
    """
    args_str = ""
    if args:
        args_str += f"args={args}"
    if kwargs:
        if args_str:
            args_str += ", "
        args_str += f"kwargs={kwargs}"
    
    gola_logger.debug(f"Calling {func_name}({args_str})")

def log_function_result(func_name: str, result: Any = None, 
                       error: Exception = None) -> None:
    """
    Log function result or error.
    
    Args:
        func_name: Function name
        result: Function result
        error: Function error
    """
    if error:
        gola_logger.error(f"{func_name} failed: {error}")
    else:
        gola_logger.debug(f"{func_name} completed successfully")
        if result is not None:
            gola_logger.debug(f"{func_name} returned: {result}")

def log_pipeline_step(step_name: str, status: str = "started", 
                     details: Dict[str, Any] = None) -> None:
    """
    Log pipeline step status.
    
    Args:
        step_name: Pipeline step name
        status: Step status (started, completed, failed)
        details: Additional details
    """
    message = f"Pipeline step '{step_name}' {status}"
    if details:
        message += f" - {details}"
    
    if status == "failed":
        gola_logger.error(message)
    elif status == "completed":
        gola_logger.info(message)
    else:
        gola_logger.info(message)

def log_model_call(provider: str, model: str, task: str, 
                   tokens_in: int = None, tokens_out: int = None,
                   cost: float = None, error: Exception = None) -> None:
    """
    Log model API call.
    
    Args:
        provider: Model provider
        model: Model name
        task: Task type
        tokens_in: Input tokens
        tokens_out: Output tokens
        cost: API cost
        error: API error
    """
    if error:
        gola_logger.error(f"Model call failed: {provider}/{model} for {task} - {error}")
    else:
        details = []
        if tokens_in:
            details.append(f"tokens_in={tokens_in}")
        if tokens_out:
            details.append(f"tokens_out={tokens_out}")
        if cost:
            details.append(f"cost=${cost:.4f}")
        
        details_str = ", ".join(details) if details else ""
        gola_logger.info(f"Model call: {provider}/{model} for {task} {details_str}")

def log_quality_metrics(metrics: Dict[str, Any], dataset: str = None) -> None:
    """
    Log quality metrics.
    
    Args:
        metrics: Quality metrics dictionary
        dataset: Dataset name
    """
    dataset_str = f" for {dataset}" if dataset else ""
    gola_logger.info(f"Quality metrics{dataset_str}: {metrics}")

def log_sync_status(sync_type: str, status: str, details: Dict[str, Any] = None) -> None:
    """
    Log sync operation status.
    
    Args:
        sync_type: Type of sync operation
        status: Sync status
        details: Additional details
    """
    message = f"Sync {sync_type}: {status}"
    if details:
        message += f" - {details}"
    
    if status == "failed":
        gola_logger.error(message)
    else:
        gola_logger.info(message) 
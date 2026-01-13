"""
Logging configuration for PyAudioEditor.
Provides a configured logger instance for the application.
"""
import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "PyAudacity",
    level: int = logging.DEBUG,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string (optional)
    
    Returns:
        Configured logger instance
    """
    log = logging.getLogger(name)
    log.setLevel(level)
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console Handler
    if not log.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        formatter = logging.Formatter(format_string)
        console_handler.setFormatter(formatter)
        
        log.addHandler(console_handler)
    
    return log


# Global logger instance
logger = setup_logger()

import logging
import sys
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up logger with Rich formatting and optional TensorBoard."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with Rich
    console_handler = RichHandler(
        show_time=True,
        show_path=False,
        markup=True
    )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler if log_dir is provided
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_dir / f"{name}_{timestamp}.log"
        )
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing logger by name."""
    return logging.getLogger(name)
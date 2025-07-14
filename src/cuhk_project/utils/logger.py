# src/cuhk_project/utils/logger.py
import functools
import logging
import time
from pathlib import Path
from typing import Optional, Union
import os

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class DummyNN:
        Conv2d = object
    nn = DummyNN()


def configure_logging(
    version: str = "0.1.0",
    log_dir: Union[Path, str, None] = None,
    module: Optional[str] = None,
    console_level: int = logging.WARNING,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """Configure hierarchical logging system with separate level controls
    
    Args:
        version: Application version string
        log_dir: Directory path for log files
        module: Module name for logger identification
        console_level: Minimum logging level for console output
        file_level: Minimum logging level for file output
    """
    log_dir = Path(log_dir) if log_dir else Path(__file__).parent.parent / "logs"
    try:
        log_dir.mkdir(exist_ok=True, mode=0o755)
    except Exception as e:
        raise RuntimeError(f"Failed to create log directory {log_dir}: {str(e)}")
    
    logger = logging.getLogger(module or "cuhk_project")
    logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers control filtering

    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 1. Debug log (all messages DEBUG and above)
    debug_handler = logging.FileHandler(log_dir / "debug.log")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    # 2. Info log (INFO and above)
    info_handler = logging.FileHandler(log_dir / "info.log")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s"
    ))
    
    # 3. Error log (ERROR and CRITICAL only)
    error_handler = logging.FileHandler(log_dir / "error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d - %(message)s"
    ))
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColorFormatter())
    
    logger.addHandler(debug_handler)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    return logger


class ColorFormatter(logging.Formatter):
    """Custom formatter that adds ANSI color codes to log levels"""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[41m'  # Red background
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        message = super().format(record)
        return f"{color}{message}{self.RESET}" if color else message


def log_exec_time(func):
    """Decorator to log function execution time
    
    Args:
        func: Function to be wrapped
    Returns:
        Wrapped function with timing logic
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logging.info(f"Function {func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper


def log_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    level: int = logging.INFO
) -> None:
    """Log tensor statistics including shape and value distribution
    
    Args:
        tensor: PyTorch tensor to analyze
        name: Descriptive name for the tensor
        level: Logging level for the output
    """
    logger = logging.getLogger("tensor")
    logger.log(
        level,
        f"{name} - shape: {tuple(tensor.shape)} | "
        f"mean: {tensor.mean().item():.4f} | "
        f"std: {tensor.std().item():.4f} | "
        f"min/max: {tensor.min().item():.4f}/{tensor.max().item():.4f}"
    )


def log_conv_layer(layer: torch.nn.Conv2d) -> None:
    """Log convolutional layer configuration parameters
    
    Args:
        layer: Conv2d layer to inspect
    """
    logger = logging.getLogger("cnn.debug")
    logger.info(
        f"Conv{layer.kernel_size} | "
        f"in/out: {layer.in_channels}/{layer.out_channels} | "
        f"stride: {layer.stride} | "
        f"padding: {layer.padding} | "
        f"dilation: {layer.dilation}"
    )


def set_log_level(level: Union[str, int]) -> None:
    """Dynamically adjust the logging level with improved type safety
    
    Args:
        level: Either string name (DEBUG/INFO/etc) or logging level constant
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    if isinstance(level, str):
        level = level.upper()
        if level not in level_map:
            valid_levels = ', '.join(level_map.keys())
            raise ValueError(f"Invalid log level '{level}'. Must be one of: {valid_levels}")
        level = level_map[level]
    
    logging.getLogger().setLevel(level)
    current_level = logging.getLevelName(level) if isinstance(level, int) else level
    logging.info("Logging level changed to: %s", current_level)



# Initialize default logger instance
logger = configure_logging()

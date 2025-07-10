# src/cuhk_project/utils/logger.py
import functools
import logging
import time
from pathlib import Path
from typing import Optional
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


def configure_logging(version: str, log_dir: Path = None, module: str = None):
    """Centralized logging configuration"""
    log_dir = log_dir or Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(module or "cuhk_project")
    
    # Remove existing handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Logger with different modules
    log_file = log_dir / f"{module or 'main'}.log" if module else log_dir / "cuhk_project.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter())
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

class ColorFormatter(logging.Formatter):
    """Log with color setting"""
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
    """Mark executing time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logging.info(f"{func.__name__} executed in {elapsed:.4f}s")
        return result
    return wrapper

def log_tensor(tensor: torch.Tensor, name: str = "tensor", level=logging.INFO):
    """Mark tensor information"""
    logger = logging.getLogger("tensor")
    logger.log(level, 
        f"{name} - shape: {tuple(tensor.shape)} | "
        f"mean: {tensor.mean().item():.4f} | "
        f"std: {tensor.std().item():.4f} | "
        f"min/max: {tensor.min().item():.4f}/{tensor.max().item():.4f}"
    )

def log_conv_layer(layer: torch.nn.Conv2d):
    """Mark convolution arguments"""
    logger = logging.getLogger("cnn.debug")
    logger.info(
        f"Conv{layer.kernel_size} | "
        f"in/out: {layer.in_channels}/{layer.out_channels} | "
        f"stride: {layer.stride} | "
        f"padding: {layer.padding}"
    )

def set_log_level(level: str):
    """Actively setup log level
    
    Args:
        level: DEBUG/INFO/WARNING/ERROR/CRITICAL
    """
    level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    logging.info("Log level changed to: %s", level)

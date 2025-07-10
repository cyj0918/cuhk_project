# src/cuhk_project/__init__.py
"""Top-level package for CUHK Project."""
from .utils.logger import configure_logging, set_log_level

__author__ = """Yu Jhen CHEN"""
__email__ = 'joonie.jhen@gmail.com'
__version__ = '0.1.0'

# Initialize logging
logger = configure_logging(__version__)

# Public API
__all__ = ['logger', 'set_log_level']

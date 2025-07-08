"""Top-level package for CUHK Project."""
import logging
import os
import sys
from pathlib import Path

__author__ = """Yu Jhen CHEN"""
__email__ = 'joonie.jhen@gmail.com'
__version__ = '0.1.0'

def _configure_logging():
    """Function of log setup"""
    # Make up log direction
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Log level setting to "INFO"
    log_level = os.getenv("CUHK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)
    
    # Setup log
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "cuhk_project.log"),
            logging.StreamHandler(sys.stdout)
        ],
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Logger including level
    logger = logging.getLogger(__name__)
    logger.info("Initialized cuhk_project package (version: %s)", __version__)
    logger.debug("Logging system configured with level: %s", log_level)

# Automatedly setup the log 
_configure_logging()

# Public API
def set_log_level(level: str):
    """Actively setup log level
    
    Args:
        level: DEBUG/INFO/WARNING/ERROR/CRITICAL
    """
    level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    logging.info("Log level changed to: %s", level)

# Output logger
logger = logging.getLogger(__name__)
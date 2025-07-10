# src/cuhk_project/utils/logger.py
import logging
import os
import sys
from pathlib import Path

def configure_logging(version: str):
    """Centralized logging configuration"""
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_level = os.getenv("CUHK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "cuhk_project.log"),
            logging.StreamHandler(sys.stdout)
        ],
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Initialized logging system (version: %s)", version)
    return logger

def set_log_level(level: str):
    """Actively setup log level
    
    Args:
        level: DEBUG/INFO/WARNING/ERROR/CRITICAL
    """
    level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    logging.info("Log level changed to: %s", level)

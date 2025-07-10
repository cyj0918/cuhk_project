# src/cuhk_project/core/config.py
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class BaseConfig:
    """Basic Config"""
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    LOG_DIR: Path = PROJECT_ROOT / "logs"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    
    DEBUG: bool = os.getenv("CUHK_DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("CUHK_LOG_LEVEL", "INFO").upper()

@dataclass 
class CNNConfig(BaseConfig):
    """CNN Module Config"""
    DEFAULT_KERNEL_SIZE: int = 3
    DEFAULT_STRIDE: int = 1
    DEFAULT_PADDING: str = "auto"  # "same"|"valid"|"auto"
    
    # Image Process Deafult Arguments
    IMAGE_MEAN: tuple = (0.485, 0.456, 0.406)  # ImageNet Standard
    IMAGE_STD: tuple = (0.229, 0.224, 0.225)
    
    @property
    def padding_value(self):
        return (self.DEFAULT_KERNEL_SIZE - 1) // 2 if self.DEFAULT_PADDING == "auto" else 0

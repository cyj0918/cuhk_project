from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union
import torch
from ... import logger

class Base(ABC):
    """Abstract method of all processors"""
    def __init__(self, config: dict):
        """Initialize the processor

        Args:
            config: Dictionary of processor
        """
        self.config = config
        self._validate_config()
        logger.debug(f"Initialized {self.__class__.__name__} with config: {config}")

    def _validate_config(self) -> None:
        """Validate of configurations"""
        required_keys = self.required_config_keys()
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(
                f"Missing required config keys for {self.__class__.__name__}: {missing_keys}"
            )

    @staticmethod
    @abstractmethod
    def required_config_keys() -> list:
        """Return the required configuration keys"""
        return []

    @abstractmethod
    def process(self, image: torch.Tensor) -> torch.Tensor:
        """Process of the image and return results
        
        Args:
            image: Input image tensors(1xCxHxW)
            
        Returns:
            Processed image tensors
        """
        pass
    
    def save_result(
        self,
        image: torch.Tensor,
        path: Union[str, Path],
        format: Optional[str] = None
    ) -> None:
        """Save processed results
        
        Args:
            image: The tensors that are going to saved
            path: The path to save
            format: Optional file format
        """
        try:
            path = Path(path)
            if format is None:
                format = path.suffix[1:] if path.suffix else "pt"
            
            if format.lower() in ("pt", "pth"):
                torch.save(image, path)
            else:
                from ..utils.image_io import save_as_image
                save_as_image(image, path, format=format)
                
            logger.info(f"Saved result to {path}")
            
        except Exception as e:
            logger.exception(f"Failed to save result to {path}")
            raise

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
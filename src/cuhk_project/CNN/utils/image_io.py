import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from typing import Optional, Tuple, Union
from ...__init__ import logger 

def load_image(
        image_path: Union[str, Path],
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True
) -> torch.Tensor:
    """Import image and turn into PyTorch tensor

    Args:
        image_path: Path of image
        target_size: Optional target size including width and height
        normalize: Normalize into [0,1]

    Returns:
        torch.Tensor: [1, C, H, W] image tensors

    Raises:
        FileNotFoundError: When file is not found
        ValueError: When file format is not matched
        RuntimeError: When error occurred while processing
    """
    try:
        # Check
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        logger.debug(f"Loading image: {image_path}")

        # Use PIL to load image
        with Image.open(image_path) as img:
            img = img.convert('RGB')

            # Resize
            if target_size is not None:
                if len(target_size) != 2:
                    raise ValueError("Target size must be (width, height) tuple.")
                img = img.resize(target_size)

            # Turn into numpy array
            img_array = np.array(img, dtype=np.float32)

        # Transpose HWC to CHW
        img_array = img_array.transpose(2, 0, 1)

        # Normalize
        if normalize:
            img_array = img_array / 255.0

        # Turn into PyTorch tensors and add a batch dimension
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        logger.info(f"Successfully loaded image: {image_path}")
        return img_tensor
    
    except UnidentifiedImageError as e:
        logger.error(f"Unsupported image format: {image_path}")
        raise ValueError(f"Unsupported image format: {image_path}") from e
    except FileNotFoundError as e:
        logger.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error loading image: {image_path}")
        raise RuntimeError(f"Failed to load image: {image_path}") from e
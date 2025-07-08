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
    
def save_tensor(
        tensor: torch.Tensor,
        output_path: Union[str, Path],
        format: str = "pt"
) -> None:
    """Save tensor to files

    Args:
        tensor: The PyTorch tensor that is going to be saved
        output_path: The output path of files
        format: ("pt" | "npy")

    Raises:
        ValueError: When input is not a tensor or format is wrong
        IOError: When the file is saved wrong 
    """
    try:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Saving tensor to {output_path}")
        
        if format == "pt":
            torch.save(tensor, output_path)
        elif format == "npy":
            np.save(output_path, tensor.numpy())
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Successfully saved tensor to {output_path}")
    
    except Exception as e:
        logger.exception(f"Failed to save tensor to {output_path}.")
        raise IOError(f"Failed to save tensor: {output_path}.") from e
    
def save_as_image(
    tensor: torch.Tensor,
    output_path: Union[str, Path],
    format: str = "JPEG",
    quality: int = 95,
    denormalize: bool = True
) -> None:
    """Save tensor to standard image format (JPEG or PNG)

    Args:
        tensor: Input tensor (CxHxW or 1xCxHxW)
        output_path: Output file path
        format: Image format (JPEG/PNG/BMP/TIFF)
        quality: Save quality (1-100)
        denormalize: Denormalize or not (0-255)
    
    Raises:
        ValueError: Input tensor or value error
        IOError: Image save failed
    """
    try:
        # Arguments validations
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")
            
        if tensor.dim() not in (3, 4):
            raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")
            
        if tensor.dim() == 4:
            if tensor.size(0) != 1:
                raise ValueError("Batch size must be 1 for image saving")
            tensor = tensor.squeeze(0)

        # Denormalize
        if denormalize:
            if tensor.min() >= 0 and tensor.max() <= 1:
                tensor = tensor * 255
            tensor = tensor.clamp(0, 255)

        # Turn into PIL image
        array = tensor.byte().cpu().numpy()
        if array.shape[0] == 1:  # 单通道转灰度
            image = Image.fromarray(array.squeeze(0), mode='L')
        elif array.shape[0] == 3:  # RGB
            image = Image.fromarray(array.transpose(1, 2, 0), mode='RGB')
        else:
            raise ValueError(f"Unsupported channel size: {array.shape[0]}")
        
        # Ensure ouput direction exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save image
        save_args = {}
        if format.upper() in ("JPEG", "JPG"):
            save_args["quality"] = quality
        elif format.upper() == "PNG":
            save_args["compress_level"] = min(9, int((100 - quality) / 10))
            
        image.save(output_path, format=format, **save_args)
        logger.info(f"Image saved to {output_path}")
    
    except Exception as e:
        logger.exception(f"Failed to save image to {output_path}.")
        raise IOError(f"Image save failed: {output_path}.") from e
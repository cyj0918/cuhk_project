import torch
import numpy as np
from PIL import Image
from typing import Optional

def load_image(
        image_path: str,
        target_size: Optional[tuple] = None,
        normalize: bool = True
) -> torch.Tensor:
    """Import image and turn into PyTorch tensor

    Args:
        image_path: Path of image
        target_size: Optional target size including width and height
        normalize: Normalize into [0,1]

    Returns:
        torch.Tensor: [1, C, H, W] image tensors
    """
    # Use PIL to load image
    img = Image.open(image_path).convert('RGB')

    # Resize
    if target_size is not None:
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

    return img_tensor
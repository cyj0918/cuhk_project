# src/cuhk_project/CNN/utils/visualization.py
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from pathlib import Path

def visualize_conv_results(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor, 
    kernel_size: int,
    out_channels: int,
    save_path: Path
):
    """Visualize convolution input and output channels"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Input image
    ax1.imshow(input_tensor.permute(1, 2, 0).cpu().numpy())
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Output features grid
    grid = make_grid(output_tensor.unsqueeze(1), nrow=4, normalize=True)
    ax2.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    ax2.set_title(f'Output Features\nk={kernel_size}, c={out_channels}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

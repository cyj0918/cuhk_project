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
    """Visualize convolution input and output channels
    
    Args:
        input_tensor: Input tensor [C,H,W]
        output_tensor: Output feature maps [C,H,W] 
        kernel_size: Used kernel size
        out_channels: Number of output channels
        save_path: Path to save visualization
    """
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    try:
        # Input image visualization
        if input_tensor.shape[0] == 3:  # RGB
            axes[0].imshow(input_tensor.permute(1, 2, 0).cpu().numpy())
        else:  # Grayscale
            axes[0].imshow(input_tensor[0].cpu().numpy(), cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # Output features grid
        grid = make_grid(output_tensor.unsqueeze(1), nrow=4, normalize=True)
        axes[1].imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[1].set_title(f'Output Features\nk={kernel_size}, c={out_channels}')
        axes[1].axis('off')

        # Save and close
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Visualization failed: {str(e)}") from e

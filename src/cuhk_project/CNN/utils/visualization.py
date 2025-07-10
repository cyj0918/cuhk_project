# src/cuhk_project/CNN/utils/visualization.py
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from pathlib import Path
from typing import Optional, Union
from .debug_utils import ConvDebugger

class ConvVisualizer:
    """Modular convolutional visualization tool providing both static and class methods"""
    
    @staticmethod
    def visualize_conv_results(
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        kernel_size: int,
        out_channels: int,
        save_path: Union[str, Path],
        annotate: bool = False,
        figsize: tuple = (12, 6),
        nrow: int = 4,
        inspect_matrix: bool = False
    ) -> Path:
        """Visualize convolution input and output results
        
        Args:
            input_tensor: Input tensor [C,H,W]
            output_tensor: Output feature maps [C,H,W]
            kernel_size: Convolution kernel size used
            out_channels: Number of output channels
            save_path: Save path (directories will be auto-created)
            annotate: Whether to annotate numerical values
            figsize: Figure size
            nrow: Number of feature maps per row
            inspect_matrix
        Returns:
            Actual saved path
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        try:
            # Plot input image
            ConvVisualizer._plot_tensor(axes[0], input_tensor, 'Input Image')
            
            # Plot output feature maps grid
            grid = make_grid(output_tensor.unsqueeze(1), nrow=nrow, normalize=True)
            ConvVisualizer._plot_tensor(axes[1], grid, 
                                      f'Output Features\nk={kernel_size}, c={out_channels}')
            
            if annotate:
                ConvDebugger.annotate_image(axes[0], input_tensor)
                ConvDebugger.annotate_image(axes[1], grid)
            
            if inspect_matrix:
                from .debug_utils import ConvDebugger
                ConvDebugger.print_matrix_values(
                    input_tensor, 
                    "Input Matrix Preview", 
                    region=(5,5)
                )
                ConvDebugger.print_matrix_values(
                    output_tensor,
                    "Output Matrix Preview",
                    region=(5,5)
                )
                
            fig.tight_layout()
            fig.savefig(save_path)
            return save_path
            
        except Exception as e:
            plt.close(fig)
            raise RuntimeError(f"Visualization failed: {str(e)}") from e
        finally:
            plt.close(fig)

    @staticmethod
    def _plot_tensor(ax, tensor: torch.Tensor, title: str) -> None:
        """Internal tensor plotting method"""
        tensor = tensor.detach().cpu()
        if tensor.dim() == 3 and tensor.shape[0] == 3:  # RGB
            ax.imshow(tensor.permute(1, 2, 0).numpy())
        else:  # Grayscale or single channel
            ax.imshow(tensor.squeeze().numpy(), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    @classmethod
    def create_visualizer(cls, config: Optional[dict] = None):
        """Factory method for configurable visualizer creation"""
        return cls(config or {})

# Maintain backward compatibility with original function interface
visualize_conv_results = ConvVisualizer.visualize_conv_results

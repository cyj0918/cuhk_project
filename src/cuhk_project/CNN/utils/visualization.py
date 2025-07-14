# src/cuhk_project/CNN/utils/visualization.py
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.nn import functional as F
from torchvision.utils import make_grid
from pathlib import Path
from typing import Optional, Union, Tuple
from .debug_utils import ConvDebugger

class ConvVisualizer:
    """Modular convolutional visualization tool providing both static and class methods"""
    @staticmethod
    def _resize_grid(grid: torch.Tensor, max_dim: int = 1000) -> torch.Tensor:
        """Resize grid with aspect ratio preservation"""
        h, w = grid.shape[1], grid.shape[2]
        if max(h, w) <= max_dim:
            return grid
            
        scale = min(max_dim/h, max_dim/w)  # Maintain width:height
        new_h, new_w = int(h*scale), int(w*scale)
        
        return F.interpolate(
            grid.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    @staticmethod
    def _optimal_grid_params(feature_count: int, max_dim: int = 800) -> Tuple[int, int]:
        """Calculate optimal grid parameters to avoid oversized outputs"""
        nrow = min(8, feature_count)
        while True:
            cols = nrow
            rows = (feature_count + nrow - 1) // nrow
            est_height = rows * 64  # Height 64
            est_width = cols * 64   # Width 64
            
            if max(est_height, est_width) <= max_dim or nrow >= 16:
                return nrow, max_dim
            nrow += 2

    @staticmethod
    def _render_grid_safely(features: torch.Tensor, nrow: int, max_dim: int) -> np.ndarray:
        """Render grid in safe chunks"""
        grid = make_grid(
            features.unsqueeze(1),
            nrow=nrow,
            normalize=True,
            pad_value=0.5,
            scale_each=True
        )
       
        if max(grid.shape[1:]) > max_dim:
            scale = max_dim / max(grid.shape[1:])
            grid = F.interpolate(
                grid.unsqueeze(0),
                scale_factor=scale,
                mode='bilinear'
            ).squeeze(0)
        
        return grid.permute(1, 2, 0).cpu().numpy()

    @staticmethod
    def visualize_conv_results(
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        kernel_size: int,
        out_channels: int,
        save_path: Union[str, Path],
        annotate: bool = False,
        figsize: tuple = (16, 10),
        nrow: int = 4,
        inspect_matrix: bool = False,
        cmap: str = 'viridis',
        max_dim: int = 800, 
        channels_per_pages: int = 16,
        **kwargs
    ) -> Path:
        """Visualize convolution input and output results
        
        Args:
            input_tensor: Input tensor [C,H,W]
            output_tensor: Output feature maps [C,H,W]
            kernel_size: Convolution kernel size used
            out_channels: Number of output channels
            save_path: Save path (directories will be auto-created)
            annotate: Whether to annotate numerical values
            figsize: Figure size (adjusted for colorbar)
            nrow: Number of feature maps per row
            inspect_matrix: Whether to inspect matrix values
            cmap: Colormap for feature visualization (default: 'viridis')
        Returns:
            Actual saved path
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Input stats - min: {input_tensor.min().item():.4f}, max: {input_tensor.max().item():.4f}")
        print(f"[DEBUG] Output stats - min: {output_tensor.min().item():.4f}, max: {output_tensor.max().item():.4f}")
    
        if torch.all(output_tensor == 0):
            raise ValueError("All output values are zero! Check layer weights and input.")
        
        def safe_normalize(tensor):
            tensor = tensor.float()  
            min_val = tensor.min()
            max_val = tensor.max()
            if min_val == max_val:
                return torch.zeros_like(tensor)
            return (tensor - min_val) / (max_val - min_val)

        saved_paths = []
        total_pages = (out_channels + channels_per_pages - 1) // channels_per_pages

        for page in range(total_pages):
            start = page * channels_per_pages
            end = min((page + 1) * channels_per_pages, out_channels)
            
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            try:
                axes[0].imshow(input_tensor.permute(1, 2, 0).cpu().numpy())
                axes[0].set_title(f'Input Image | Page {page+1}/{total_pages}')
                axes[0].axis('off')

                features = output_tensor[start:end]
                normalized = torch.stack([safe_normalize(ch) for ch in features])
                
                grid = make_grid(
                    normalized.unsqueeze(1),
                    nrow=min(nrow, channels_per_pages),
                    pad_value=0.5,
                    normalize=False
                )

                if max(grid.shape[1:]) > max_dim:
                    grid = F.interpolate(
                        grid.unsqueeze(0),
                        size=(max_dim, max_dim),
                        mode='bilinear'
                    ).squeeze(0)

                im = axes[1].imshow(
                    grid.permute(1, 2, 0).cpu().numpy(),
                    cmap=cmap,
                    vmin=normalized.min().item(),
                    vmax=normalized.max().item()
                )
                cbar = fig.colorbar(im, ax=axes[1])
                cbar.set_label('Normalized Activation')
                axes[1].set_title(f'Channels {start}-{end-1} (k={kernel_size})')
                axes[1].axis('off')

                if annotate:
                    ConvDebugger.annotate_image(axes[0], input_tensor)
                    ConvDebugger.annotate_image(axes[1], grid)

                page_path = save_path.parent / f"{save_path.stem}_page{page+1}{save_path.suffix}"
                fig.tight_layout()
                fig.savefig(page_path, bbox_inches='tight', dpi=120)
                saved_paths.append(page_path)

            except Exception as e:
                plt.close(fig)
                raise RuntimeError(f"Page {page+1} visualization failed: {str(e)}") from e
            finally:
                plt.close(fig)

        return saved_paths if total_pages > 1 else saved_paths[0]

    @staticmethod
    def _plot_tensor(ax, tensor: torch.Tensor, title: str, cmap: str = None) -> None:
        """Internal tensor plotting method with enhanced visualization"""
        tensor = tensor.detach().cpu()
        if tensor.dim() == 3 and tensor.shape[0] == 3:  # RGB
            ax.imshow(tensor.permute(1, 2, 0).numpy())
        else:  # Grayscale or feature maps
            im = ax.imshow(
                tensor.squeeze().numpy(),
                cmap=cmap or 'viridis',
                vmin=tensor.min().item(),
                vmax=tensor.max().item()
            )
        ax.set_title(title)
        ax.axis('off')

    @classmethod
    def create_visualizer(cls, config: Optional[dict] = None):
        """Factory method for configurable visualizer creation"""
        return cls(config or {})

# Maintain backward compatibility with original function interface
visualize_conv_results = ConvVisualizer.visualize_conv_results

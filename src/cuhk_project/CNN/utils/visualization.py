# src/cuhk_project/CNN/utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from torchvision.utils import make_grid
from pathlib import Path
from typing import Optional, Union, Tuple, List
from .debug_utils import ConvDebugger
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.gridspec import GridSpec

# Initialize logger
logger = logging.getLogger(__name__)

class ConvVisualizer:
    """Enhanced convolutional visualization tool with optimized performance"""
    
    @staticmethod
    def _safe_normalize(tensor: torch.Tensor, method: str = 'histogram') -> torch.Tensor:
        """Enhanced normalization with histogram equalization"""
        tensor = tensor.float()
        if tensor.numel() == 0:
            return tensor
            
        if method == 'minmax':
            min_val = tensor.min()
            max_val = tensor.max()
            if min_val == max_val:
                return torch.zeros_like(tensor)
            return (tensor - min_val) / (max_val - min_val)
            
        elif method == 'histogram':  # Histogram equalization
            # Move to CPU for histogram computation
            tensor_cpu = tensor.cpu()
            hist = torch.histc(tensor_cpu, bins=256)
            cdf = hist.cumsum(0)
            cdf_min = cdf.min()
            
            # Avoid division by zero
            if cdf_min == cdf[-1]:
                return torch.zeros_like(tensor)
                
            # Apply histogram equalization
            cdf_normalized = (cdf - cdf_min) / (cdf[-1] - cdf_min)
            return cdf_normalized[tensor_cpu.long()].to(tensor.device)
    
    @staticmethod
    def _handle_dimensions(tensor: torch.Tensor) -> torch.Tensor:
        """Automatically handle batch and channel dimensions"""
        # Remove batch dimension if present
        if tensor.dim() == 4:
            if tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            else:
                logger.warning(f"Batch size >1 detected ({tensor.size(0)}). Using first sample.")
                tensor = tensor[0]
                
        # Ensure correct channel position
        if tensor.dim() == 3 and tensor.shape[0] not in [1, 3]:
            tensor = tensor.permute(1, 2, 0)
            
        return tensor

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
        max_dim: int = 1200,
        channels_per_page: int = 16,
        normalization: str = 'histogram',
        **kwargs
    ) -> Union[Path, List[Path]]:
        """
        Visualize convolution input and output with optimized performance
        
        Args:
            input_tensor: Input tensor [B,C,H,W] or [C,H,W]
            output_tensor: Output feature maps [B,C,H,W] or [C,H,W]
            kernel_size: Convolution kernel size
            out_channels: Number of output channels
            save_path: Save path (auto-creates directories)
            normalization: Normalization method ('histogram' or 'minmax')
            ... other parameters ...
        Returns:
            Saved file path(s)
        """
        # Convert to Path object and create directories
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving visualization to: {save_path}")
        
        # Handle tensor dimensions
        input_tensor = ConvVisualizer._handle_dimensions(input_tensor)
        output_tensor = ConvVisualizer._handle_dimensions(output_tensor)
        
        # Validate tensors
        if torch.all(output_tensor == 0):
            logger.error("All output values are zero! Check layer weights and input.")
            raise ValueError("All output values are zero")
        
        # Log tensor statistics
        logger.debug(f"Input stats: min={input_tensor.min().item():.4f}, max={input_tensor.max().item():.4f}")
        logger.debug(f"Output stats: min={output_tensor.min().item():.4f}, max={output_tensor.max().item():.4f}")
        
        saved_paths = []
        total_pages = (out_channels + channels_per_page - 1) // channels_per_page
        
        # Calculate global min/max for consistent colormap
        global_min = output_tensor.min().item()
        global_max = output_tensor.max().item()
        
        for page in range(total_pages):
            start = page * channels_per_page
            end = min((page + 1) * channels_per_page, out_channels)
            page_features = output_tensor[start:end]
            
            try:
                # === Create main figure with GridSpec for better layout control ===
                fig = plt.figure(figsize=figsize, constrained_layout=True)
                gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 2])
                
                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])
                
                fig.suptitle(f'Convolution Visualization (Page {page+1}/{total_pages})', fontsize=14)
                
                # === Plot input image ===
                input_img = input_tensor.permute(1, 2, 0).cpu().numpy()
                ax1.imshow(input_img)
                ax1.set_title(f'Input Image\nShape: {input_tensor.shape}')
                ax1.axis('off')
                
                # === Create feature grid ===
                # Calculate grid layout
                n_features = end - start
                ncols = min(nrow, n_features)
                nrows = (n_features + ncols - 1) // ncols
                
                # Create grid figure with constrained layout
                grid_fig = plt.figure(figsize=(figsize[0]*0.9, figsize[1]*0.7), 
                                     constrained_layout=True)
                
                # Create grid axes
                grid_axes = grid_fig.subplots(nrows=nrows, ncols=ncols)
                
                # Flatten axes if necessary
                if isinstance(grid_axes, np.ndarray):
                    grid_axes = grid_axes.flatten()
                else:
                    grid_axes = [grid_axes]
                
                # Plot each feature map
                im = None
                for idx, (ax, feature) in enumerate(zip(grid_axes, page_features)):
                    # Apply normalization
                    normalized = ConvVisualizer._safe_normalize(feature, normalization)
                    
                    # Plot with consistent colormap range
                    im = ax.imshow(
                        normalized.cpu().numpy(),
                        cmap=cmap,
                        vmin=global_min,
                        vmax=global_max
                    )
                    ax.set_title(f'Ch {start+idx}', fontsize=9)
                    ax.axis('off')
                
                # Remove empty axes
                for ax in grid_axes[n_features:]:
                    grid_fig.delaxes(ax)
                
                # Add colorbar to grid
                if im is not None:
                    grid_fig.colorbar(im, ax=grid_axes, fraction=0.02, pad=0.04)
                
                # Render grid to memory buffer
                canvas = FigureCanvasAgg(grid_fig)
                canvas.draw()
                grid_img = np.array(canvas.renderer.buffer_rgba())
                plt.close(grid_fig)
                
                # === Plot feature grid ===
                ax2.imshow(grid_img)
                ax2.set_title(f'Feature Maps {start}-{end-1}\nKernel: {kernel_size}x{kernel_size}')
                ax2.axis('off')
                
                # Add annotation if requested
                if annotate:
                    ConvDebugger.annotate_image(ax1, input_tensor)
                
                # === Save page ===
                page_save_path = save_path.parent / f"{save_path.stem}_page{page+1}{save_path.suffix}"
                fig.savefig(page_save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                
                saved_paths.append(page_save_path)
                logger.info(f"Saved visualization page {page+1} to: {page_save_path}")
                
            except Exception as e:
                logger.error(f"Page {page+1} visualization failed: {str(e)}")
                plt.close('all')
                raise
        
        return saved_paths[0] if len(saved_paths) == 1 else saved_paths

    @classmethod
    def create_visualizer(cls, config: Optional[dict] = None):
        """Factory method for configurable visualizer"""
        return cls(config or {})

# Backward compatibility alias
visualize_conv_results = ConvVisualizer.visualize_conv_results
# src/cuhk_project/CNN/utils/debug_utils.py
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class ConvDebugger:
    @staticmethod
    def print_matrix_values(tensor: torch.Tensor, 
                          name: str,
                          region: Optional[Tuple[int, int]] = None,
                          precision: int = 4):
        """Print matrix values with optional region selection
        
        Args:
            tensor: Input tensor to inspect
            name: Identifier for the matrix
            region: (height, width) of region to display (None for full matrix)
            precision: Decimal places to show
        """
        print(f"\nMatrix: {name} | Shape: {tensor.shape}")
        data = tensor.detach().cpu().numpy()
        
        if region:
            h, w = region
            data = data[..., :h, :w]  # Select region
            
        np.set_printoptions(precision=precision, suppress=True)
        print(data)
        
    @staticmethod
    def save_matrix_values(tensor: torch.Tensor,
                         file_path: str,
                         region: Optional[Tuple[int, int]] = None):
        """Save matrix values to file
        
        Args:
            tensor: Tensor to save
            file_path: Output file path
            region: (height, width) of region to save
        """
        data = tensor.detach().cpu().numpy()
        if region:
            h, w = region
            data = data[..., :h, :w]
            
        np.save(file_path, data)
        print(f"Matrix values saved to {file_path}.npy")

    @staticmethod
    def generate_matrix_report(tensor: torch.Tensor, 
                             name: str,
                             save_dir: Path,
                             region: tuple = (5,5)):
        """Generate detailed numerical report with file output"""
        # Ensure directory exists
        save_dir.mkdir(exist_ok=True)
        
        # Get tensor data
        data = tensor.detach().cpu().numpy()
        h, w = region
        
        # Create text report
        report = f"Matrix: {name}\nShape: {data.shape}\n"
        report += f"Region: First {h}x{w} values\n\n"
        report += "Values:\n"
        report += np.array2string(data[..., :h, :w], 
                                precision=4, 
                                suppress_small=True)
        
        # Save to file
        report_path = save_dir / f"{name.lower().replace(' ', '_')}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Generated numerical report at: {report_path}")
        return report_path
    
    @staticmethod
    def visualize_kernels(kernel_info, save_dir: Path):
        """Visualize and save kernel weights"""
        save_dir.mkdir(exist_ok=True)
        weights = kernel_info['weights']
        
        # Create grid of kernel visualizations
        grid = make_grid(weights, nrow=4, normalize=True, scale_each=True)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title('Convolution Kernels')
        plt.axis('off')
        
        save_path = save_dir / 'kernel_visualization.png'
        plt.savefig(save_path)
        plt.close()
        
        return save_path
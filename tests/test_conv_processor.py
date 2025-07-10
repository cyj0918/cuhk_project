# python -m tests.test_conv_processor --input tests/test_data/input1.jpg

import sys
import torch
import argparse
from pathlib import Path
from src.cuhk_project.CNN.processors.conv import Conv
from src.cuhk_project.CNN.utils.image_io import load_image, save_as_image
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_conv_processing(input_path: str, kernel_size: int, out_channels: int, stride: int):
    """Testing of completed Conv processing with dynamic parameters"""
    # Dynamic configuration
    config = {
        'in_channels': 3,  # Fixed for RGB images
        'out_channels': out_channels,
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': (kernel_size - 1) // 2  # Auto-calculate padding
    }
    print(f"Using config: {config}")
    
    processor = Conv(config)
    output_dir = Path("tests/test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Load and process image
    image_tensor = load_image(input_path)
    output_tensor = processor.process(image_tensor)
    
    # Save results with parameter info in filename
    param_str = f"k{kernel_size}_c{out_channels}"
    processor.save_result(output_tensor, output_dir/f"conv_output_{param_str}.pt")
    save_as_image(
        output_tensor[:, 0:1],
        output_dir/f"conv_feature_{param_str}.jpg",
        denormalize=True
    )
    print(f"Results saved with prefix: {param_str}")

def debug_conv_tensor(
    input_tensor: torch.Tensor,
    kernel_size: int = 3,
    out_channels: int = 16,
    stride: int = 1,
    save_all_channels: bool = False
) -> torch.Tensor:
    """Debug convolution effect on input tensor
    
    Args:
        input_tensor: Input tensor [C,H,W]
        kernel_size: Convolution kernel size
        out_channels: Number of output channels
        stride: Stride value
        save_all_channels: Whether to save all channel outputs as images
        
    Returns:
        Output tensor [out_channels,H,W]
    """
    config = {
        'in_channels': input_tensor.shape[0],
        'out_channels': out_channels,
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': (kernel_size - 1) // 2
    }
    
    processor = Conv(config)
    output_tensor = processor.process(input_tensor)
    
    if save_all_channels:
        output_dir = Path("tests/test_output")
        output_dir.mkdir(exist_ok=True)
        param_str = f"k{kernel_size}_c{out_channels}"
        
        for c in range(output_tensor.shape[0]):
            save_as_image(
                output_tensor[c:c+1],
                output_dir/f"conv_ch{c}_{param_str}.jpg",
                denormalize=True
            )
    
    return output_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test CNN convolution processor')
    
    # Input source selection
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input", type=str,
                           help="Path to input image")
    input_group.add_argument("--tensor-shape", type=int, nargs=3,
                           default=[3,256,256],
                           help="Random tensor shape [C,H,W]")
    
    # Convolution parameters
    parser.add_argument("--kernel-size", type=int, default=3,
                      help="Kernel size (3,5,7...)")
    parser.add_argument("--out-channels", type=int, default=16,
                      help="Number of output channels")
    parser.add_argument("--stride", type=int, default=1,
                      help="Stride value (1,2,3...)")
    parser.add_argument("--save-all", action="store_true",
                      help="Save all output channels as images")
    
    args = parser.parse_args()
    
    # Prepare input tensor
    if args.input:
        input_tensor = load_image(args.input)
    else:
        input_tensor = torch.rand(*args.tensor_shape)
    
    # Execute debug
    output = debug_conv_tensor(
        input_tensor=input_tensor,
        kernel_size=args.kernel_size,
        out_channels=args.out_channels,
        stride=args.stride,
        save_all_channels=args.save_all
    )
    
    # Print summary
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Value range: {output.min().item():.4f} ~ {output.max().item():.4f}")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test CNN convolution with dynamic parameters')
    parser.add_argument("--input", type=str, 
                       default="tests/test_data/input.jpg",
                       help="Path to input image")
    parser.add_argument("--kernel_size", type=int,
                       default=3,
                       help="Size of convolution kernel (3,5,7...)")
    parser.add_argument("--out_channels", type=int,
                       default=16,
                       help="Number of output channels")
    parser.add_argument("--stride", type=int,
                       default=1,
                       help="Convolution stride value (1,2,3...)")
    args = parser.parse_args()
    
    test_conv_processing(
        input_path=args.input,
        kernel_size=args.kernel_size,
        out_channels=args.out_channels,
        stride=args.stride
    )
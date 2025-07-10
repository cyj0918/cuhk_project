# python -m tests.test_conv_processor --input tests/test_data/input1.jpg

import sys
import torch
import argparse
from pathlib import Path
from src.cuhk_project.CNN.utils.debug_utils import ConvDebugger
from src.cuhk_project.CNN.processors.conv import Conv
from src.cuhk_project.CNN.utils.image_io import load_image, save_as_image
from src.cuhk_project.CNN.utils.visualization import visualize_conv_results
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

def create_parser():
    """Create and configure argument parser"""
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
    
    # Visualization control
    parser.add_argument("--visualize", action="store_true", default=True,
                  help="Enable visualization output (default: True)")
    parser.add_argument("--no-vis", action="store_false", dest="visualize",
                  help="Disable visualization output")

    # Test parameters
    parser.add_argument("--test-kernel", type=int, default=3,
                      help="Kernel size for visualization test")
    parser.add_argument("--test-channels", type=int, default=16,
                      help="Output channels for visualization test")
    parser.add_argument("--test-shape", type=int, nargs=3, default=[3,256,256],
                      help="Tensor shape for visualization test [C,H,W]")
    parser.add_argument("--run-test", action="store_true",
                      help="Run visualization test case")
    parser.add_argument("--inspect", action="store_true",
                  help="Inspect matrix values during processing")
    parser.add_argument("--matrix-region", type=int, nargs=2, default=[5,5],
                  help="Region size to display matrix values [height width]")
    parser.add_argument("--numerical", action="store_true",
                  help="Generate detailed numerical matrix reports")
    parser.add_argument("--kernels", action="store_true",
                  help="Inspect convolution kernel weights")
    
    return parser

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
    save_all_channels: bool = False,
    visualize: bool = True,
    inspect=False, 
    matrix_region=(5,5),
    debug_numerical=True,
    inspect_kernels=True
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
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0) 

    config = {
        'in_channels': 3 if input_tensor.size(1) == 3 else 1,
        'out_channels': out_channels,
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': (kernel_size - 1) // 2
    }
    
    processor = Conv(config)
    output_tensor = processor.process(input_tensor)

    if inspect_kernels:
        kernel_info = processor.get_kernel_info()
        kernel_path = ConvDebugger.visualize_kernels(
            kernel_info,
            Path("tests/test_output/kernel_inspection")
        )
        print(f"Kernel visualization saved to {kernel_path}")
        
        # Print kernel numerical values
        ConvDebugger.generate_matrix_report(
            kernel_info['weights'],
            "Kernel Weights",
            Path("tests/test_output/numerical_reports")
        )

         # Add kernel shape verification here
        print(f"Kernel shape: {kernel_info['weights'].shape}")
        print(f"Bias shape: {kernel_info['bias'].shape}")
    
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
    
    if visualize: 
        output_dir = Path("tests/test_output")
        save_path = output_dir/f"conv_vis_k{kernel_size}_c{out_channels}.png"
        visualize_conv_results(
            input_tensor=input_tensor.squeeze(0),
            output_tensor=output_tensor.squeeze(0),
            kernel_size=kernel_size,
            out_channels=out_channels,
            save_path=save_path
        )
        print(f"Visualization saved to {save_path}")

    if inspect:
        ConvDebugger.print_matrix_values(
            input_tensor,
            "Input Matrix",
            region=matrix_region
        )
        ConvDebugger.print_matrix_values(
            output_tensor,
            "Output Matrix", 
            region=matrix_region
        )

    if debug_numerical:
        from src.cuhk_project.CNN.utils.debug_utils import ConvDebugger
        report_dir = Path("tests/test_output/numerical_reports")
        ConvDebugger.generate_matrix_report(
            input_tensor,
            "Input Matrix",
            report_dir
        )
        ConvDebugger.generate_matrix_report(
            output_tensor,
            "Output Matrix", 
            report_dir
        )

    return output_tensor

def test_visualization(
    kernel_size: int = 3,
    out_channels: int = 16,
    tensor_shape: tuple = (3, 256, 256),
    visualize: bool = True
):
    """Test visualization functionality with configurable parameters
    
    Args:
        kernel_size: Convolution kernel size to test
        out_channels: Number of output channels to test
        tensor_shape: Shape of random test tensor [C,H,W]
        visualize: Whether to generate visualization
    """
    # Create test tensor
    test_img = torch.rand(1, *tensor_shape)
    output_dir = Path("tests/test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Run with parameters
    output = debug_conv_tensor(
        test_img,
        kernel_size=kernel_size,
        out_channels=out_channels,
        visualize=visualize
    )
    
    # Verify output
    vis_path = output_dir/f"conv_vis_k{kernel_size}_c{out_channels}.png"
    if visualize:
        assert vis_path.exists(), f"Visualization file {vis_path} not created"
    assert output.shape == (out_channels, *tensor_shape[1:]), "Output shape mismatch"

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    
    if args.run_test:
        test_visualization(
            kernel_size=args.test_kernel,
            out_channels=args.test_channels,
            tensor_shape=args.test_shape,
            visualize=args.visualize
        )
    else:
        # Prepare input tensor
        if args.input:
            input_tensor = load_image(args.input)
        else:
            input_tensor = torch.rand(*args.tensor_shape)
        
        # Execute debug
        output_tensor = debug_conv_tensor(
            input_tensor=input_tensor,
            kernel_size=args.kernel_size,
            out_channels=args.out_channels,
            stride=args.stride,
            save_all_channels=args.save_all,
            visualize=args.visualize,
            inspect=args.inspect,
            matrix_region=args.matrix_region,
            debug_numerical=args.numerical,
            inspect_kernels=args.kernels
        )
    
    # Print summary
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print(f"Value range: {output_tensor.min().item():.4f} ~ {output_tensor.max().item():.4f}")
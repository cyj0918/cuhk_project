import sys
import torch
from pathlib import Path
from src.cuhk_project.CNN.processors.conv import Conv
from src.cuhk_project.CNN.utils.image_io import load_image, save_as_image
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_conv_processing(input_path: str="tests/test_data/input.jpg"):
    """Testing of completed Conv processing"""
    # 1. Configuration of Conv processor
    config = {
        'in_channels': 3,
        'out_channels': 16,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    }
    processor = Conv(config)

    # Use passed input_path
    output_dir = Path("tests/test_output")
    output_dir.mkdir(exist_ok=True)
    image_tensor = load_image(input_path)  # use argument path
    print(f"Loaded image from: {input_path}")

    # Process the image
    output_tensor = processor.process(image_tensor)
    print(f"Output tensor shape: {output_tensor.shape}")

    # Save results
    # Save tensor results
    processor.save_result(output_tensor, output_dir/"conv_output.pt")
    # Visulize the feature image
    save_as_image(
        output_tensor[:, 0:1],  # Take the first feature image
        output_dir/"conv_feature.jpg",
        denormalize=True
    )
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, 
                       default="tests/test_data/input.jpg",
                       help="Path to input image")
    args = parser.parse_args()
    
    test_conv_processing(input_path=args.input)

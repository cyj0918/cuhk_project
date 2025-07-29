# python -m src.cuhk_project.scripts.evaluate_detection --data-dir data/yolo_mf_dataset --out-channels 16 --model-path models/test_4.pth --output-dir output/test_4
import argparse
import torch
from cuhk_project.detection.dataset import YOLOMFDataset
from cuhk_project.detection.model import SimpleDetectionModel
from cuhk_project.detection.evaluator import DetectionEvaluator
from cuhk_project.utils.logger import logger 

def parse_args():
        parser = argparse.ArgumentParser(description='Evaluate detection model')
        parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
        parser.add_argument('--out-channels', type=int, default=16,
                        help='Number of output channels')
        parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
        parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save evaluation results')
        parser.add_argument('--conf-thresh', type=float, default=0.6,
                        help='Confidence threshold for predictions')
        parser.add_argument('--iou-thresh', type=float, default=0.6,
                        help='IoU threshold for evaluation')
        parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for evaluation (cpu/cuda)')
        
        return parser.parse_args()

def main():
    args = parse_args()

    # 创建数据集和模型
    dataset = YOLOMFDataset(base_dir=args.data_dir, split="test")
    model = SimpleDetectionModel(out_channels=args.out_channels)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # 执行评估
    evaluator = DetectionEvaluator(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        device="cpu", 
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh
    )
    metrics = evaluator.evaluate()
    
    # 打印结果
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()
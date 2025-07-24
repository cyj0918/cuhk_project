# python -m src.cuhk_project.scripts.evaluate_detection --data-dir data/yolo_mf_dataset --batch-size 1 --out-channels 16 --model-path models/test_1.pth --output-dir output/test_1
import argparse
import torch
from cuhk_project.utils.logger import logger
from cuhk_project.detection.dataset import YOLOMFDataset
from cuhk_project.detection.model import SimpleDetectionModel
from cuhk_project.detection.evaluator import DetectionEvaluator
from cuhk_project.detection.visualizer import DetectionVisualizer

def main():
    # 配置日志
    logger.info("Starting object detection evaluation")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Evaluate object detection model")
    parser.add_argument("--data-dir", type=str, default="data/yolo_mf_dataset",
                        help="Dataset directory")
    parser.add_argument("--model-path", type=str, default="models/detection_model.pth",
                        help="Path to trained model")
    parser.add_argument("--out-channels", type=int, default=16,
                        help="Number of output channels in convolution layer (must match trained model)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--visualize-samples", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--output-dir", type=str, default="output/evaluation",
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    logger.info(f"Evaluation configuration: "
                f"data_dir={args.data_dir}, model_path={args.model_path}, "
                f"out_channels={args.out_channels}, visualize_samples={args.visualize_samples}")
    
    # 创建测试数据集
    try:
        test_dataset = YOLOMFDataset(
            base_dir=args.data_dir, split="test", target_size=(512, 96))
        logger.info(f"Test dataset size: {len(test_dataset)}")
    except Exception as e:
        logger.error(f"Failed to create test dataset: {str(e)}")
        return
    
    # 创建模型并加载权重
    try:
        # Force CPU evaluation for stability
        device = "cpu"
        logger.info(f"Forcing CPU evaluation for stability")
        
        model = SimpleDetectionModel(
            in_channels=3, 
            out_channels=args.out_channels,
            kernel_size=3
        ).to(device)
        
        # 加载训练好的权重
        model.load_state_dict(torch.load(args.model_path))
        logger.info(f"Model loaded successfully from {args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return
    
    # 运行评估器
    try:
        evaluator = DetectionEvaluator(
            model=model,
            test_dataset=test_dataset,
            batch_size=args.batch_size
        )
        metrics = evaluator.evaluate()
        
        # 打印评估结果
        logger.info("\n===== Evaluation Results =====")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"Average IoU: {metrics['avg_iou']:.4f}")
        logger.info(f"True Positives: {metrics['true_positives']}")
        logger.info(f"False Positives: {metrics['false_positives']}")
        logger.info(f"False Negatives: {metrics['false_negatives']}")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return
    
    # 运行可视化器
    try:
        visualizer = DetectionVisualizer(
            model=model,
            dataset=test_dataset,
            output_dir=args.output_dir
        )
        visualizer.visualize_dataset(num_samples=args.visualize_samples)
        logger.info(f"Visualizations saved to {args.output_dir}")
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        return
    
    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main()

# python -m src.cuhk_project.scripts.evaluate_detection --data-dir data/yolo_mf_dataset --out-channels 16 --model-path models/test_5.pth --output-dir output/test_5
import argparse
import torch
import json
from cuhk_project.detection.evaluator import DetectionEvaluator
from cuhk_project.detection.dataset import YOLOMFDataset
from cuhk_project.detection.model import SimpleDetectionModel
from cuhk_project.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO Detection Model')
    parser.add_argument('--data-dir', required=True, help='Dataset directory')
    parser.add_argument('--model-path', required=True, help='Model checkpoint path')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--out-channels', type=int, default=16, help='Model output channels')
    args = parser.parse_args()
    
    # 初始化数据集
    dataset = YOLOMFDataset(base_dir=args.data_dir, split='test', target_size=(96, 512))
    
    # 加载模型
    model = SimpleDetectionModel(
        in_channels=3,
        out_channels=args.out_channels,
        num_anchors=3,
        grid_size=(6, 32))
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # 执行评估
    evaluator = DetectionEvaluator(model, dataset, args.output_dir)
    metrics = evaluator.evaluate()
    
    # 输出结果 - 添加更详细的格式
    print("\n===== Evaluation Results =====")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print("\n--- Detailed Counts ---")
    print(f"True Positives:  {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Total Samples:   {len(metrics['sample_results'])}")
    print("=============================")
    
    # 保存详细结果 - 添加JSON格式导出
    torch.save(metrics, f"{args.output_dir}/evaluation_metrics.pt")
    with open(f"{args.output_dir}/evaluation_metrics.json", 'w') as f:
        json.dump({
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'true_positives': int(metrics['true_positives']),
            'false_positives': int(metrics['false_positives']),
            'false_negatives': int(metrics['false_negatives'])
        }, f, indent=2)
    
    print(f"Detailed results saved to {args.output_dir}/evaluation_metrics.pt and .json")

if __name__ == '__main__':
    main()
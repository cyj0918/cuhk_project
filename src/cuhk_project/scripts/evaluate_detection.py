# python -m src.cuhk_project.scripts.evaluate_detection --data-dir data/yolo_mf_dataset --out-channels 16 --model-path models/new1.pth --output-dir output/new1

import argparse
import torch
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from torch.profiler import profile, record_function, ProfilerActivity
import psutil
import gc
from collections import defaultdict

from cuhk_project.detection.evaluator import DetectionEvaluator
from cuhk_project.detection.dataset import YOLOMFDataset
from cuhk_project.detection.model import SimpleDetectionModel
from cuhk_project.utils.logger import logger

class PerformanceProfiler:
    """性能分析器 - 測量推理時間和資源使用"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.times = []
        self.memory_usage = []
        self.start_time = None
        self.process = psutil.Process()
        
    def start_timing(self):
        """開始計時"""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        
    def end_timing(self):
        """結束計時並記錄"""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        if self.start_time is not None:
            inference_time = end_time - self.start_time
            self.times.append(inference_time)
            
            # 記錄內存使用
            if self.device == "cuda" and torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            else:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
            self.memory_usage.append(memory_mb)
            
            return inference_time
        return 0.0
    
    def get_statistics(self) -> Dict:
        """獲取性能統計"""
        if not self.times:
            return {}
            
        times_ms = [t * 1000 for t in self.times]  # 轉換為毫秒
        
        return {
            'total_inference_time_ms': sum(times_ms),
            'avg_inference_time_ms': np.mean(times_ms),
            'median_inference_time_ms': np.median(times_ms),
            'min_inference_time_ms': np.min(times_ms),
            'max_inference_time_ms': np.max(times_ms),
            'std_inference_time_ms': np.std(times_ms),
            'fps': len(self.times) / sum(self.times) if sum(self.times) > 0 else 0,
            'avg_memory_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
            'max_memory_mb': np.max(self.memory_usage) if self.memory_usage else 0,
            'total_samples': len(self.times)
        }

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='Evaluate YOLO Detection Model')
    
    # 基本參數
    parser.add_argument('--data-dir', required=True, 
                       help='Dataset directory path')
    parser.add_argument('--model-path', required=True, 
                       help='Model checkpoint path (.pth file)')
    parser.add_argument('--output-dir', required=True, 
                       help='Output directory for evaluation results')
    
    # 模型參數
    parser.add_argument('--out-channels', type=int, default=16,
                       help='Model backbone output channels')
    parser.add_argument('--num-anchors', type=int, default=3,
                       help='Number of anchor boxes')
    parser.add_argument('--grid-size', type=str, default="6x32",
                       help='Grid size (format: height x width)')
    
    # 數據集參數
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--target-size', type=str, default="96x512",
                       help='Target image size (format: height x width)')
    
    # 評估參數
    parser.add_argument('--conf-thresh', type=float, default=0.4,
                       help='Confidence threshold for predictions')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                       help='IoU threshold for matching boxes')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to run evaluation on')
    
    # 高級功能
    parser.add_argument('--multi-threshold', action='store_true',
                       help='Evaluate at multiple thresholds')
    parser.add_argument('--batch-eval', type=int, default=1,
                       help='Batch size for evaluation (1=single image)')
    parser.add_argument('--profile', action='store_true',
                       help='Enable detailed profiling with PyTorch profiler')
    parser.add_argument('--max-samples', type=int, default=-1,
                       help='Maximum number of samples to evaluate (-1 for all)')
    parser.add_argument('--save-visualizations', action='store_true',
                       help='Save visualization images')
    
    return parser.parse_args()

def determine_device(device_arg: str) -> str:
    """確定最佳運行設備"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = 'cpu'
            logger.info("Auto-selected CPU device")
    else:
        device = device_arg
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    
    return device

def load_model_with_checkpoint(model_path: str, out_channels: int, num_anchors: int, 
                              grid_size: tuple, device: str) -> SimpleDetectionModel:
    """從檢查點加載模型"""
    logger.info(f"Loading model from: {model_path}")
    
    try:
        # 加載檢查點
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 檢查檢查點格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 新格式：包含訓練狀態的檢查點
            model_state = checkpoint['model_state_dict']
            anchor_boxes = checkpoint.get('anchor_boxes', None)
            saved_grid_size = checkpoint.get('grid_size', grid_size)
            epoch = checkpoint.get('epoch', -1)
            
            logger.info(f"Loaded checkpoint from epoch {epoch}")
            if anchor_boxes is not None:
                logger.info(f"Using saved anchor boxes: {anchor_boxes.tolist()}")
            if saved_grid_size != grid_size:
                logger.warning(f"Grid size mismatch: saved={saved_grid_size}, requested={grid_size}")
                grid_size = saved_grid_size
                
        else:
            # 舊格式：直接的state_dict
            model_state = checkpoint
            anchor_boxes = None
            logger.info("Loaded legacy checkpoint format")
        
        # 創建模型
        model = SimpleDetectionModel(
            in_channels=3,
            out_channels=out_channels,
            kernel_size=3,
            num_anchors=num_anchors,
            grid_size=grid_size,
            anchor_boxes=anchor_boxes
        )
        
        # 加載權重
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
        # 打印模型信息
        model_info = model.get_model_info()
        logger.info(f"Model loaded successfully:")
        logger.info(f"  - Parameters: {model_info['total_params']:,}")
        logger.info(f"  - Grid size: {model_info['grid_size']}")
        logger.info(f"  - Anchor boxes: {model_info['anchor_boxes']}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def create_dataset(args, grid_size: tuple, target_size: tuple) -> YOLOMFDataset:
    """創建評估數據集"""
    logger.info(f"Creating dataset: split={args.split}, target_size={target_size}, grid_size={grid_size}")
    
    try:
        dataset = YOLOMFDataset(
            base_dir=args.data_dir,
            split=args.split,
            target_size=target_size,
            grid_size=grid_size
        )
        
        logger.info(f"Dataset created: {len(dataset)} samples")
        
        # 限制樣本數量（如果指定）
        if args.max_samples > 0 and args.max_samples < len(dataset):
            logger.info(f"Limiting evaluation to {args.max_samples} samples")
            # 創建子集（這裡簡化處理，實際可以用Subset）
            dataset.samples = dataset.samples[:args.max_samples]
        
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        raise

def run_basic_evaluation(model, dataset, args, device: str) -> Dict:
    """運行基本評估"""
    logger.info("Starting basic evaluation...")
    
    # 創建性能分析器
    profiler = PerformanceProfiler(device)
    
    # 創建評估器
    evaluator = DetectionEvaluator(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        device=device
    )
    
    # 修改評估器以支持性能測量
    original_predict = model.predict
    
    def timed_predict(*args_inner, **kwargs):
        profiler.start_timing()
        result = original_predict(*args_inner, **kwargs)
        profiler.end_timing()
        return result
    
    # 替換predict方法
    model.predict = timed_predict
    
    try:
        # 執行評估
        metrics = evaluator.evaluate()
        
        # 獲取性能統計
        perf_stats = profiler.get_statistics()
        metrics.update(perf_stats)
        
        logger.info("Basic evaluation completed")
        return metrics
        
    finally:
        # 恢復原始方法
        model.predict = original_predict

def run_multi_threshold_evaluation(model, dataset, args, device: str) -> Dict:
    """運行多閾值評估"""
    logger.info("Starting multi-threshold evaluation...")
    
    evaluator = DetectionEvaluator(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        device=device
    )
    
    # 定義閾值範圍
    conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    iou_thresholds = [0.3, 0.5, 0.7, 0.9]
    
    multi_results = evaluator.evaluate_at_multiple_thresholds(
        conf_thresholds=conf_thresholds,
        iou_thresholds=iou_thresholds
    )
    
    logger.info("Multi-threshold evaluation completed")
    return multi_results

def run_detailed_profiling(model, dataset, args, device: str) -> Dict:
    """運行詳細性能分析"""
    logger.info("Starting detailed profiling...")
    
    profiling_results = {}
    
    # 選擇少量樣本進行詳細分析
    num_profile_samples = min(10, len(dataset))
    
    with profile(
        activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == 'cuda' else []), 
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        model.eval()
        with torch.no_grad():
            for i in range(num_profile_samples):
                image, target = dataset[i]
                image_tensor = image.unsqueeze(0).to(device, dtype=torch.float32)
                
                with record_function("model_prediction"):
                    # 模型現在返回三個輸出
                    _, _, _ = model(image_tensor)
    
    # 保存分析結果
    profile_output_path = Path(args.output_dir) / "profiling_trace.json"
    prof.export_chrome_trace(str(profile_output_path))
    
    # 獲取關鍵統計
    key_averages = prof.key_averages(group_by_stack_n=5)
    
    profiling_results['profile_trace_path'] = str(profile_output_path)
    profiling_results['top_operations'] = []
    
    for avg in key_averages[:10]:  # 前10個最耗時的操作
        profiling_results['top_operations'].append({
            'name': avg.key,
            'cpu_time_total': avg.cpu_time_total,
            'cuda_time_total': avg.cuda_time_total if hasattr(avg, 'cuda_time_total') else 0,
            'cpu_time_avg': avg.cpu_time / avg.count if avg.count > 0 else 0,
            'count': avg.count
        })
    
    logger.info(f"Detailed profiling completed, trace saved to: {profile_output_path}")
    return profiling_results

def benchmark_inference_speed(model, dataset, device: str, num_warmup: int = 10, num_benchmark: int = 100) -> Dict:
    """基準測試推理速度"""
    logger.info(f"Benchmarking inference speed (warmup={num_warmup}, benchmark={num_benchmark})...")
    
    model.eval()
    
    # 準備測試數據
    if len(dataset) == 0:
        logger.warning("Empty dataset for benchmarking")
        return {}
    
    # 使用第一個樣本進行測試
    sample_image, _ = dataset[0]
    sample_tensor = sample_image.unsqueeze(0).to(device, dtype=torch.float32)
    
    # 預熱
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(sample_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
    
        # 正式測試
        times = []
        with torch.no_grad():
            for _ in range(num_benchmark):
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                # 模型現在返回三個輸出
                _, _, _ = model(sample_tensor)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # 轉換為毫秒
    
    times = np.array(times)
    
    benchmark_results = {
        'warmup_runs': num_warmup,
        'benchmark_runs': num_benchmark,
        'avg_forward_time_ms': float(np.mean(times)),
        'median_forward_time_ms': float(np.median(times)),
        'min_forward_time_ms': float(np.min(times)),
        'max_forward_time_ms': float(np.max(times)),
        'std_forward_time_ms': float(np.std(times)),
        'forward_fps': float(1000.0 / np.mean(times)),
        'percentile_95_ms': float(np.percentile(times, 95)),
        'percentile_99_ms': float(np.percentile(times, 99))
    }
    
    logger.info(f"Benchmark completed: {benchmark_results['avg_forward_time_ms']:.2f}ms avg, {benchmark_results['forward_fps']:.1f} FPS")
    return benchmark_results

def save_comprehensive_results(results: Dict, args, output_dir: Path):
    """保存全面的評估結果"""
    logger.info("Saving comprehensive evaluation results...")
    
    # 確保輸出目錄存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存JSON格式結果
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # 2. 保存PyTorch格式結果
    pt_path = output_dir / "evaluation_results.pt"
    torch.save(results, pt_path)
    
    # 3. 生成人類可讀的報告
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("YOLO Detection Model Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        
        # 基本信息
        f.write("Configuration:\n")
        f.write(f"  Model Path: {args.model_path}\n")
        f.write(f"  Dataset: {args.data_dir} (split: {args.split})\n")
        f.write(f"  Device: {args.device}\n")
        f.write(f"  Confidence Threshold: {args.conf_thresh}\n")
        f.write(f"  IoU Threshold: {args.iou_thresh}\n\n")
        
        # 基本評估結果
        if 'basic_evaluation' in results:
            basic = results['basic_evaluation']
            f.write("Basic Evaluation Results:\n")
            f.write(f"  Precision: {basic.get('precision', 0):.4f}\n")
            f.write(f"  Recall: {basic.get('recall', 0):.4f}\n")
            f.write(f"  F1 Score: {basic.get('f1_score', 0):.4f}\n")
            
            if 'mAP' in basic:
                f.write(f"  mAP: {basic['mAP']:.4f}\n")
            if 'mAP_50' in basic:
                f.write(f"  mAP@0.5: {basic['mAP_50']:.4f}\n")
            if 'mAP_75' in basic:
                f.write(f"  mAP@0.75: {basic['mAP_75']:.4f}\n")
            
            f.write(f"  True Positives: {basic.get('true_positives', 0)}\n")
            f.write(f"  False Positives: {basic.get('false_positives', 0)}\n")
            f.write(f"  False Negatives: {basic.get('false_negatives', 0)}\n")
            f.write(f"  Total Images: {basic.get('total_images', 0)}\n\n")
        
        # 性能統計
        if 'basic_evaluation' in results:
            basic = results['basic_evaluation']
            if 'avg_inference_time_ms' in basic:
                f.write("Performance Statistics:\n")
                f.write(f"  Average Inference Time: {basic.get('avg_inference_time_ms', 0):.2f} ms\n")
                f.write(f"  FPS: {basic.get('fps', 0):.1f}\n")
                f.write(f"  Min/Max Time: {basic.get('min_inference_time_ms', 0):.2f}/{basic.get('max_inference_time_ms', 0):.2f} ms\n")
                f.write(f"  Memory Usage: {basic.get('avg_memory_mb', 0):.1f} MB (avg), {basic.get('max_memory_mb', 0):.1f} MB (max)\n\n")
        
        # 基準測試結果
        if 'benchmark' in results:
            bench = results['benchmark']
            f.write("Benchmark Results:\n")
            f.write(f"  Forward Pass Time: {bench.get('avg_forward_time_ms', 0):.2f} ± {bench.get('std_forward_time_ms', 0):.2f} ms\n")
            f.write(f"  Forward FPS: {bench.get('forward_fps', 0):.1f}\n")
            f.write(f"  95th Percentile: {bench.get('percentile_95_ms', 0):.2f} ms\n")
            f.write(f"  99th Percentile: {bench.get('percentile_99_ms', 0):.2f} ms\n\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logger.info(f"Results saved to:")
    logger.info(f"  - JSON: {json_path}")
    logger.info(f"  - PyTorch: {pt_path}")
    logger.info(f"  - Report: {report_path}")

def print_summary(results: Dict):
    """打印評估結果摘要"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    if 'basic_evaluation' in results:
        basic = results['basic_evaluation']
        
        print("\n📊 Detection Performance:")
        print(f"  Precision:     {basic.get('precision', 0):.4f}")
        print(f"  Recall:        {basic.get('recall', 0):.4f}")
        print(f"  F1 Score:      {basic.get('f1_score', 0):.4f}")
        
        if 'mAP' in basic:
            print(f"  mAP:           {basic['mAP']:.4f}")
        if 'mAP_50' in basic:
            print(f"  mAP@0.5:       {basic['mAP_50']:.4f}")
        if 'mAP_75' in basic:
            print(f"  mAP@0.75:      {basic['mAP_75']:.4f}")
        
        print(f"\n📈 Detection Counts:")
        print(f"  True Positives:  {basic.get('true_positives', 0)}")
        print(f"  False Positives: {basic.get('false_positives', 0)}")
        print(f"  False Negatives: {basic.get('false_negatives', 0)}")
        print(f"  Total Images:    {basic.get('total_images', 0)}")
        
        if 'avg_inference_time_ms' in basic:
            print(f"\n⚡ Inference Performance:")
            print(f"  Avg Time:        {basic.get('avg_inference_time_ms', 0):.2f} ms")
            print(f"  FPS:             {basic.get('fps', 0):.1f}")
            print(f"  Memory Usage:    {basic.get('avg_memory_mb', 0):.1f} MB")
    
    if 'benchmark' in results:
        bench = results['benchmark']
        print(f"\n🏃 Benchmark (Forward Pass Only):")
        print(f"  Avg Time:        {bench.get('avg_forward_time_ms', 0):.2f} ± {bench.get('std_forward_time_ms', 0):.2f} ms")
        print(f"  FPS:             {bench.get('forward_fps', 0):.1f}")
        print(f"  95th Percentile: {bench.get('percentile_95_ms', 0):.2f} ms")
    
    if 'multi_threshold' in results:
        print(f"\n🎯 Multi-threshold evaluation completed with {len(results['multi_threshold'])} threshold combinations")
    
    print("="*80)

def main():
    """主評估流程"""
    # 解析參數
    args = parse_arguments()
    
    # 設置日誌
    logger.info("="*60)
    logger.info("Starting YOLO Detection Model Evaluation")
    logger.info("="*60)
    
    try:
        # 1. 解析尺寸參數
        grid_height, grid_width = map(int, args.grid_size.split('x'))
        grid_size = (grid_height, grid_width)
        
        target_height, target_width = map(int, args.target_size.split('x'))
        target_size = (target_height, target_width)
        
        # 2. 確定設備
        device = determine_device(args.device)
        
        # 3. 加載模型
        model = load_model_with_checkpoint(
            args.model_path, args.out_channels, args.num_anchors, grid_size, device
        )
        
        # 4. 創建數據集
        dataset = create_dataset(args, grid_size, target_size)
        
        # 5. 創建輸出目錄
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 6. 收集所有評估結果
        all_results = {
            'evaluation_config': {
                'model_path': args.model_path,
                'data_dir': args.data_dir,
                'split': args.split,
                'device': device,
                'conf_thresh': args.conf_thresh,
                'iou_thresh': args.iou_thresh,
                'grid_size': grid_size,
                'target_size': target_size,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # 7. 基本評估
        logger.info("Running basic evaluation...")
        basic_results = run_basic_evaluation(model, dataset, args, device)
        all_results['basic_evaluation'] = basic_results
        
        # 8. 基準測試
        logger.info("Running benchmark test...")
        benchmark_results = benchmark_inference_speed(model, dataset, device)
        all_results['benchmark'] = benchmark_results
        
        # 9. 多閾值評估（可選）
        if args.multi_threshold:
            multi_results = run_multi_threshold_evaluation(model, dataset, args, device)
            all_results['multi_threshold'] = multi_results
        
        # 10. 詳細性能分析（可選）
        if args.profile:
            profiling_results = run_detailed_profiling(model, dataset, args, device)
            all_results['profiling'] = profiling_results
        
        # 11. 保存結果
        save_comprehensive_results(all_results, args, output_dir)
        
        # 12. 打印摘要
        print_summary(all_results)
        
        logger.info("="*60)
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
    finally:
        # 清理資源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()

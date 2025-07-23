import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from cuhk_project.utils.logger import logger
from .model import SimpleDetectionModel
from .dataset import YOLOMFDataset

class DetectionEvaluator:
    """目标检测模型评估器"""
    
    def __init__(self,
                 model: SimpleDetectionModel,
                 test_dataset: YOLOMFDataset,
                 batch_size: int = 4,
                 device: str = "mps" if torch.backends.mps.is_available() else "cpu",
                 iou_threshold: float = 0.5):
        """
        初始化评估器
        
        参数:
            model: 检测模型
            test_dataset: 测试数据集
            batch_size: 批大小
            device: 评估设备
            iou_threshold: IoU阈值，用于判定检测是否正确
        """
        self.model = model.to(device)
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.device = device
        self.iou_threshold = iou_threshold
        
        # 初始化logger
        logger.info("Initializing detection evaluator")
        
        # 数据加载器
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        logger.info(
            f"Evaluator initialized: batch_size={batch_size}, "
            f"iou_threshold={iou_threshold}, device={device}"
        )
    
    def calculate_iou(self, box1, box2):
        """计算两个边界框之间的IoU"""
        # 解包边界框坐标 (cx, cy, w, h)
        cx1, cy1, w1, h1 = box1
        cx2, cy2, w2, h2 = box2
        
        # 转换为 (x1, y1, x2, y2) 格式
        x1 = cx1 - w1/2
        y1 = cy1 - h1/2
        x2 = cx1 + w1/2
        y2 = cy1 + h1/2
        
        x3 = cx2 - w2/2
        y3 = cy2 - h2/2
        x4 = cx2 + w2/2
        y4 = cy2 + h2/2
        
        # 计算交集区域
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算并集区域
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        results = []
        
        logger.info("Starting evaluation...")
        
        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                
                # 模型预测
                predictions = self.model.predict(images)
                pred_boxes = predictions['boxes'].cpu().numpy()
                pred_scores = predictions['scores'].cpu().numpy()
                
                # 处理每个样本
                for i in range(len(images)):
                    gt_boxes = targets[i]['boxes'].numpy()
                    gt_labels = targets[i]['labels'].numpy()
                    
                    # 如果没有预测框，全部为假阳性
                    if len(pred_boxes[i]) == 0:
                        for _ in range(len(gt_boxes)):
                            results.append({
                                'true_positive': False,
                                'false_positive': True,
                                'score': 0.0
                            })
                        continue
                    
                    # 检查每个真实框是否有匹配的预测框
                    matched = [False] * len(gt_boxes)
                    
                    # 为每个预测框找到最佳匹配的真实框
                    for j in range(len(pred_boxes[i])):
                        max_iou = 0
                        best_match = -1
                        
                        for k in range(len(gt_boxes)):
                            iou = self.calculate_iou(pred_boxes[i][j], gt_boxes[k])
                            if iou > max_iou:
                                max_iou = iou
                                best_match = k
                        
                        # 如果IoU超过阈值且该真实框尚未匹配
                        if max_iou >= self.iou_threshold and not matched[best_match]:
                            matched[best_match] = True
                            results.append({
                                'true_positive': True,
                                'false_positive': False,
                                'iou': max_iou,
                                'score': pred_scores[i][j]
                            })
                        else:
                            results.append({
                                'true_positive': False,
                                'false_positive': True,
                                'iou': max_iou,
                                'score': pred_scores[i][j]
                            })
                    
                    # 处理未匹配的真实框（假阴性）
                    for k in range(len(gt_boxes)):
                        if not matched[k]:
                            results.append({
                                'true_positive': False,
                                'false_positive': False,  # 假阴性
                                'iou': 0.0,
                                'score': 0.0
                            })
        
        # 计算性能指标
        true_positives = sum(1 for r in results if r['true_positive'])
        false_positives = sum(1 for r in results if r['false_positive'])
        false_negatives = sum(1 for r in results if not r['true_positive'] and not r['false_positive'])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算平均IoU
        ious = [r['iou'] for r in results if r['iou'] > 0]
        avg_iou = sum(ious) / len(ious) if ious else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_iou': avg_iou,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
        
        logger.info(f"Evaluation completed: Precision={precision:.4f}, Recall={recall:.4f}, "
                         f"F1={f1_score:.4f}, Avg IoU={avg_iou:.4f}")
        
        return metrics

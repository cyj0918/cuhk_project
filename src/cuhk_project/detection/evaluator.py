import os
import json
import torch
import numpy as np
from torchvision.ops import nms
from tqdm import tqdm
from torch.utils.data import DataLoader
from .model import SimpleDetectionModel
from .dataset import YOLOMFDataset
from .visualizer import DetectionVisualizer

class DetectionEvaluator:
    def __init__(self, model, dataset, output_dir, device, conf_thresh=0.6, iou_thresh=0.6):
        self.model = model
        self.dataset = dataset
        self.loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=dataset.collate_fn,  # 使用统一的collate函数
            num_workers=0
        )
        self.output_dir = output_dir
        self.device=device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.visualizer = DetectionVisualizer(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)

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
        results = []
        metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        for images, targets in tqdm(self.loader, desc="Evaluating"):
            images = images.to(self.device)
            
            with torch.no_grad():
                # 确保predict返回列表格式
                batch_preds = self.model.predict(images, conf_thresh=self.conf_thresh)
            
            for i, target in enumerate(targets):
                # 获取真实框（已过滤padding）
                num_boxes = target['num_boxes'].item()
                true_boxes = target['boxes'][:num_boxes].cpu().numpy()
                
                # 获取预测结果
                pred = batch_preds[i]
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                
                # 可视化
                self.visualizer.visualize_sample(
                    images[i].cpu().permute(1, 2, 0).numpy(),
                    pred_boxes,
                    true_boxes,
                    f"sample_{len(results)}.png"
                )
                
                # 计算匹配
                matched_true, matched_pred = self._match_boxes(true_boxes, pred_boxes)
                
                # 更新指标
                metrics['true_positives'] += sum(matched_pred)
                metrics['false_positives'] += len(pred_boxes) - sum(matched_pred)
                metrics['false_negatives'] += len(true_boxes) - sum(matched_true)
                
                results.append({
                    'image_id': target['image_id'].item(),
                    'true_boxes': true_boxes,
                    'pred_boxes': pred_boxes,
                    'matched': matched_pred
                })
        
        # 计算最终指标
        return self._calculate_final_metrics(metrics, results)

    def _match_boxes(self, true_boxes, pred_boxes):
        """匹配真实框和预测框"""
        matched_true = [False] * len(true_boxes)
        matched_pred = [False] * len(pred_boxes)
        
        if len(true_boxes) == 0 or len(pred_boxes) == 0:
            return matched_true, matched_pred
        
        # 计算IoU矩阵
        iou_matrix = np.zeros((len(true_boxes), len(pred_boxes)))
        for i, true_box in enumerate(true_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_matrix[i, j] = self.calculate_iou(pred_box, true_box)
        
        # 确保1:1最佳匹配
        for i in range(len(true_boxes)):
            best_j = np.argmax(iou_matrix[i])
            if iou_matrix[i, best_j] > self.iou_thresh and not matched_pred[best_j]:
                matched_true[i] = True
                matched_pred[best_j] = True
        
        return matched_true, matched_pred
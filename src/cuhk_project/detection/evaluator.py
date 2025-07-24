import os
import json
import torch
import numpy as np
from tqdm import tqdm
from .model import SimpleDetectionModel
from .dataset import YOLOMFDataset
from .visualizer import DetectionVisualizer

class DetectionEvaluator:
    def __init__(self, model, dataset, output_dir, conf_thresh=0.2, iou_thresh=0.4):
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
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
        
        for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
            image, target = self.dataset[idx]
            image_tensor = image.unsqueeze(0)
            
            # 直接复用验证脚本的预测方式
            with torch.no_grad():
                pred = self.model.predict(image_tensor, conf_thresh=self.conf_thresh)
            
            true_boxes = target['boxes'].numpy()
            pred_boxes = pred['boxes'].cpu().numpy()
            
            # 保存可视化结果
            self.visualizer.visualize_sample(
                image, 
                pred_boxes, 
                true_boxes, 
                f"sample_{idx}.png"
            )
            
            # 简化的评估指标计算
            matched = [False] * len(true_boxes)
            for pred_box in pred_boxes:
                best_iou = 0
                for i, true_box in enumerate(true_boxes):
                    iou = self.calculate_iou(pred_box, true_box)
                    if iou > best_iou and iou > self.iou_thresh:
                        best_iou = iou
                        matched[i] = True
                
                if best_iou > 0:
                    metrics['true_positives'] += 1
                else:
                    metrics['false_positives'] += 1
            
            metrics['false_negatives'] += sum(1 for m in matched if not m)
        
        # 计算最终指标
        precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
        recall = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': metrics['true_positives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives']
        }
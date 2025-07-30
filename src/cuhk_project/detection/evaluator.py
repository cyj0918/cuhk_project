import numpy as np
import torch
from tqdm import tqdm
from .visualizer import DetectionVisualizer
from cuhk_project.utils.logger import logger

class DetectionEvaluator:
    def __init__(self, model, dataset, output_dir, conf_thresh=0.1, iou_thresh=0.1):
        self.model = model
        self.dataset = dataset
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.visualizer = DetectionVisualizer(output_dir)
        self.results = []
        self.metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }

    def evaluate(self):
        """执行模型评估并返回结果"""
        logger.info("Starting evaluation...")
        
        for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
            image, target = self.dataset[idx]
            image_tensor = image.unsqueeze(0)
            
            # 模型预测 - 修复返回格式处理
            with torch.no_grad():
            # 模型返回预测结果列表（每个元素对应一个batch）
                preds_list = self.model.predict(image_tensor, conf_thresh=self.conf_thresh)
                
                # 直接获取第一个（也是唯一一个）预测结果
                preds = preds_list[0]  # 直接获取预测结果字典
            
            # 获取真实框
            num_boxes = target['num_boxes'].item()
            true_boxes = target['boxes'][:num_boxes].cpu().numpy()
            
            # 可视化结果
            self._visualize_sample(image, preds, true_boxes, idx)
            
            # 计算指标
            self._calculate_metrics(preds, true_boxes, idx)
        
        return self._finalize_metrics()

    def _visualize_sample(self, image, preds, true_boxes, idx):
        """可视化样本结果"""
        # 处理空预测情况
        if preds['boxes'].numel() == 0:
            pred_boxes = np.empty((0, 4))
        else:
            pred_boxes = preds['boxes'].cpu().numpy()
        
        self.visualizer.visualize_sample(
            image.permute(1, 2, 0).numpy(),
            pred_boxes,
            true_boxes,
            f"eval_sample_{idx}.png"
        )

    def _calculate_metrics(self, preds, true_boxes, idx):
        """计算评估指标"""
        if preds['boxes'].numel() == 0:
            pred_boxes = np.empty((0, 4))
            pred_scores = np.empty(0)
        else:
            pred_boxes = preds['boxes'].cpu().numpy()
            pred_scores = preds['scores'].cpu().numpy()
        
        # 匹配预测与真实框
        matched_true, matched_pred = self._match_boxes(true_boxes, pred_boxes)
        
        # 更新指标
        self.metrics['true_positives'] += sum(matched_pred)
        self.metrics['false_positives'] += len(pred_boxes) - sum(matched_pred)
        self.metrics['false_negatives'] += len(true_boxes) - sum(matched_true)
        
        # 保存详细结果
        self.results.append({
            'image_id': idx,
            'true_boxes': true_boxes,
            'pred_boxes': pred_boxes,
            'pred_scores': pred_scores,
            'matched': matched_pred
        })

    def _match_boxes(self, true_boxes, pred_boxes):
        """匹配真实框和预测框"""
        if true_boxes.ndim == 1:
            true_boxes = true_boxes.reshape(1, -1)
        if pred_boxes.ndim == 1:
            pred_boxes = pred_boxes.reshape(1, -1)
        
        matched_true = [False] * len(true_boxes)
        matched_pred = [False] * len(pred_boxes)
        
        if not true_boxes.size or not pred_boxes.size:
            return matched_true, matched_pred
        
        # 计算IoU矩阵
        iou_matrix = np.zeros((len(true_boxes), len(pred_boxes)))
        for i, true_box in enumerate(true_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_matrix[i, j] = self._calculate_iou(pred_box, true_box)
        
        # 1:1最佳匹配
        for i in range(len(true_boxes)):
            best_j = np.argmax(iou_matrix[i])
            if iou_matrix[i, best_j] > self.iou_thresh and not matched_pred[best_j]:
                matched_true[i] = True
                matched_pred[best_j] = True
        
        return matched_true, matched_pred

    @staticmethod
    def _calculate_iou(box1, box2):
        """计算两个边界框的IoU"""
        # 转换为[x1, y1, x2, y2]格式
        box1_x1 = box1[0] - box1[2] / 2
        box1_y1 = box1[1] - box1[3] / 2
        box1_x2 = box1[0] + box1[2] / 2
        box1_y2 = box1[1] + box1[3] / 2
        
        box2_x1 = box2[0] - box2[2] / 2
        box2_y1 = box2[1] - box2[3] / 2
        box2_x2 = box2[0] + box2[2] / 2
        box2_y2 = box2[1] + box2[3] / 2
        
        # 计算交集
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # 计算并集
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / max(union_area, 1e-6)

    def _finalize_metrics(self):
        """计算最终评估指标"""
        tp = self.metrics['true_positives']
        fp = self.metrics['false_positives']
        fn = self.metrics['false_negatives']
        
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'sample_results': self.results
        }
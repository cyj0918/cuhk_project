import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from .visualizer import DetectionVisualizer
from cuhk_project.utils.logger import logger

class DetectionEvaluator:
    """目標檢測模型評估器 - 適配新數據集格式和訓練架構"""
    
    def __init__(self, 
                 model, 
                 dataset, 
                 output_dir: str,
                 conf_thresh: float = 0.5, 
                 iou_thresh: float = 0.5,
                 use_torchmetrics: bool = True,
                 device: str = "cpu"):
        """
        初始化評估器
        
        參數:
            model: 訓練好的檢測模型
            dataset: 評估數據集
            output_dir: 結果輸出目錄
            conf_thresh: 置信度閾值
            iou_thresh: IoU閾值用於匹配
            use_torchmetrics: 是否使用torchmetrics計算mAP
            device: 設備
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.use_torchmetrics = use_torchmetrics
        
        # 初始化可視化器
        self.visualizer = DetectionVisualizer(str(self.output_dir))
        
        # 初始化評估指標
        self.reset_metrics()
        
        # 如果使用torchmetrics，初始化mAP計算器
        if self.use_torchmetrics:
            try:
                self.map_metric = MeanAveragePrecision(
                    box_format="cxcywh",  # 匹配數據集格式
                    iou_type="bbox",
                    iou_thresholds=[0.5, 0.75] + [0.5 + 0.05 * i for i in range(10)],  # [0.5:0.95:0.05]
                    class_metrics=True
                )
                self.map_metric.to(device)
                logger.info("Using TorchMetrics for mAP calculation")
            except ImportError:
                logger.warning("TorchMetrics not available, using custom mAP calculation")
                self.use_torchmetrics = False
        
        logger.info(f"Evaluator initialized: conf_thresh={conf_thresh}, iou_thresh={iou_thresh}")
        logger.info(f"Output directory: {self.output_dir}")

    def reset_metrics(self):
        """重置評估指標"""
        self.results = []
        self.metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_gt_boxes': 0,
            'total_pred_boxes': 0,
            'total_images': 0
        }

    def evaluate(self) -> Dict:
        """執行模型評估並返回結果"""
        logger.info("Starting model evaluation...")
        logger.info(f"Evaluating on {len(self.dataset)} samples")
        
        self.model.eval()
        self.reset_metrics()
        
        # 收集所有預測和真實標籤用於torchmetrics
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
                try:
                    # 獲取數據
                    image, target = self.dataset[idx]
                    
                    # 確保圖像是CHW Tensor格式
                    if image.dim() == 3:
                        image_tensor = image.unsqueeze(0).to(self.device, dtype=torch.float32)
                    else:
                        image_tensor = image.to(self.device, dtype=torch.float32)
                    
                    # 模型預測
                    preds_list = self.model.predict(
                        image_tensor, 
                        conf_thresh=self.conf_thresh,
                        iou_thresh=self.iou_thresh
                    )
                    
                    # 獲取第一個（也是唯一一個）預測結果
                    preds = preds_list[0] if preds_list else {'boxes': torch.empty(0, 4), 'scores': torch.empty(0)}
                    
                    # 處理真實標註
                    num_boxes = target['num_boxes'].item()
                    true_boxes = target['boxes'][:num_boxes].cpu()  # [N, 4] cxcywh format
                    true_labels = target['labels'][:num_boxes].cpu() if 'labels' in target else torch.ones(num_boxes, dtype=torch.long)
                    
                    # 更新統計
                    self.metrics['total_images'] += 1
                    self.metrics['total_gt_boxes'] += num_boxes
                    self.metrics['total_pred_boxes'] += len(preds['boxes'])
                    
                    # 可視化結果（每10個樣本可視化一次）
                    if idx % 10 == 0 or idx < 5:
                        self._visualize_sample(image, preds, true_boxes.numpy(), idx)
                    
                    # 計算基本指標
                    self._calculate_basic_metrics(preds, true_boxes.numpy(), idx)
                    
                    # 為torchmetrics準備數據
                    if self.use_torchmetrics:
                        pred_dict, target_dict = self._prepare_torchmetrics_data(
                            preds, true_boxes, true_labels, idx
                        )
                        all_predictions.append(pred_dict)
                        all_targets.append(target_dict)
                    
                except Exception as e:
                    logger.error(f"Error processing sample {idx}: {str(e)}")
                    continue
        
        # 計算最終指標
        final_results = self._finalize_metrics()
        
        # 使用torchmetrics計算mAP
        if self.use_torchmetrics and all_predictions:
            try:
                self.map_metric.update(all_predictions, all_targets)
                map_results = self.map_metric.compute()
                
                # 添加mAP結果到最終結果
                final_results.update({
                    'mAP': map_results['map'].item(),
                    'mAP_50': map_results['map_50'].item(),
                    'mAP_75': map_results['map_75'].item(),
                    'mAP_small': map_results.get('map_small', torch.tensor(-1)).item(),
                    'mAP_medium': map_results.get('map_medium', torch.tensor(-1)).item(),
                    'mAP_large': map_results.get('map_large', torch.tensor(-1)).item(),
                })
                
                if 'map_per_class' in map_results and map_results['map_per_class'].numel() > 0:
                    final_results['mAP_per_class'] = map_results['map_per_class'].cpu().numpy().tolist()
                
                logger.info(f"TorchMetrics mAP: {final_results['mAP']:.4f}")
                logger.info(f"TorchMetrics mAP@0.5: {final_results['mAP_50']:.4f}")
                logger.info(f"TorchMetrics mAP@0.75: {final_results['mAP_75']:.4f}")
                
            except Exception as e:
                logger.error(f"Error computing torchmetrics mAP: {str(e)}")
        
        # 保存詳細結果
        self._save_results(final_results)
        
        return final_results

    def _prepare_torchmetrics_data(self, preds, true_boxes, true_labels, image_id):
        """為torchmetrics準備數據格式"""
        # 預測數據
        if preds['boxes'].numel() > 0:
            pred_boxes = preds['boxes'].cpu()
            pred_scores = preds['scores'].cpu()
            # 假設所有預測都是類別1（非背景）
            pred_labels = torch.ones(len(pred_boxes), dtype=torch.long)
        else:
            pred_boxes = torch.empty(0, 4, dtype=torch.float32)
            pred_scores = torch.empty(0, dtype=torch.float32)
            pred_labels = torch.empty(0, dtype=torch.long)
        
        pred_dict = {
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': pred_labels
        }
        
        # 真實標籤數據
        target_dict = {
            'boxes': true_boxes,
            'labels': true_labels
        }
        
        return pred_dict, target_dict

    def _visualize_sample(self, image: torch.Tensor, preds: Dict, true_boxes: np.ndarray, idx: int):
        """可視化樣本結果"""
        try:
            # 轉換圖像格式用於可視化
            if image.dim() == 3 and image.shape[0] == 3:  # CHW
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image.cpu().numpy()
            
            # 確保圖像值在[0,1]範圍內
            if image_np.max() <= 1.0:
                image_np = np.clip(image_np, 0, 1)
            else:
                image_np = np.clip(image_np / 255.0, 0, 1)
            
            # 處理預測框
            if preds['boxes'].numel() > 0:
                pred_boxes = preds['boxes'].cpu().numpy()
                pred_scores = preds['scores'].cpu().numpy()
            else:
                pred_boxes = np.empty((0, 4))
                pred_scores = np.empty(0)
            
            # 可視化
            self.visualizer.visualize_sample(
                image_np,
                pred_boxes,
                true_boxes,
                f"eval_sample_{idx:04d}.png",
                pred_scores=pred_scores,
                conf_thresh=self.conf_thresh
            )
            
        except Exception as e:
            logger.warning(f"Failed to visualize sample {idx}: {str(e)}")

    def _calculate_basic_metrics(self, preds: Dict, true_boxes: np.ndarray, idx: int):
        """計算基本評估指標"""
        # 處理預測結果
        if preds['boxes'].numel() == 0:
            pred_boxes = np.empty((0, 4))
            pred_scores = np.empty(0)
        else:
            pred_boxes = preds['boxes'].cpu().numpy()
            pred_scores = preds['scores'].cpu().numpy()
        
        # 匹配預測與真實框
        matched_true, matched_pred = self._match_boxes(true_boxes, pred_boxes)
        
        # 更新指標
        tp = sum(matched_pred)
        fp = len(pred_boxes) - tp
        fn = len(true_boxes) - sum(matched_true)
        
        self.metrics['true_positives'] += tp
        self.metrics['false_positives'] += fp
        self.metrics['false_negatives'] += fn
        
        # 保存詳細結果
        self.results.append({
            'image_id': idx,
            'num_gt_boxes': len(true_boxes),
            'num_pred_boxes': len(pred_boxes),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'matched_pred': matched_pred,
            'pred_scores': pred_scores.tolist() if len(pred_scores) > 0 else []
        })

    def _match_boxes(self, true_boxes: np.ndarray, pred_boxes: np.ndarray) -> Tuple[List[bool], List[bool]]:
        """匹配真實框和預測框"""
        # 處理空數組情況
        if len(true_boxes) == 0:
            return [], [False] * len(pred_boxes)
        if len(pred_boxes) == 0:
            return [False] * len(true_boxes), []
        
        # 確保維度正確
        if true_boxes.ndim == 1:
            true_boxes = true_boxes.reshape(1, -1)
        if pred_boxes.ndim == 1:
            pred_boxes = pred_boxes.reshape(1, -1)
        
        matched_true = [False] * len(true_boxes)
        matched_pred = [False] * len(pred_boxes)
        
        # 計算IoU矩陣
        iou_matrix = np.zeros((len(true_boxes), len(pred_boxes)))
        for i, true_box in enumerate(true_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_matrix[i, j] = self._calculate_iou(pred_box, true_box)
        
        # 貪心匹配：為每個真實框找到最佳預測框
        for i in range(len(true_boxes)):
            if len(pred_boxes) == 0:
                break
                
            best_j = np.argmax(iou_matrix[i])
            best_iou = iou_matrix[i, best_j]
            
            if best_iou >= self.iou_thresh and not matched_pred[best_j]:
                matched_true[i] = True
                matched_pred[best_j] = True
                # 將已匹配的預測框從後續匹配中排除
                iou_matrix[:, best_j] = 0
        
        return matched_true, matched_pred

    @staticmethod
    def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """計算兩個邊界框的IoU（輸入格式：cxcywh）"""
        try:
            # 轉換為xyxy格式
            def cxcywh_to_xyxy(box):
                cx, cy, w, h = box
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                return x1, y1, x2, y2
            
            x1_1, y1_1, x2_1, y2_1 = cxcywh_to_xyxy(box1)
            x1_2, y1_2, x2_2, y2_2 = cxcywh_to_xyxy(box2)
            
            # 計算交集
            inter_x1 = max(x1_1, x1_2)
            inter_y1 = max(y1_1, y1_2)
            inter_x2 = min(x2_1, x2_2)
            inter_y2 = min(y2_1, y2_2)
            
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            
            # 計算並集
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - inter_area
            
            return inter_area / max(union_area, 1e-6)
            
        except Exception as e:
            logger.warning(f"Error calculating IoU: {str(e)}")
            return 0.0

    def _finalize_metrics(self) -> Dict:
        """計算最終評估指標"""
        tp = self.metrics['true_positives']
        fp = self.metrics['false_positives']
        fn = self.metrics['false_negatives']
        
        # 計算基本指標
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        
        # 計算統計信息
        avg_gt_boxes_per_image = self.metrics['total_gt_boxes'] / max(1, self.metrics['total_images'])
        avg_pred_boxes_per_image = self.metrics['total_pred_boxes'] / max(1, self.metrics['total_images'])
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'total_gt_boxes': self.metrics['total_gt_boxes'],
            'total_pred_boxes': self.metrics['total_pred_boxes'],
            'total_images': self.metrics['total_images'],
            'avg_gt_boxes_per_image': avg_gt_boxes_per_image,
            'avg_pred_boxes_per_image': avg_pred_boxes_per_image,
            'conf_thresh': self.conf_thresh,
            'iou_thresh': self.iou_thresh,
            'sample_results': self.results
        }
        
        # 日誌輸出
        logger.info("="*50)
        logger.info("評估結果摘要:")
        logger.info(f"處理圖像數量: {results['total_images']}")
        logger.info(f"真實框總數: {results['total_gt_boxes']}")
        logger.info(f"預測框總數: {results['total_pred_boxes']}")
        logger.info(f"準確率 (Precision): {results['precision']:.4f}")
        logger.info(f"召回率 (Recall): {results['recall']:.4f}")
        logger.info(f"F1分數: {results['f1_score']:.4f}")
        logger.info(f"真正例 (TP): {results['true_positives']}")
        logger.info(f"假正例 (FP): {results['false_positives']}")
        logger.info(f"假負例 (FN): {results['false_negatives']}")
        logger.info("="*50)
        
        return results

    def _save_results(self, results: Dict):
        """保存評估結果到文件"""
        try:
            import json
            
            # 創建結果摘要
            summary = {k: v for k, v in results.items() if k != 'sample_results'}
            
            # 保存摘要結果
            summary_path = self.output_dir / "evaluation_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            # 保存詳細結果
            detailed_path = self.output_dir / "evaluation_detailed.json" 
            with open(detailed_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"評估結果已保存到: {summary_path}")
            logger.info(f"詳細結果已保存到: {detailed_path}")
            
        except Exception as e:
            logger.error(f"保存結果時出錯: {str(e)}")

    def evaluate_at_multiple_thresholds(self, 
                                       conf_thresholds: List[float] = None,
                                       iou_thresholds: List[float] = None) -> Dict:
        """在多個閾值下評估模型性能"""
        if conf_thresholds is None:
            conf_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        if iou_thresholds is None:
            iou_thresholds = [0.3, 0.5, 0.7]
        
        logger.info(f"多閾值評估: conf_thresholds={conf_thresholds}, iou_thresholds={iou_thresholds}")
        
        all_results = {}
        
        for conf_thresh in conf_thresholds:
            for iou_thresh in iou_thresholds:
                logger.info(f"評估閾值組合: conf={conf_thresh}, iou={iou_thresh}")
                
                # 更新閾值
                self.conf_thresh = conf_thresh
                self.iou_thresh = iou_thresh
                
                # 執行評估
                results = self.evaluate()
                
                # 保存結果
                key = f"conf_{conf_thresh}_iou_{iou_thresh}"
                all_results[key] = {
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score'],
                    'mAP': results.get('mAP', -1),
                    'mAP_50': results.get('mAP_50', -1)
                }
        
        # 保存多閾值結果
        multi_thresh_path = self.output_dir / "multi_threshold_evaluation.json"
        with open(multi_thresh_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"多閾值評估結果已保存到: {multi_thresh_path}")
        
        return all_results
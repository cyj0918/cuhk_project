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
                 iou_threshold: float = 0.5,
                 conf_threshold: float = 0.5):
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
        self.conf_threshold = conf_threshold
        
        # 初始化logger
        logger.info("Initializing detection evaluator")
        
        # 数据加载器配置
        num_workers = 0 if device == "mps" else 2
        if device == "mps":
            logger.warning("MPS device detected - disabling multiprocessing (num_workers=0)")
            
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=device != "mps"  # 禁用MPS的pin_memory
        )
        logger.info(
                f"Evaluator initialized: batch_size={batch_size}, "
                f"iou_threshold={iou_threshold}, device={device}"
            )    
    
    def _collate_fn(self, batch):
        """新的批处理函数，保持原始数据结构"""
        # batch是[(image1, target1), (image2, target2), ...]的列表
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets  # 返回(images_list, targets_list)元组
    
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
    
    def __getstate__(self):
        """支持pickle序列化"""
        state = self.__dict__.copy()
        # 移除不可pickle的属性
        if '_collate_fn' in state:
            del state['_collate_fn']
        return state
        
    def __setstate__(self, state):
        """支持pickle反序列化"""
        self.__dict__.update(state)
        # 重新初始化不可pickle的属性
        if not hasattr(self, '_collate_fn'):
            self._collate_fn = lambda x: tuple(zip(*x))
            
    def _validate_boxes(self, boxes) -> bool:
        """验证边界框坐标是否有效"""
        # 统一转换为numpy数组处理
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        # 检查空数组
        if boxes.size == 0:
            return False
            
        for box in boxes:
            # 确保转换为Python float类型
            cx, cy, w, h = map(float, box[:4])
            
            # 检查坐标范围
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                logger.warning(f"Invalid center coordinates: ({cx:.4f}, {cy:.4f})")
                return False
                
            # 检查宽高
            if w <= 0 or h <= 0:
                logger.warning(f"Invalid box dimensions: width={w:.4f}, height={h:.4f}")
                return False
                
            # 检查边界
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                logger.warning(f"Invalid box bounds: ({x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f})")
                return False
                
        return True

    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        results = []
        self.conf_threshold = 0.3
        
        # MPS设备警告
        if str(self.device) == "mps":
            logger.warning("Running evaluation on MPS - performance may be limited")
        
        # 添加pickle支持
        if not hasattr(self, '_collate_fn'):
            self._collate_fn = lambda x: tuple(zip(*x))
            
        logger.info("Starting evaluation with confidence threshold=0.3...")

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                    # 安全解包批数据
                    try:
                        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                            raise ValueError(f"Unexpected batch format: {type(batch)}")
                            
                        # 首先解包batch
                        images_list, targets = batch
                        
                        # 验证解包结果
                        if not isinstance(images_list, list) or not all(isinstance(img, torch.Tensor) for img in images_list):
                            raise ValueError("Images must be list of tensors")
                            
                        if not isinstance(targets, list) or not all(isinstance(t, dict) for t in targets):
                            raise ValueError("Targets must be list of dicts")
                            
                        # 转换images为tensor
                        images = torch.stack(images_list).to(self.device)

                    except Exception as e:
                        logger.error(f"Batch unpacking failed: {str(e)}")
                        logger.error(f"Batch type: {type(batch)}")
                        raise

                    # 转换并验证图像
                    try:
                        # images已经是stack后的tensor
                        if not isinstance(images, torch.Tensor):
                            raise ValueError(f"Expected tensor, got {type(images)}")
                        if len(images.shape) != 4:
                            raise ValueError(f"Invalid image tensor shape: {images.shape}")
                        
                        # 确保设备正确
                        images = images.to(self.device)
                        
                       # 修改targets设备转换方式
                        targets = [{
                            k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in t.items()
                        } for t in targets]

                    except Exception as e:
                        logger.error(f"Image processing failed: {str(e)}")
                        raise

                    # 获取预测
                    try:
                        predictions = self.model.predict(images)
                        if not isinstance(predictions, dict):
                            raise ValueError("Model should return dict with 'boxes' and 'scores'")
                        
                        pred_boxes = predictions['boxes'].cpu().numpy()
                        pred_scores = predictions['scores'].cpu().numpy()
                    except Exception as e:
                        logger.error(f"Prediction failed: {str(e)}")
                        logger.error(f"Input shape: {images.shape}")
                        raise

                    # 处理每个样本
                    for i in range(len(images)):
                        try:
                            # 获取实际框(忽略填充)
                            num_boxes = targets[i]['num_boxes'].item()
                            gt_boxes = targets[i]['boxes'][:num_boxes].cpu().numpy()
                            gt_labels = targets[i]['labels'][:num_boxes].cpu().numpy()
                            
                            logger.debug(f"Sample {i}: {num_boxes} real boxes (of {len(targets[i]['boxes'])})")

                            # 跳过无效GT框
                            if num_boxes == 0:
                                logger.warning(f"Sample {i}: No ground truth boxes - skipping")
                                continue
                            
                            # 验证GT框坐标
                            if not self._validate_boxes(gt_boxes):
                                logger.warning(f"Sample {i}: Invalid GT boxes - skipping")
                                continue
                                
                            # 过滤预测
                            valid_mask = (pred_scores[i].squeeze() >= self.conf_threshold)  # 确保1D布尔数组
                            filtered_boxes = pred_boxes[i][valid_mask]
                            filtered_scores = pred_scores[i][valid_mask]
                            
                            # 验证预测框
                            if not self._validate_boxes(filtered_boxes):
                                logger.warning(f"Sample {i}: Invalid predicted boxes - skipping")
                                continue
                            
                            logger.debug(f"Sample {i}: {len(filtered_boxes)} detections after filtering")

                            # 初始化匹配状态
                            matched_gt = [False] * len(gt_boxes)
                            matched_pred = [False] * len(filtered_boxes)
                            
                            # 匹配逻辑
                            for gt_idx, gt_box in enumerate(gt_boxes):
                                best_iou = 0
                                best_pred_idx = -1
                                
                                for pred_idx, pred_box in enumerate(filtered_boxes):
                                    if matched_pred[pred_idx]:
                                        continue
                                        
                                    try:
                                        iou = self.calculate_iou(pred_box, gt_box)
                                        if iou > best_iou:
                                            best_iou = iou
                                            best_pred_idx = pred_idx
                                    except Exception as e:
                                        logger.warning(f"Sample {i}: IoU calc failed - {str(e)}")
                                        continue
                                
                                # 记录匹配结果
                                if best_iou >= self.iou_threshold and best_pred_idx != -1:
                                    matched_gt[gt_idx] = True
                                    matched_pred[best_pred_idx] = True
                                    results.append({
                                        'true_positive': True,
                                        'false_positive': False,
                                        'iou': best_iou,
                                        'score': filtered_scores[best_pred_idx],
                                        'gt_box': gt_box,
                                        'pred_box': filtered_boxes[best_pred_idx]
                                    })
                            
                            # 记录未匹配的预测框(假阳性)
                            for pred_idx in range(len(filtered_boxes)):
                                if not matched_pred[pred_idx]:
                                    results.append({
                                        'true_positive': False,
                                        'false_positive': True,
                                        'iou': 0.0,
                                        'score': filtered_scores[pred_idx],
                                        'pred_box': filtered_boxes[pred_idx]
                                    })
                            
                            # 记录未匹配的真实框(假阴性)
                            for gt_idx in range(len(gt_boxes)):
                                if not matched_gt[gt_idx]:
                                    results.append({
                                        'true_positive': False,
                                        'false_positive': False,
                                        'iou': 0.0,
                                        'score': 0.0,
                                        'gt_box': gt_boxes[gt_idx]
                                    })
                        
                        except Exception as e:
                            logger.error(f"Sample {i} processing failed: {str(e)}")
                            continue

        except Exception as e:
            logger.error(f"Evaluation failed completely: {str(e)}")
            raise
        
        # 计算性能指标
        try:
            true_positives = sum(1 for r in results if r['true_positive'])
            false_positives = sum(1 for r in results if r['false_positive'])
            false_negatives = sum(1 for r in results if not r['true_positive'] and not r['false_positive'])
            
            # 样本量警告
            if len(results) < 10:
                logger.warning(f"Low sample size ({len(results)}) may affect metric reliability")
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # 计算平均IoU(仅使用匹配的预测)
            matched_ious = [r['iou'] for r in results if r['true_positive']]
            avg_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'avg_iou': avg_iou,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'sample_size': len(results),
                'evaluation_status': 'completed'
            }
            
            logger.info(f"Evaluation completed: Precision={precision:.4f}, Recall={recall:.4f}, "
                        f"F1={f1_score:.4f}, Avg IoU={avg_iou:.4f}")
            logger.info(f"Total samples evaluated: {len(results)}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Metric calculation failed: {str(e)}")
            return {
                'evaluation_status': 'failed',
                'error': str(e)
            }

import os
import math
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from cuhk_project.utils.logger import logger
from .model import SimpleDetectionModel
from .dataset import YOLOMFDataset
import torch.nn.functional as F

class DetectionTrainer:
    """目標檢測模型訓練器（YOLO網格版本）- 適配新數據集格式"""
    
    @staticmethod
    def compute_anchors(dataset, num_anchors=3):
        """靜態方法：基於數據集統計計算最優錨框"""
        all_boxes = []
        
        # 收集所有邊界框尺寸
        for i in range(min(len(dataset), 1000)):  # 限制樣本數量避免過長計算
            try:
                _, target = dataset[i]
                boxes = target['boxes']
                if len(boxes) > 0:
                    for box in boxes:
                        w, h = box[2].item(), box[3].item()  # 寬度和高度
                        if w > 0 and h > 0:  # 確保有效尺寸
                            all_boxes.append([w, h])
            except Exception as e:
                logger.warning(f"Error processing sample {i} for anchor computation: {str(e)}")
                continue
        
        if not all_boxes:
            # 返回默認錨框尺寸
            logger.warning("No valid boxes found, using default anchors")
            return torch.tensor([
                [0.05, 0.15],   # 小物體
                [0.08, 0.25],   # 中等物體  
                [0.12, 0.35]    # 大物體
            ], dtype=torch.float32)
        
        # 使用K-means聚類確定錨框尺寸
        try:
            from sklearn.cluster import KMeans
            all_boxes = np.array(all_boxes)
            
            kmeans = KMeans(n_clusters=num_anchors, random_state=0, n_init=10)
            kmeans.fit(all_boxes)
            
            # 按面積排序錨框
            anchors = kmeans.cluster_centers_
            anchors = sorted(anchors, key=lambda x: x[0] * x[1])
            anchors_tensor = torch.tensor(anchors, dtype=torch.float32)
            
            logger.info(f"Computed {num_anchors} anchors from {len(all_boxes)} boxes: {anchors_tensor.tolist()}")
            return anchors_tensor
            
        except ImportError:
            logger.warning("sklearn not available, using default anchors")
            return torch.tensor([
                [0.05, 0.15], [0.08, 0.25], [0.12, 0.35]
            ], dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error computing anchors: {str(e)}, using defaults")
            return torch.tensor([
                [0.05, 0.15], [0.08, 0.25], [0.12, 0.35]
            ], dtype=torch.float32)

    def __init__(self, 
                 model: SimpleDetectionModel, 
                 train_dataset: YOLOMFDataset,
                 val_dataset: YOLOMFDataset,
                 grid_size: tuple = (6, 32),      # 與數據集一致的網格尺寸
                 num_anchors: int = 3,            
                 batch_size: int = 4,
                 learning_rate: float = 0.001,
                 num_epochs: int = 10,
                 device: str = "cpu"):            
        """
        初始化訓練器
        
        參數:
            model: 檢測模型
            train_dataset: 訓練數據集  
            val_dataset: 驗證數據集
            grid_size: 網格尺寸 (高度, 寬度)
            num_anchors: 錨框數量
            batch_size: 批次大小
            learning_rate: 學習率
            num_epochs: 訓練輪數
            device: 訓練設備
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        
        # 驗證網格尺寸一致性
        dataset_grid = train_dataset.grid_height, train_dataset.grid_width
        if dataset_grid != grid_size:
            logger.warning(f"Grid size mismatch: trainer={grid_size}, dataset={dataset_grid}")
            self.grid_size = dataset_grid  # 使用數據集的網格尺寸
        
        # 初始化錨框（從模型獲取或重新計算）
        if hasattr(model, 'anchor_boxes') and model.anchor_boxes is not None:
            self.anchor_boxes = model.anchor_boxes
            logger.info(f"Using model anchors: {self.anchor_boxes.tolist()}")
        else:
            self.anchor_boxes = self.compute_anchors(train_dataset, num_anchors)
            logger.info(f"Computed new anchors: {self.anchor_boxes.tolist()}")
            
        # 創建數據加載器 - 使用數據集的collate_fn
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=YOLOMFDataset.collate_fn,
            num_workers=0,  # 避免多進程問題
            pin_memory=False
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=YOLOMFDataset.collate_fn,
            num_workers=0,
            pin_memory=False
        )
        
        # 優化器和損失函數
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.bbox_loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        self.obj_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        
        logger.info(
            f"Trainer initialized: grid_size={self.grid_size}, anchors={num_anchors}, "
            f"batch_size={batch_size}, lr={learning_rate}, epochs={num_epochs}, device={device}"
        )
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    def _build_targets(self, targets):
        """構建YOLO格式的網格目標 - 適配新數據格式"""
        B = len(targets)
        H, W = self.grid_size
        
        # 初始化目標張量 [B, anchors, 5, H, W]
        targets_tensor = torch.zeros(B, self.num_anchors, 5, H, W, dtype=torch.float32)
        
        for b, target in enumerate(targets):
            try:
                num_boxes = target['num_boxes'].item()
                if num_boxes == 0:
                    continue
                    
                boxes = target['boxes'][:num_boxes]  # 只取真實boxes
                labels = target['labels'][:num_boxes]  # 對應的標籤
                
                # 處理每個邊界框
                for box, label in zip(boxes, labels):
                    cx, cy, w, h = box.tolist()
                    
                    # 邊界檢查
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        logger.warning(f"Invalid box coordinates: cx={cx}, cy={cy}, w={w}, h={h}")
                        continue
                    
                    # 計算網格位置
                    grid_x = min(W-1, max(0, int(cx * W)))
                    grid_y = min(H-1, max(0, int(cy * H)))
                    
                    # 計算在網格內的偏移量 [0,1]
                    offset_x = cx * W - grid_x
                    offset_y = cy * H - grid_y
                    
                    # 找到最匹配的錨框（基於IoU）
                    best_iou = 0
                    best_anchor = 0
                    
                    for anchor_idx, anchor in enumerate(self.anchor_boxes):
                        anchor_w, anchor_h = anchor[0].item(), anchor[1].item()
                        
                        # 計算IoU（假設中心重合）
                        box_area = w * h
                        anchor_area = anchor_w * anchor_h
                        inter_w = min(w, anchor_w)
                        inter_h = min(h, anchor_h)
                        inter_area = inter_w * inter_h
                        
                        iou = inter_area / (box_area + anchor_area - inter_area + 1e-6)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_anchor = anchor_idx
                    
                    # 設置目標值
                    anchor_w = self.anchor_boxes[best_anchor][0].item()
                    anchor_h = self.anchor_boxes[best_anchor][1].item()
                    
                    # 偏移量 [0,1]
                    targets_tensor[b, best_anchor, 0, grid_y, grid_x] = offset_x
                    targets_tensor[b, best_anchor, 1, grid_y, grid_x] = offset_y
                    
                    # 尺度（對數空間）
                    targets_tensor[b, best_anchor, 2, grid_y, grid_x] = math.log(max(w / anchor_w, 1e-6))
                    targets_tensor[b, best_anchor, 3, grid_y, grid_x] = math.log(max(h / anchor_h, 1e-6))
                    
                    # 置信度
                    targets_tensor[b, best_anchor, 4, grid_y, grid_x] = 1.0
                    
            except Exception as e:
                logger.error(f"Error building targets for batch {b}: {str(e)}")
                continue
                
        return targets_tensor

    def _compute_loss(self, bbox_pred, obj_pred, targets_tensor):
        """計算訓練損失"""
        # 提取目標值
        target_offsets = targets_tensor[..., :2, :, :]    # [B, anchors, 2, H, W]
        target_scales = targets_tensor[..., 2:4, :, :]    # [B, anchors, 2, H, W]  
        target_obj = targets_tensor[..., 4, :, :]         # [B, anchors, H, W]
        
        # 提取預測值
        pred_offsets = bbox_pred[..., :2, :, :]           # [B, anchors, 2, H, W]
        pred_scales = bbox_pred[..., 2:4, :, :]           # [B, anchors, 2, H, W]
        
        # 創建目標存在的mask
        obj_mask = (target_obj > 0.5)  # [B, anchors, H, W]
        num_pos = obj_mask.sum().float().clamp(min=1.0)  # 防止除零
        
        # 1. 偏移量損失（僅在有目標的位置計算）
        if obj_mask.any():
            # 展開mask以匹配offset預測的形狀 [B, anchors, 2, H, W]
            offset_mask = obj_mask.unsqueeze(2).expand_as(pred_offsets)
            
            offset_loss = self.bbox_loss_fn(
                pred_offsets[offset_mask],
                target_offsets[offset_mask]
            )
        else:
            offset_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 2. 尺度損失（僅在有目標的位置計算）
        if obj_mask.any():
            # 展開mask以匹配scale預測的形狀
            scale_mask = obj_mask.unsqueeze(2).expand_as(pred_scales)
            
            scale_loss = self.bbox_loss_fn(
                pred_scales[scale_mask],
                target_scales[scale_mask]
            )
        else:
            scale_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 3. 目標置信度損失（所有位置都計算）
        obj_loss_val = self.obj_loss_fn(obj_pred, target_obj)
        
        # 加權總損失
        total_loss = 5.0 * offset_loss + 5.0 * scale_loss + 1.0 * obj_loss_val
        
        return {
            'total_loss': total_loss,
            'offset_loss': offset_loss,
            'scale_loss': scale_loss,
            'obj_loss': obj_loss_val,
            'num_pos': num_pos
        }

    def train_epoch(self, epoch: int) -> dict:
        """訓練一個epoch"""
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'offset_loss': 0.0, 
            'scale_loss': 0.0,
            'obj_loss': 0.0
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            try:
                # 數據移到設備
                images = images.to(self.device, dtype=torch.float32)
                
                # 驗證圖像格式 - 應該是 [B, C, H, W]
                if images.dim() != 4 or images.size(1) != 3:
                    logger.error(f"Invalid image tensor shape: {images.shape}")
                    continue
                
                # 構建YOLO目標
                yolo_targets = self._build_targets(targets).to(self.device)
                
                # 前向傳播
                self.optimizer.zero_grad()
                bbox_pred, obj_pred = self.model(images)
                
                # 計算損失
                losses = self._compute_loss(bbox_pred, obj_pred, yolo_targets)
                
                # 反向傳播
                losses['total_loss'].backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 累積損失
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key].item()
                
                # 更新進度條
                progress_bar.set_postfix({
                    'loss': f'{losses["total_loss"].item():.4f}',
                    'off': f'{losses["offset_loss"].item():.4f}',
                    'scl': f'{losses["scale_loss"].item():.4f}',
                    'obj': f'{losses["obj_loss"].item():.4f}',
                    'pos': f'{losses["num_pos"].item():.0f}'
                })
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        # 計算平均損失
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses

    def validate(self) -> dict:
        """驗證模型"""
        self.model.eval()
        val_losses = {
            'total_loss': 0.0,
            'offset_loss': 0.0,
            'scale_loss': 0.0, 
            'obj_loss': 0.0
        }
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                try:
                    # 數據移到設備
                    images = images.to(self.device, dtype=torch.float32)
                    
                    # 構建YOLO目標
                    yolo_targets = self._build_targets(targets).to(self.device)
                    
                    # 前向傳播
                    bbox_pred, obj_pred = self.model(images)
                    
                    # 計算損失
                    losses = self._compute_loss(bbox_pred, obj_pred, yolo_targets)
                    
                    # 累積損失
                    for key in val_losses:
                        if key in losses:
                            val_losses[key] += losses[key].item()
                            
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
        
        # 計算平均損失
        num_batches = len(self.val_loader) 
        for key in val_losses:
            val_losses[key] /= num_batches
            
        return val_losses

    def train(self, save_path: str = "models/detection_model.pth"):
        """執行完整訓練流程"""
        logger.info("Starting training...")
        logger.info(f"Model anchors: {self.anchor_boxes.tolist()}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # 訓練階段
            train_metrics = self.train_epoch(epoch)
            
            # 驗證階段
            val_metrics = self.validate()
            
            # 記錄訓練指標
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.4f} "
                f"(Off: {train_metrics['offset_loss']:.4f}, "
                f"Scl: {train_metrics['scale_loss']:.4f}, "
                f"Obj: {train_metrics['obj_loss']:.4f})"
            )
            
            # 記錄驗證指標
            logger.info( 
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Val Loss: {val_metrics['total_loss']:.4f} "
                f"(Off: {val_metrics['offset_loss']:.4f}, "
                f"Scl: {val_metrics['scale_loss']:.4f}, "
                f"Obj: {val_metrics['obj_loss']:.4f})"
            )
            
            # 保存最佳模型
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                if save_path:
                    save_dir = os.path.dirname(save_path)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                    
                    # 保存模型狀態和訓練信息
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'best_val_loss': best_val_loss,
                        'anchor_boxes': self.anchor_boxes,
                        'grid_size': self.grid_size
                    }, save_path)
                    
                    logger.info(f"Saved best model to {save_path} at epoch {epoch+1} with val loss {best_val_loss:.4f}")
        
        logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        return best_val_loss
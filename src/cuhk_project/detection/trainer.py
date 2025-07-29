import os
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
    """目标检测模型训练器（YOLO网格版本）"""
    
    @staticmethod
    def collate_fn(batch):
        """确保返回统一格式的批数据"""
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # 确保images是tensor
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
        else:
            images = torch.tensor(np.stack(images))
            
        return images, targets
        
    def __init__(self, 
                 model: SimpleDetectionModel, 
                 train_dataset: YOLOMFDataset,
                 val_dataset: YOLOMFDataset,
                 grid_size: tuple = (16, 16),  # 新增网格尺寸参数
                 num_anchors: int = 3,         # 新增锚框数量
                 batch_size: int = 4,
                 learning_rate: float = 0.001,
                 num_epochs: int = 10,
                 device: str = "cpu"):  # Force CPU to avoid MPS issues
        """
        初始化训练器
        
        参数:
            model: 检测模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            grid_size: 网格尺寸 (高度, 宽度)
            num_anchors: 锚框数量
            batch_size: 批大小
            learning_rate: 学习率
            num_epochs: 训练轮数
            device: 训练设备
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
        
        # 初始化锚框 (基于数据统计)
        self.anchor_boxes = self._init_anchors()
        
        # 初始化logger
        logger.info("Initializing detection trainer")
        logger.info(f"Grid size: {grid_size}, Anchors: {num_anchors}")
        logger.info(f"Anchor boxes: {self.anchor_boxes.tolist()}")
        
        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 损失函数
        self.bbox_loss_fn = torch.nn.SmoothL1Loss()  # 边界框坐标损失
        self.obj_loss_fn = torch.nn.BCEWithLogitsLoss()  # 目标存在损失
        
        logger.info(
            f"Trainer initialized: grid_size={grid_size}, anchors={num_anchors}, "
            f"batch_size={batch_size}, lr={learning_rate}, "
            f"epochs={num_epochs}, device={device}"
        )
    
    def _init_anchors(self):
        """基于数据集统计初始化锚框"""
        # 收集所有边界框尺寸
        all_boxes = []
        for i in range(len(self.train_dataset)):
            _, target = self.train_dataset[i]
            boxes = target['boxes']
            for box in boxes:
                w, h = box[2], box[3]  # 宽度和高度
                all_boxes.append([w, h])
        
        if not all_boxes:
            # 默认锚框尺寸
            return torch.tensor([
                [0.05, 0.2],  # 小物體
                [0.15, 0.5],  # 中物體
                [0.25, 0.8] 
            ])
        
        all_boxes = torch.tensor(all_boxes)
        
        # 使用K-means聚类确定锚框尺寸
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.num_anchors, random_state=0)
        kmeans.fit(all_boxes)
        
        anchors = torch.tensor(kmeans.cluster_centers_)
        logger.info(f"Computed anchors: {anchors.tolist()}")
        return anchors

    def _build_targets(self, targets):
        """构建YOLO格式的网格目标"""
        B = len(targets)
        H, W = self.grid_size
        # 目标张量: [batch, anchors, 5, grid_h, grid_w]
        # 5: [offset_x, offset_y, log(w_ratio), log(h_ratio), obj_confidence]
        targets_tensor = torch.zeros(B, self.num_anchors, 5, H, W)
        
        for b, target in enumerate(targets):
            boxes = target['boxes']
            for box in boxes:
                cx, cy, w, h = box
                
                # 计算网格位置
                grid_x = int(cx * W)
                grid_y = int(cy * H)
                
                # 确保在网格范围内
                grid_x = max(0, min(W-1, grid_x))
                grid_y = max(0, min(H-1, grid_y))
                
                # 计算偏移量
                offset_x = cx * W - grid_x
                offset_y = cy * H - grid_y
                
                # 计算最匹配的锚框
                ious = []
                for anchor in self.anchor_boxes:
                    # 计算IoU
                    box_area = w * h
                    anchor_area = anchor[0] * anchor[1]
                    inter_w = min(w, anchor[0])
                    inter_h = min(h, anchor[1])
                    inter_area = inter_w * inter_h
                    iou = inter_area / (box_area + anchor_area - inter_area)
                    ious.append(iou)
                
                best_anchor = torch.argmax(torch.tensor(ious))
                
                # 计算宽高比例的对数
                w_ratio = w / self.anchor_boxes[best_anchor][0]
                h_ratio = h / self.anchor_boxes[best_anchor][1]
                
                # 设置目标值
                targets_tensor[b, best_anchor, 0, grid_y, grid_x] = offset_x
                targets_tensor[b, best_anchor, 1, grid_y, grid_x] = offset_y
                targets_tensor[b, best_anchor, 2, grid_y, grid_x] = torch.log(w_ratio)
                targets_tensor[b, best_anchor, 3, grid_y, grid_x] = torch.log(h_ratio)
                targets_tensor[b, best_anchor, 4, grid_y, grid_x] = 1.0  # obj confidence
        
        return targets_tensor

    def train_epoch(self, epoch: int) -> dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        bbox_loss = 0.0
        obj_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 安全解包批数据
            images, targets = batch
            images = images.to(self.device)
            
            # 构建YOLO目标
            yolo_targets = self._build_targets(targets).to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            bbox_pred, obj_pred = self.model(images)
            
            # 提取目标值
            target_offsets = yolo_targets[..., :2, :, :]
            target_scales = yolo_targets[..., 2:4, :, :]
            target_obj = yolo_targets[..., 4, :, :]
            
            # 提取预测值
            pred_offsets = bbox_pred[..., :2, :, :]
            pred_scales = bbox_pred[..., 2:4, :, :]
            
            # 创建匹配预测形状的mask [B, anchors, 1, H, W]
            obj_mask = (target_obj > 0.5).unsqueeze(2)  # 添加第3维度
            
            # 偏移量损失
            if torch.any(obj_mask):
                # 扩展mask以匹配预测张量的形状
                expanded_mask = obj_mask.expand_as(pred_offsets)
                
                # 使用扩展后的mask选择元素
                offset_loss = self.bbox_loss_fn(
                    pred_offsets[expanded_mask].view(-1, 2),  # 重塑为 [N, 2]
                    target_offsets[expanded_mask].view(-1, 2)  # 重塑为 [N, 2]
                )
            else:
                offset_loss = torch.tensor(0.0).to(self.device)
            
            # 尺度损失
            if torch.any(obj_mask):
                # 扩展mask以匹配预测张量的形状
                expanded_mask = obj_mask.expand_as(pred_scales)
                
                # 使用扩展后的mask选择元素
                scale_loss = self.bbox_loss_fn(
                    pred_scales[expanded_mask].view(-1, 2),  # 重塑为 [N, 2]
                    target_scales[expanded_mask].view(-1, 2)  # 重塑为 [N, 2]
                )
            else:
                scale_loss = torch.tensor(0.0).to(self.device)
            
            # 目标存在损失
            obj_loss_val = self.obj_loss_fn(obj_pred, target_obj)
            
            # 总损失
            loss = offset_loss + scale_loss + obj_loss_val
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            bbox_loss += (offset_loss.item() + scale_loss.item())
            obj_loss += obj_loss_val.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bbox': f'{(offset_loss.item() + scale_loss.item()):.4f}',
                'obj': f'{obj_loss_val.item():.4f}'
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_bbox_loss = bbox_loss / len(self.train_loader)
        avg_obj_loss = obj_loss / len(self.train_loader)
        
        return {
            'total_loss': avg_loss,
            'bbox_loss': avg_bbox_loss,
            'obj_loss': avg_obj_loss
        }

    def validate(self) -> dict:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        bbox_loss = 0.0
        obj_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 安全解包批数据
                images, targets = batch
                images = images.to(self.device)
                
                # 构建YOLO目标
                yolo_targets = self._build_targets(targets).to(self.device)
                
                # 前向传播
                bbox_pred, obj_pred = self.model(images)
                
                # 提取目标值
                target_offsets = yolo_targets[..., :2, :, :]
                target_scales = yolo_targets[..., 2:4, :, :]
                target_obj = yolo_targets[..., 4, :, :]
                
                # 提取预测值
                pred_offsets = bbox_pred[..., :2, :, :]
                pred_scales = bbox_pred[..., 2:4, :, :]
                
                # 创建匹配预测形状的mask [B, anchors, 1, H, W]
                obj_mask = (target_obj > 0.5).unsqueeze(2)  # 添加第3维度
                
                # 偏移量损失
                if torch.any(obj_mask):
                    # 扩展mask以匹配预测张量的形状
                    expanded_mask = obj_mask.expand_as(pred_offsets)
                    
                    # 使用扩展后的mask选择元素
                    offset_loss = self.bbox_loss_fn(
                        pred_offsets[expanded_mask].view(-1, 2),  # 重塑为 [N, 2]
                        target_offsets[expanded_mask].view(-1, 2)  # 重塑为 [N, 2]
                    )
                else:
                    offset_loss = torch.tensor(0.0).to(self.device)
                
                # 尺度损失
                if torch.any(obj_mask):
                    # 扩展mask以匹配预测张量的形状
                    expanded_mask = obj_mask.expand_as(pred_scales)
                    
                    # 使用扩展后的mask选择元素
                    scale_loss = self.bbox_loss_fn(
                        pred_scales[expanded_mask].view(-1, 2),  # 重塑为 [N, 2]
                        target_scales[expanded_mask].view(-1, 2)  # 重塑为 [N, 2]
                    )
                else:
                    scale_loss = torch.tensor(0.0).to(self.device)
                
                # 目标存在损失
                obj_loss_val = self.obj_loss_fn(obj_pred, target_obj)
                
                # 总损失
                loss = offset_loss + scale_loss + obj_loss_val
                
                # 记录损失
                total_loss += loss.item()
                bbox_loss += (offset_loss.item() + scale_loss.item())
                obj_loss += obj_loss_val.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(self.val_loader)
        avg_bbox_loss = bbox_loss / len(self.val_loader)
        avg_obj_loss = obj_loss / len(self.val_loader)
        
        return {
            'total_loss': avg_loss,
            'bbox_loss': avg_bbox_loss,
            'obj_loss': avg_obj_loss
        }
    
    def train(self, save_path: str = "models/detection_model.pth"):
        """训练模型并保存"""
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.4f} "
                f"(Bbox: {train_metrics['bbox_loss']:.4f}, "
                f"Obj: {train_metrics['obj_loss']:.4f})"
            )
            
            # 验证
            val_metrics = self.validate()
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Val Loss: {val_metrics['total_loss']:.4f} "
                f"(Bbox: {val_metrics['bbox_loss']:.4f}, "
                f"Obj: {val_metrics['obj_loss']:.4f})"
            )
            
            # 保存最佳模型
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                if save_path:
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
                logger.info(f"Saved best model to {save_path} with val loss {best_val_loss:.4f}")
        
        logger.info("Training completed!")

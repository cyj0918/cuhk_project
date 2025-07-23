import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from cuhk_project.utils.logger import logger
from .model import SimpleDetectionModel
from .dataset import YOLOMFDataset

class DetectionTrainer:
    """目标检测模型训练器"""
    
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
                 batch_size: int = 4,
                 learning_rate: float = 0.001,
                 num_epochs: int = 10,
                 device: str = "mps" if torch.backends.mps.is_available() else "cpu"):
        """
        初始化训练器
        
        参数:
            model: 检测模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            batch_size: 批大小
            learning_rate: 学习率
            num_epochs: 训练轮数
            device: 训练设备
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        
        # 初始化logger
        logger.info("Initializing detection trainer")
        
        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 损失函数
        self.bbox_loss_fn = torch.nn.MSELoss()  # 边界框坐标损失
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss()  # 分类损失
        
        logger.info(
            f"Trainer initialized: batch_size={batch_size}, lr={learning_rate}, "
            f"epochs={num_epochs}, device={device}"
        )


    def train_epoch(self, epoch: int) -> dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        bbox_loss = 0.0
        cls_loss = 0.0
        max_boxes = 10
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch in progress_bar:
            # 安全解包批数据
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, targets = batch
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            images = images.to(self.device)
            
            # 处理批量目标数据
            max_boxes = 10  # 最大边界框数量
            gt_boxes = []
            gt_labels = []
            
            for target in targets:
                # 获取所有有效边界框
                valid_boxes = target['boxes'][target['boxes'].sum(dim=1) > 0]
                valid_labels = target['labels'][:len(valid_boxes)].float()
                
                # 填充或截断到max_boxes
                if len(valid_boxes) > max_boxes:
                    boxes = valid_boxes[:max_boxes]
                    labels = valid_labels[:max_boxes]
                else:
                    # 填充零边界框
                    pad_size = max_boxes - len(valid_boxes)
                    boxes = torch.cat([
                        valid_boxes,
                        torch.zeros((pad_size, 4))
                    ])
                    labels = torch.cat([
                        valid_labels,
                        torch.zeros(pad_size)
                    ])
                
                gt_boxes.append(boxes)
                gt_labels.append(labels)
            
            gt_boxes = torch.stack(gt_boxes).to(self.device)  # [batch, max_boxes, 4]
            gt_labels = torch.stack(gt_labels).to(self.device).unsqueeze(-1)  # [batch, max_boxes, 1]
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)  # [batch_size, 5]
            
            # 拆分输出并扩展维度
            pred_boxes = outputs[:, :4].unsqueeze(1).expand(-1, max_boxes, -1)  # [batch, max_boxes, 4]
            pred_cls = outputs[:, 4].unsqueeze(1).expand(-1, max_boxes)  # [batch, max_boxes]
            
            # 计算损失
            loss_bbox = self.bbox_loss_fn(pred_boxes, gt_boxes)
            loss_cls = self.cls_loss_fn(pred_cls, gt_labels.squeeze(-1))  # 去掉最后一个维度
            loss = loss_bbox + loss_cls
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            bbox_loss += loss_bbox.item()
            cls_loss += loss_cls.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bbox': f'{loss_bbox.item():.4f}',
                'cls': f'{loss_cls.item():.4f}'
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_bbox_loss = bbox_loss / len(self.train_loader)
        avg_cls_loss = cls_loss / len(self.train_loader)
        
        return {
            'total_loss': avg_loss,
            'bbox_loss': avg_bbox_loss,
            'cls_loss': avg_cls_loss
        }
    
    def validate(self) -> dict:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        bbox_loss = 0.0
        cls_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 安全解包批数据
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, targets = batch
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch)}")
                
                images = images.to(self.device)
                
                # 处理批量目标数据
                max_boxes = 10  # 最大边界框数量
                gt_boxes = []
                gt_labels = []
                
                for target in targets:
                    # 获取所有有效边界框
                    valid_boxes = target['boxes'][target['boxes'].sum(dim=1) > 0]
                    valid_labels = target['labels'][:len(valid_boxes)].float()
                    
                    # 填充或截断到max_boxes
                    if len(valid_boxes) > max_boxes:
                        boxes = valid_boxes[:max_boxes]
                        labels = valid_labels[:max_boxes]
                    else:
                        # 填充零边界框
                        pad_size = max_boxes - len(valid_boxes)
                        boxes = torch.cat([
                            valid_boxes,
                            torch.zeros((pad_size, 4))
                        ])
                        labels = torch.cat([
                            valid_labels,
                            torch.zeros(pad_size)
                        ])
                    
                    gt_boxes.append(boxes)
                    gt_labels.append(labels)
                
                gt_boxes = torch.stack(gt_boxes).to(self.device)  # [batch, max_boxes, 4]
                gt_labels = torch.stack(gt_labels).to(self.device).unsqueeze(-1)  # [batch, max_boxes, 1]
                
                # 前向传播
                outputs = self.model(images)  # [batch_size, 5]
                
                # 拆分输出并扩展维度
                pred_boxes = outputs[:, :4].unsqueeze(1).expand(-1, max_boxes, -1)  # [batch, max_boxes, 4]
                pred_cls = outputs[:, 4].unsqueeze(1).expand(-1, max_boxes)  # [batch, max_boxes]
                
                # 计算损失
                loss_bbox = self.bbox_loss_fn(pred_boxes, gt_boxes)
                loss_cls = self.cls_loss_fn(pred_cls.unsqueeze(1), gt_labels)
                loss = loss_bbox + loss_cls
                
                # 记录损失
                total_loss += loss.item()
                bbox_loss += loss_bbox.item()
                cls_loss += loss_cls.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(self.val_loader)
        avg_bbox_loss = bbox_loss / len(self.val_loader)
        avg_cls_loss = cls_loss / len(self.val_loader)
        
        return {
            'total_loss': avg_loss,
            'bbox_loss': avg_bbox_loss,
            'cls_loss': avg_cls_loss
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
                f"Cls: {train_metrics['cls_loss']:.4f})"
            )
            
            # 验证
            val_metrics = self.validate()
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Val Loss: {val_metrics['total_loss']:.4f} "
                f"(Bbox: {val_metrics['bbox_loss']:.4f}, "
                f"Cls: {val_metrics['cls_loss']:.4f})"
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

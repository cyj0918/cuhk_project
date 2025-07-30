from typing import Tuple, Optional
import torch
import torch.nn as nn
from torchvision.ops import nms as torch_nms
from cuhk_project.CNN.processors.conv import Conv
from cuhk_project.utils.logger import logger

class SimpleDetectionModel(nn.Module):
    """簡化的目標檢測模型 - 適配新數據集格式"""
    
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 16, 
                 kernel_size: int = 3,
                 num_anchors: int = 3, 
                 grid_size: Tuple[int, int] = (6, 32),
                 anchor_boxes: Optional[torch.Tensor] = None):
        """
        初始化檢測模型
        
        參數:
            in_channels: 輸入通道數
            out_channels: 卷積層輸出通道數
            kernel_size: 卷積核大小
            num_anchors: 錨框數量
            grid_size: 網格尺寸 (高度, 寬度)
            anchor_boxes: 外部傳入的錨框 [num_anchors, 2]，如果為None則使用默認錨框
        """
        super().__init__()
        
        # 保存網格尺寸和錨框數量
        self.num_anchors = num_anchors
        self.grid_size = grid_size
        
        logger.info(f"Initializing SimpleDetectionModel: grid_size={grid_size}, num_anchors={num_anchors}")
        
        # 構建骨幹網絡 - 只使用Conv處理器
        self.backbone = nn.Sequential(
            Conv(config={
                'in_channels': in_channels,
                'out_channels': out_channels,
                'kernel_size': kernel_size,
                'stride': 1,
                'padding': 1
            })
        )
        
        # YOLO檢測頭 - 輸出錨框預測
        self.detection_head = nn.Conv2d(
            out_channels,
            num_anchors * 5,  # 每個錨框: 4個坐標 + 1個置信度
            kernel_size=1
        )
        
        # 註冊錨框緩衝區
        if anchor_boxes is not None:
            # 使用外部傳入的錨框
            if anchor_boxes.shape != (num_anchors, 2):
                raise ValueError(f"anchor_boxes shape must be ({num_anchors}, 2), got {anchor_boxes.shape}")
            self.register_buffer('anchor_boxes', anchor_boxes.clone().detach())
            logger.info(f"Using external anchors: {anchor_boxes.tolist()}")
        else:
            # 使用默認錨框（適合96x512圖像）
            default_anchors = torch.tensor([
                [0.05, 0.15],  # 小物體
                [0.08, 0.25],  # 中等物體  
                [0.12, 0.35]   # 大物體
            ], dtype=torch.float32)
            self.register_buffer('anchor_boxes', default_anchors)
            logger.info(f"Using default anchors: {default_anchors.tolist()}")
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.parameters())} parameters")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        參數:
            x: 輸入圖像張量 [B, C, H, W]，已歸一化的CHW格式
            
        返回:
            bbox: 邊界框預測 [B, anchors, 4, grid_H, grid_W]
            obj: 目標置信度預測 [B, anchors, grid_H, grid_W]
        """
        # 驗證輸入格式
        if x.dim() != 4:
            raise ValueError(f"Input must be 4D tensor [B,C,H,W], got shape {x.shape}")
        
        B, C, H, W = x.shape
        if C != 3:
            logger.warning(f"Expected 3 input channels, got {C}")
        
        # 特徵提取
        features = self.backbone(x)  # [B, out_channels, feat_H, feat_W]
        
        # 檢測頭預測
        predictions = self.detection_head(features)  # [B, num_anchors*5, feat_H, feat_W]
        
        _, _, feat_H, feat_W = predictions.shape
        
        # 重塑為錨框格式
        predictions = predictions.view(B, self.num_anchors, 5, feat_H, feat_W)
        
        # 確保輸出網格尺寸匹配
        target_H, target_W = self.grid_size
        if feat_H != target_H or feat_W != target_W:
            logger.info(f"Resizing feature map from ({feat_H}, {feat_W}) to {self.grid_size}")
            # 使用自適應池化調整尺寸
            predictions = nn.functional.adaptive_avg_pool2d(
                predictions.view(B, -1, feat_H, feat_W),
                self.grid_size
            ).view(B, self.num_anchors, 5, target_H, target_W)
        
        # 分離邊界框和置信度預測
        bbox = predictions[:, :, :4, :, :]  # [B, anchors, 4, H, W]
        obj = predictions[:, :, 4, :, :]    # [B, anchors, H, W]
        
        return bbox, obj

    def predict(self, 
                x: torch.Tensor, 
                conf_thresh: float = 0.5, 
                iou_thresh: float = 0.3) -> list:
        """
        模型預測方法
        
        參數:
            x: 輸入圖像張量 [B, C, H, W]
            conf_thresh: 置信度閾值
            iou_thresh: NMS的IoU閾值
            
        返回:
            results: 每個batch的預測結果列表
        """
        self.eval()
        
        with torch.no_grad():
            # 前向傳播獲取原始輸出
            bbox_map, obj_map = self.forward(x)
            B, num_anchors, _, H, W = bbox_map.shape
            
            # 對置信度應用sigmoid激活
            obj_prob = torch.sigmoid(obj_map)
            
            device = x.device
            results = []
            
            # 處理每個batch
            for b in range(B):
                batch_boxes = []
                batch_scores = []
                
                # 獲取錨框數據
                anchors = self.anchor_boxes.to(device)
                
                # 遍歷所有網格位置和錨框
                for anchor_idx in range(num_anchors):
                    anchor_w = anchors[anchor_idx, 0]
                    anchor_h = anchors[anchor_idx, 1]
                    
                    for y in range(H):
                        for x_coord in range(W):
                            confidence = obj_prob[b, anchor_idx, y, x_coord]
                            
                            # 置信度過濾
                            if confidence < conf_thresh:
                                continue
                            
                            # 獲取邊界框預測值
                            dx = bbox_map[b, anchor_idx, 0, y, x_coord]
                            dy = bbox_map[b, anchor_idx, 1, y, x_coord]
                            dw = bbox_map[b, anchor_idx, 2, y, x_coord]
                            dh = bbox_map[b, anchor_idx, 3, y, x_coord]
                            
                            # 解碼邊界框坐標
                            # 中心坐標（應用sigmoid確保在網格內）
                            cx = (x_coord + torch.sigmoid(dx)) / W
                            cy = (y + torch.sigmoid(dy)) / H
                            
                            # 尺寸（指數變換）
                            w = anchor_w * torch.exp(torch.clamp(dw, -10, 10))  # 防止溢出
                            h = anchor_h * torch.exp(torch.clamp(dh, -10, 10))
                            
                            # 坐標範圍限制
                            cx = torch.clamp(cx, 0.0, 1.0)
                            cy = torch.clamp(cy, 0.0, 1.0)
                            w = torch.clamp(w, 0.01, 1.0)
                            h = torch.clamp(h, 0.01, 1.0)
                            
                            # 邊界檢查
                            x1, y1 = cx - w/2, cy - h/2
                            x2, y2 = cx + w/2, cy + h/2
                            
                            if x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
                                batch_boxes.append(torch.tensor([cx, cy, w, h], device=device))
                                batch_scores.append(confidence)
                
                # 轉換為張量並應用NMS
                if len(batch_boxes) > 0:
                    boxes_tensor = torch.stack(batch_boxes)
                    scores_tensor = torch.stack(batch_scores)
                    
                    # 轉換為xyxy格式進行NMS
                    x1 = boxes_tensor[:, 0] - boxes_tensor[:, 2] / 2
                    y1 = boxes_tensor[:, 1] - boxes_tensor[:, 3] / 2
                    x2 = boxes_tensor[:, 0] + boxes_tensor[:, 2] / 2
                    y2 = boxes_tensor[:, 1] + boxes_tensor[:, 3] / 2
                    xyxy_boxes = torch.stack([x1, y1, x2, y2], dim=1)
                    
                    # 應用NMS
                    try:
                        keep = torch_nms(xyxy_boxes, scores_tensor, iou_thresh)
                        final_boxes = boxes_tensor[keep]
                        final_scores = scores_tensor[keep]
                    except Exception as e:
                        logger.warning(f"NMS failed: {str(e)}, using all boxes")
                        final_boxes = boxes_tensor
                        final_scores = scores_tensor
                else:
                    final_boxes = torch.zeros((0, 4), device=device, dtype=torch.float32)
                    final_scores = torch.zeros(0, device=device, dtype=torch.float32)
                
                # 添加當前batch的結果
                results.append({
                    'boxes': final_boxes,
                    'scores': final_scores
                })
            
            return results

    def get_anchor_boxes(self) -> torch.Tensor:
        """獲取當前使用的錨框"""
        return self.anchor_boxes.clone()
    
    def update_anchor_boxes(self, new_anchors: torch.Tensor):
        """更新錨框（用於訓練過程中的動態調整）"""
        if new_anchors.shape != (self.num_anchors, 2):
            raise ValueError(f"New anchors shape must be ({self.num_anchors}, 2)")
        
        self.anchor_boxes.data.copy_(new_anchors)
        logger.info(f"Updated anchor boxes: {new_anchors.tolist()}")
    
    def get_model_info(self) -> dict:
        """獲取模型配置信息"""
        return {
            'num_anchors': self.num_anchors,
            'grid_size': self.grid_size,
            'anchor_boxes': self.anchor_boxes.tolist(),
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
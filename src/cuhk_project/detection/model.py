from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
from torchvision.ops import nms as torch_nms
from cuhk_project.utils.logger import logger

class SimpleDetectionModel(nn.Module):
    """簡化的目標檢測模型 - 完整實現版本"""
    
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 16, 
                 kernel_size: int = 3,
                 num_anchors: int = 3, 
                 grid_size: Tuple[int, int] = (6, 32),
                 num_classes: int = 2,  # 包含背景類
                 anchor_boxes: Optional[torch.Tensor] = None):
        """
        初始化檢測模型
        
        參數:
            in_channels: 輸入通道數
            out_channels: 卷積層輸出通道數
            kernel_size: 卷積核大小
            num_anchors: 錨框數量
            grid_size: 網格尺寸 (高度, 寬度)
            num_classes: 類別數量（包含背景）
            anchor_boxes: 外部傳入的錨框 [num_anchors, 2]
        """
        super().__init__()
        
        # 保存網格尺寸和錨框數量
        self.num_anchors = num_anchors
        self.grid_size = grid_size
        self.num_classes = num_classes
        
        logger.info(f"Initializing SimpleDetectionModel:")
        logger.info(f"  - Grid size: {grid_size}")
        logger.info(f"  - Num anchors: {num_anchors}")
        logger.info(f"  - Num classes: {num_classes}")
        
        # 構建骨幹網絡 - 增強版本
        self.backbone = nn.Sequential(
            # 第一層
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # 第二層
            nn.Conv2d(out_channels, out_channels*2, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # 第三層
            nn.Conv2d(out_channels*2, out_channels*4, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # 額外的卷積層提升特徵表達能力
            nn.Conv2d(out_channels*4, out_channels*4, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels*4),
            nn.ReLU(inplace=True)
        )
        
        # 檢測頭 - 三個分支
        self.bbox_conv = nn.Conv2d(out_channels*4, num_anchors * 4, kernel_size=1)
        self.cls_conv = nn.Conv2d(out_channels*4, num_anchors * num_classes, kernel_size=1)
        self.obj_conv = nn.Conv2d(out_channels*4, num_anchors * 1, kernel_size=1)
        
        # 註冊錨框緩衝區
        if anchor_boxes is not None:
            if anchor_boxes.shape != (num_anchors, 2):
                raise ValueError(f"anchor_boxes shape must be ({num_anchors}, 2), got {anchor_boxes.shape}")
            self.register_buffer('anchor_boxes', anchor_boxes.clone().detach())
            logger.info(f"Using external anchors: {anchor_boxes.tolist()}")
        else:
            # 默認錨框 - 調整為更合適的尺寸
            default_anchors = torch.tensor([
                [0.04, 0.12],   # 小目標
                [0.08, 0.24],   # 中等目標  
                [0.16, 0.48]    # 大目標
            ], dtype=torch.float32)
            self.register_buffer('anchor_boxes', default_anchors)
            logger.info(f"Using default anchors: {default_anchors.tolist()}")
        
        # 初始化權重
        self._initialize_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model initialized: {total_params:,} total params, {trainable_params:,} trainable")

    def _initialize_weights(self):
        """初始化模型權重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 特殊初始化 - obj_conv 的 bias 設為負值，降低初始置信度
        nn.init.constant_(self.obj_conv.bias, -4.0)
        
        logger.info("Model weights initialized")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        返回:
            bbox: 邊界框預測 [B, anchors, 4, grid_H, grid_W]
            cls: 分類預測 [B, anchors, num_classes, grid_H, grid_W]
            obj: 目標置信度預測 [B, anchors, grid_H, grid_W]
        """
        # 檢查輸入維度
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (BCHW), got {x.dim()}D")
        
        B, C, H, W = x.shape
        logger.debug(f"Forward input shape: {x.shape}")
        
        # 特徵提取
        features = self.backbone(x)
        _, _, feat_H, feat_W = features.shape
        logger.debug(f"Feature map shape: {features.shape}")
        
        # 三個分支預測
        bbox_pred = self.bbox_conv(features)
        cls_pred = self.cls_conv(features) 
        obj_pred = self.obj_conv(features)
        
        # 重塑為錨框格式
        target_H, target_W = self.grid_size
        
        bbox_pred = bbox_pred.view(B, self.num_anchors, 4, feat_H, feat_W)
        cls_pred = cls_pred.view(B, self.num_anchors, self.num_classes, feat_H, feat_W)
        obj_pred = obj_pred.view(B, self.num_anchors, feat_H, feat_W)
        
        # 調整尺寸到目標網格
        if feat_H != target_H or feat_W != target_W:
            logger.debug(f"Resizing from ({feat_H}, {feat_W}) to ({target_H}, {target_W})")
            
            bbox_pred = nn.functional.adaptive_avg_pool2d(
                bbox_pred.view(B, -1, feat_H, feat_W),
                self.grid_size
            ).view(B, self.num_anchors, 4, target_H, target_W)
            
            cls_pred = nn.functional.adaptive_avg_pool2d(
                cls_pred.view(B, -1, feat_H, feat_W),
                self.grid_size
            ).view(B, self.num_anchors, self.num_classes, target_H, target_W)
            
            obj_pred = nn.functional.adaptive_avg_pool2d(
                obj_pred.view(B, self.num_anchors, feat_H, feat_W),
                self.grid_size
            )
        
        logger.debug(f"Output shapes: bbox={bbox_pred.shape}, cls={cls_pred.shape}, obj={obj_pred.shape}")
        
        return bbox_pred, cls_pred, obj_pred

    def predict(self, 
                x: torch.Tensor, 
                conf_thresh: float = 0.1, 
                iou_thresh: float = 0.3) -> List[Dict]:
        """
        完整的預測方法實現 - 這是修正的關鍵部分
        
        參數:
            x: 輸入圖像張量 [B, C, H, W]
            conf_thresh: 置信度閾值
            iou_thresh: NMS的IoU閾值
            
        返回:
            List[Dict]: 每個圖像的預測結果
        """
        self.eval()
        device = x.device
        B = x.size(0)
        H, W = self.grid_size
        
        logger.debug(f"Predicting for batch size {B} with conf_thresh={conf_thresh}, iou_thresh={iou_thresh}")
        
        with torch.no_grad():
            # 前向傳播
            bbox_pred, cls_pred, obj_pred = self.forward(x)
            
            # 轉換置信度
            obj_prob = torch.sigmoid(obj_pred)  # [B, anchors, H, W]
            
            # 統計置信度分布
            logger.debug(f"Confidence stats: min={obj_prob.min():.6f}, max={obj_prob.max():.6f}, mean={obj_prob.mean():.6f}")
            logger.debug(f"Above {conf_thresh}: {(obj_prob >= conf_thresh).sum().item()}/{obj_prob.numel()}")
            
            batch_results = []
            
            for b in range(B):
                batch_boxes = []
                batch_scores = []
                batch_labels = []
                
                # 遍歷每個錨框和網格位置
                for anchor_idx in range(self.num_anchors):
                    anchor_w = self.anchor_boxes[anchor_idx, 0]
                    anchor_h = self.anchor_boxes[anchor_idx, 1]
                    
                    for y in range(H):
                        for x_coord in range(W):
                            confidence = obj_prob[b, anchor_idx, y, x_coord]
                            
                            # 置信度過濾
                            if confidence < conf_thresh:
                                continue
                            
                            # 獲取邊界框預測值
                            dx = bbox_pred[b, anchor_idx, 0, y, x_coord]
                            dy = bbox_pred[b, anchor_idx, 1, y, x_coord]
                            dw = bbox_pred[b, anchor_idx, 2, y, x_coord]
                            dh = bbox_pred[b, anchor_idx, 3, y, x_coord]
                            
                            # 解碼邊界框坐標 - 關鍵修正
                            # 中心坐標（應用sigmoid確保在網格內）
                            cx = (x_coord + torch.sigmoid(dx)) / W
                            cy = (y + torch.sigmoid(dy)) / H
                            
                            # 尺寸（指數變換）
                            w = anchor_w * torch.exp(torch.clamp(dw, -10, 10))
                            h = anchor_h * torch.exp(torch.clamp(dh, -10, 10))
                            
                            # 坐標範圍限制
                            cx = torch.clamp(cx, 0.0, 1.0)
                            cy = torch.clamp(cy, 0.0, 1.0)
                            w = torch.clamp(w, 0.01, 1.0)
                            h = torch.clamp(h, 0.01, 1.0)
                            
                            # 邊界檢查 - 修正版本
                            x1, y1 = cx - w/2, cy - h/2
                            x2, y2 = cx + w/2, cy + h/2
                            
                            # 放寬邊界檢查，只要不完全超出邊界即可
                            if x2 > 0 and y2 > 0 and x1 < 1 and y1 < 1:
                                # 處理分類預測（如果需要）
                                if self.num_classes > 1:
                                    cls_logits = cls_pred[b, anchor_idx, :, y, x_coord]
                                    cls_prob = torch.softmax(cls_logits, dim=0)
                                    best_class = torch.argmax(cls_prob).item()
                                    class_conf = cls_prob[best_class].item()
                                    
                                    # 組合置信度（目標置信度 × 分類置信度）
                                    final_score = confidence.item() * class_conf
                                else:
                                    best_class = 1  # 假設只有一個前景類
                                    final_score = confidence.item()
                                
                                batch_boxes.append(torch.tensor([cx, cy, w, h], device=device))
                                batch_scores.append(final_score)
                                batch_labels.append(best_class)
                
                logger.debug(f"Batch {b}: Found {len(batch_boxes)} boxes before NMS")
                
                # 應用NMS
                if len(batch_boxes) > 0:
                    boxes_tensor = torch.stack(batch_boxes)
                    scores_tensor = torch.tensor(batch_scores, device=device)
                    labels_tensor = torch.tensor(batch_labels, device=device, dtype=torch.long)
                    
                    # 轉換為xyxy格式進行NMS
                    boxes_xyxy = self._cxcywh_to_xyxy(boxes_tensor)
                    
                    # 執行NMS
                    keep_indices = torch_nms(boxes_xyxy, scores_tensor, iou_thresh)
                    
                    final_boxes = boxes_tensor[keep_indices]
                    final_scores = scores_tensor[keep_indices]
                    final_labels = labels_tensor[keep_indices]
                    
                    logger.debug(f"Batch {b}: {len(final_boxes)} boxes after NMS")
                    
                    batch_results.append({
                        'boxes': final_boxes,
                        'scores': final_scores,
                        'labels': final_labels
                    })
                else:
                    batch_results.append({
                        'boxes': torch.empty(0, 4, device=device),
                        'scores': torch.empty(0, device=device),
                        'labels': torch.empty(0, device=device, dtype=torch.long)
                    })
        
        return batch_results

    def _cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """轉換cxcywh格式到xyxy格式"""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def get_model_info(self) -> dict:
        """獲取模型的基本信息"""
        return {
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "grid_size": self.grid_size,
            "anchor_boxes": self.anchor_boxes.tolist(),
            "num_anchors": self.num_anchors,
            "num_classes": self.num_classes,
            "input_size": "Any (will be processed)",
            "output_grid": self.grid_size
        }

    def get_anchor_boxes(self) -> torch.Tensor:
        """獲取錨框"""
        return self.anchor_boxes.clone()
    
    def update_anchor_boxes(self, new_anchors: torch.Tensor):
        """更新錨框"""
        if new_anchors.shape != (self.num_anchors, 2):
            raise ValueError(f"New anchors shape must be ({self.num_anchors}, 2)")
        self.anchor_boxes.data.copy_(new_anchors)
        logger.info(f"Updated anchor boxes: {new_anchors.tolist()}")

    def freeze_backbone(self):
        """凍結骨幹網絡權重"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_backbone(self):
        """解凍骨幹網絡權重"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """獲取特徵圖（用於可視化和調試）"""
        with torch.no_grad():
            features = self.backbone(x)
            return features
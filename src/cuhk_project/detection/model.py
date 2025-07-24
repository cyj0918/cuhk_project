from typing import Tuple
import torch
import torch.nn as nn
from cuhk_project.CNN.processors.conv import Conv
from cuhk_project.utils.logger import logger

class SimpleDetectionModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, kernel_size=3, num_anchors=3):
        super().__init__()
        # 特徵提取層
        self.backbone = nn.Sequential(
            Conv(config={
                'in_channels':3,
                'out_channels':16,
                'kernel_size':3,
                'stride':1,
                'padding':1
            }),
            nn.MaxPool2d(2),
            Conv(config={
                'in_channels':16,
                'out_channels':32,
                'kernel_size':3,
                'stride':1,
                'padding':1
            }),
            nn.MaxPool2d(2),
            Conv(config={
                'in_channels':32,
                'out_channels':64,
                'kernel_size':3,
                'stride':1,
                'padding':1
            })
        )
        
        # YOLO輸出層 (5 = 4個坐標 + 1個置信度)
        self.detection_head = nn.Conv2d(
            64, 
            num_anchors * (5), 
            kernel_size=1
        )
        self.num_anchors = num_anchors
        # Initialize anchor boxes (width, height) based on dataset analysis
        self.anchor_boxes = [
            [0.05, 0.2],  # 小物體
            [0.15, 0.5],  # 中物體
            [0.25, 0.8] 
        ]

    def forward(self, x):
        features = self.backbone(x)
        # [B, num_anchors*5, H, W]
        predictions = self.detection_head(features)
        B, _, H, W = predictions.shape
        
        # Reshape to [B, anchors, 5, H, W]
        predictions = predictions.view(B, self.num_anchors, 5, H, W)
        # Ensure spatial dimensions match between predictions and mask
        if H != 16 or W != 16:
            predictions = nn.functional.interpolate(
                predictions.view(B, -1, H, W),
                size=(16, 16),
                mode='bilinear'
            ).view(B, self.num_anchors, 5, 16, 16)
        # 分離坐標和置信度
        bbox = predictions[:, :, :4, :, :]  # [B, anchors, 4, H, W]
        obj = predictions[:, :, 4, :, :]    # [B, anchors, H, W]
        
        return bbox, obj

    def predict(self, x, conf_thresh=0.5, iou_thresh=0.5):
        """修正后的预测方法"""
        with torch.no_grad():
            # 获取原始输出
            bbox_map, obj_map = self.forward(x)
            B, num_anchors, _, H, W = bbox_map.shape
            
            # 应用激活函数
            obj_prob = torch.sigmoid(obj_map)
            bbox_map = torch.sigmoid(bbox_map)
            
            # 获取设备信息
            device = x.device
            
            all_boxes = []
            all_scores = []
            
            for b in range(B):
                batch_boxes = []
                batch_scores = []
                
                for anchor_idx in range(num_anchors):
                    anchor_w, anchor_h = self.anchor_boxes[anchor_idx]
                    anchor_w = anchor_w.to(device) if torch.is_tensor(anchor_w) else torch.tensor(anchor_w, device=device)
                    anchor_h = anchor_h.to(device) if torch.is_tensor(anchor_h) else torch.tensor(anchor_h, device=device)
                    
                    for y in range(H):
                        for x in range(W):
                            if obj_prob[b, anchor_idx, y, x] < conf_thresh:
                                continue
                                
                            # 解码预测框
                            dx, dy, dw, dh = bbox_map[b, anchor_idx, :, y, x]
                            
                            # 转换为绝对坐标
                            cx = (x + dx) / W
                            cy = (y + dy) / H
                            w = anchor_w * dw
                            h = anchor_h * dh
                            
                            # 确保坐标有效
                            cx = torch.clamp(cx, 0.0, 1.0)
                            cy = torch.clamp(cy, 0.0, 1.0)
                            w = torch.clamp(w, 0.01, 0.99)
                            h = torch.clamp(h, 0.01, 0.99)
                            
                            batch_boxes.append(torch.stack([cx, cy, w, h]))
                            batch_scores.append(obj_prob[b, anchor_idx, y, x])
                
                # 转换为tensor
                if len(batch_boxes) > 0:
                    boxes_tensor = torch.stack(batch_boxes).to(device)
                    scores_tensor = torch.stack(batch_scores).to(device)
                    
                    # 应用NMS
                    keep = self.nms(boxes_tensor, scores_tensor, iou_thresh)
                    boxes_tensor = boxes_tensor[keep]
                    scores_tensor = scores_tensor[keep]
                else:
                    boxes_tensor = torch.zeros((0, 4), device=device)
                    scores_tensor = torch.zeros(0, device=device)
                
                all_boxes.append(boxes_tensor)
                all_scores.append(scores_tensor)
            
            return {
                'boxes': all_boxes[0],  # 假设batch_size=1
                'scores': all_scores[0]
            }

    @staticmethod
    def nms(boxes, scores, threshold):
        """修正后的NMS实现"""
        if boxes.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        
        # 转换到xyxy格式
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(descending=True)
        
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            
            if order.numel() == 1:
                break
            
            # 计算IoU
            xx1 = x1[order[1:]].clamp(min=x1[i].item())
            yy1 = y1[order[1:]].clamp(min=y1[i].item())
            xx2 = x2[order[1:]].clamp(max=x2[i].item())
            yy2 = y2[order[1:]].clamp(max=y2[i].item())
            
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # 保留IoU低于阈值的框
            mask = iou <= threshold
            order = order[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
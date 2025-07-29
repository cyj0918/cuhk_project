from typing import Tuple
import torch
import torch.nn as nn
from torchvision.ops import nms as torch_nms  # 使用torchvision的NMS实现
from cuhk_project.CNN.processors.conv import Conv
from cuhk_project.utils.logger import logger

class SimpleDetectionModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, kernel_size=3, 
                 num_anchors=3, grid_size=(6, 32)):  # 使用6x32网格尺寸
        super().__init__()
        self.backbone = nn.Sequential(
            Conv(config={
                'in_channels': in_channels,
                'out_channels': out_channels,
                'kernel_size': kernel_size,
                'stride': 1,
                'padding': 1
            })
        )
        
        # YOLO输出层
        self.detection_head = nn.Conv2d(
            out_channels,
            num_anchors * 5,  # 5 = 4个坐标 + 1个置信度
            kernel_size=1
        )
        self.num_anchors = num_anchors
        self.grid_size = grid_size  # 保存网格尺寸
        
        # 锚框 - 根据实际目标尺寸调整
        # 96x512图像，目标通常较小
        anchor_boxes = [
            [0.05, 0.15],  # 小物体 (4.8x76.8像素)
            [0.08, 0.25],  # 中等物体 (7.68x128像素)
            [0.12, 0.35]   # 大物体 (11.52x179.2像素)
        ]
        
        # 注册锚框为缓冲区
        self.register_buffer('anchor_boxes', torch.tensor(anchor_boxes, dtype=torch.float32))

    def forward(self, x):
        features = self.backbone(x)
        # [B, num_anchors*5, H, W]
        predictions = self.detection_head(features)
        B, _, H, W = predictions.shape
        
        # 重塑为[B, anchors, 5, H, W]
        predictions = predictions.view(B, self.num_anchors, 5, H, W)
        
        # 检查特征图尺寸是否匹配网格尺寸
        if H != self.grid_size[0] or W != self.grid_size[1]:
            # 使用自适应池化确保输出尺寸匹配
            predictions = nn.functional.adaptive_avg_pool2d(
                predictions.view(B, -1, H, W),
                self.grid_size
            ).view(B, self.num_anchors, 5, self.grid_size[0], self.grid_size[1])
        
        # 分离坐标和置信度
        bbox = predictions[:, :, :4, :, :]  # [B, anchors, 4, H, W]
        obj = predictions[:, :, 4, :, :]    # [B, anchors, H, W]
        
        return bbox, obj
    
    def predict(self, x, conf_thresh=0.5, iou_thresh=0.3):
        """修正后的预测方法"""
        with torch.no_grad():
            # 获取原始输出
            bbox_map, obj_map = self.forward(x)
            B, num_anchors, _, H, W = bbox_map.shape
            
            # 只对置信度应用sigmoid
            obj_prob = torch.sigmoid(obj_map)
            
            # 获取设备信息
            device = x.device
            
            all_boxes = []
            all_scores = []
            
            for b in range(B):
                batch_boxes = []
                batch_scores = []
                
                # 获取当前batch的锚框数据
                anchors = self.anchor_boxes.to(device)
                
                for anchor_idx in range(num_anchors):
                    anchor_w = anchors[anchor_idx, 0]
                    anchor_h = anchors[anchor_idx, 1]
                    
                    for y in range(H):
                        for x in range(W):
                            confidence = obj_prob[b, anchor_idx, y, x]
                            if confidence < conf_thresh:
                                continue
                            
                            # 获取原始预测值
                            dx = bbox_map[b, anchor_idx, 0, y, x]
                            dy = bbox_map[b, anchor_idx, 1, y, x]
                            dw = bbox_map[b, anchor_idx, 2, y, x]
                            dh = bbox_map[b, anchor_idx, 3, y, x]
                            
                            # 正确解码预测框
                            # 1. 中心坐标偏移 (应用sigmoid)
                            cx = (x + torch.sigmoid(dx)) / W
                            cy = (y + torch.sigmoid(dy)) / H
                            
                            # 2. 尺寸缩放 (直接使用指数函数)
                            w = anchor_w * torch.exp(dw)
                            h = anchor_h * torch.exp(dh)
                            
                            # 确保坐标有效
                            cx = torch.clamp(cx, 0.0, 1.0)
                            cy = torch.clamp(cy, 0.0, 1.0)
                            w = torch.clamp(w, 0.01, 0.99)
                            h = torch.clamp(h, 0.01, 0.99)
                            
                            batch_boxes.append(torch.tensor([cx, cy, w, h], device=device))
                            batch_scores.append(confidence)
                
                # 转换为tensor
                if len(batch_boxes) > 0:
                    boxes_tensor = torch.stack(batch_boxes)
                    scores_tensor = torch.stack(batch_scores)
                    
                    # 应用NMS (使用torchvision实现)
                    # 转换为中心坐标到xyxy格式
                    x1 = boxes_tensor[:, 0] - boxes_tensor[:, 2] / 2
                    y1 = boxes_tensor[:, 1] - boxes_tensor[:, 3] / 2
                    x2 = boxes_tensor[:, 0] + boxes_tensor[:, 2] / 2
                    y2 = boxes_tensor[:, 1] + boxes_tensor[:, 3] / 2
                    xyxy_boxes = torch.stack([x1, y1, x2, y2], dim=1)
                    
                    keep = torch_nms(xyxy_boxes, scores_tensor, iou_thresh)
                    
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
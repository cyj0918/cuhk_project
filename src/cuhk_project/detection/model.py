import torch
import torch.nn as nn
from cuhk_project.CNN.processors.conv import Conv2dProcessor
from cuhk_project.utils.logger import configure_logging

class SimpleDetectionModel(nn.Module):
    """简单目标检测模型，仅包含一层卷积"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 16, kernel_size: int = 3):
        """
        初始化模型
        
        参数:
            in_channels: 输入通道数 (RGB图像为3)
            out_channels: 输出通道数
            kernel_size: 卷积核大小
        """
        super().__init__()
        self.logger = configure_logging(module="SimpleDetectionModel")
        self.logger.info("Initializing simple detection model")
        
        # 使用项目中的Conv2dProcessor作为卷积层
        self.conv = Conv2dProcessor(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1
        )
        
        # 输出层 - 每个锚点预测4个坐标+1个置信度+N个类别
        # 简化版：直接预测边界框坐标和类别概率
        self.fc_bbox = nn.Linear(out_channels, 4)  # 边界框坐标 (cx, cy, w, h)
        self.fc_class = nn.Linear(out_channels, 1)  # 二分类简化版 (后续可扩展)
        
        self.logger.info(
            f"Model created: in_channels={in_channels}, "
            f"out_channels={out_channels}, kernel_size={kernel_size}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 卷积特征提取
        features = self.conv(x)
        
        # 全局平均池化
        pooled = torch.mean(features, dim=[2, 3])
        
        # 预测边界框和类别
        bbox = self.fc_bbox(pooled)
        cls_prob = torch.sigmoid(self.fc_class(pooled))
        
        # 组合输出 [batch_size, 5] (cx, cy, w, h, confidence)
        return torch.cat([bbox, cls_prob], dim=1)
    
    def predict(self, x: torch.Tensor) -> dict:
        """预测接口"""
        with torch.no_grad():
            output = self.forward(x)
            return {
                'boxes': output[:, :4],
                'scores': output[:, 4]
            }

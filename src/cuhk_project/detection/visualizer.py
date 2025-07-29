import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import torch
import numpy as np

class DetectionVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_sample(self, image, pred_boxes, true_boxes, filename):
        """修正后的可视化函数"""
        # 创建figure和axes
        fig, ax = plt.subplots(1, figsize=(12, 8))
        
        # 确保图像是HWC格式的numpy数组
        if isinstance(image, torch.Tensor):
            image = image.cpu().permute(1, 2, 0).numpy()
        
        # 显示图像
        ax.imshow(image)
        
        # 转换boxes到像素坐标
        h, w = image.shape[:2]
        
        # 绘制预测框（红色）
        for box in pred_boxes:
            cx, cy, box_w, box_h = box
            x = (cx - box_w/2) * w
            y = (cy - box_h/2) * h
            rect = Rectangle(
                (x, y), box_w * w, box_h * h,
                linewidth=2, 
                edgecolor='r', 
                facecolor='none',
                label='Prediction'
            )
            ax.add_patch(rect)
        
        # 绘制真实框（绿色）
        for box in true_boxes:
            cx, cy, box_w, box_h = box
            x = (cx - box_w/2) * w
            y = (cy - box_h/2) * h
            rect = Rectangle(
                (x, y), box_w * w, box_h * h,
                linewidth=2, 
                edgecolor='g', 
                facecolor='none',
                label='Ground Truth'
            )
            ax.add_patch(rect)
        
        # 添加图例（避免重复）
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        # 保存图像
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
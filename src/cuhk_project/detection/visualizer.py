import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from cuhk_project.utils.logger import configure_logging
from .model import SimpleDetectionModel
from .dataset import YOLOMFDataset

class DetectionVisualizer:
    """目标检测结果可视化器"""
    
    def __init__(self,
                 model: SimpleDetectionModel,
                 dataset: YOLOMFDataset,
                 output_dir: str = "output/visualizations",
                 device: str = "mps" if torch.backends.mps.is_available() else "cpu"):
        """
        初始化可视化器
        
        参数:
            model: 检测模型
            dataset: 数据集
            output_dir: 可视化结果输出目录
            device: 设备
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.device = device
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化logger
        self.logger = configure_logging(module="DetectionVisualizer")
        self.logger.info(f"Initializing visualizer. Output will be saved to {self.output_dir}")
    
    def denormalize_bbox(self, bbox, width, height):
        """将归一化的边界框坐标转换为像素坐标"""
        cx, cy, w, h = bbox
        x = cx * width
        y = cy * height
        w = w * width
        h = h * height
        return (x - w/2, y - h/2, w, h)
    
    def visualize_sample(self, idx: int):
        """可视化单个样本的预测结果"""
        # 获取样本
        image, target = self.dataset[idx]
        orig_height, orig_width = target['orig_size'].tolist()
        
        # 转换为CHW格式
        image = image.unsqueeze(0).to(self.device)
        
        # 模型预测
        with torch.no_grad():
            prediction = self.model.predict(image)
        
        # 转换为numpy数组
        image_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        gt_boxes = target['boxes'].cpu().numpy()
        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        
        # 创建图像
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image_np)
        
        # 绘制真实边界框 (绿色)
        for box in gt_boxes:
            x, y, w, h = self.denormalize_bbox(box, orig_width, orig_height)
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        
        # 绘制预测边界框 (红色) 并显示置信度
        for i, box in enumerate(pred_boxes):
            x, y, w, h = self.denormalize_bbox(box, orig_width, orig_height)
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, f"{pred_scores[i]:.2f}", color='red', fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.7))
        
        # 添加标题
        plt.title(f"Sample {idx} - GT: Green, Pred: Red")
        
        # 保存图像
        output_path = self.output_dir / f"sample_{idx}.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def visualize_dataset(self, num_samples: int = 10):
        """可视化数据集中的多个样本"""
        self.logger.info(f"Visualizing {num_samples} samples from dataset")
        output_paths = []
        
        # 随机选择样本
        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        for idx in tqdm(indices, desc="Visualizing samples"):
            try:
                output_path = self.visualize_sample(idx)
                output_paths.append(output_path)
            except Exception as e:
                self.logger.error(f"Error visualizing sample {idx}: {str(e)}")
        
        self.logger.info(f"Visualization completed. Results saved to {self.output_dir}")
        return output_paths

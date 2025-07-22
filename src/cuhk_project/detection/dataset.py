# Dataset Loading
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# logger功能
from cuhk_project.utils.logger import configure_logging

class DetectionDataset(Dataset):
    """目标检测数据集加载器，支持YOLO格式标注"""
    
    def __init__(self, 
                 data_dir: str, 
                 split: str = 'train', 
                 transform: Optional[callable] = None,
                 target_size: Tuple[int, int] = (416, 416)):
        """
        初始化数据集
        
        参数:
            data_dir: 数据集根目录
            split: 数据集分割 (train/val/test)
            transform: 数据增强变换
            target_size: 目标图像尺寸 (宽, 高)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.classes = self._load_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 初始化logger
        self.logger = configure_logging(module="DetectionDataset")
        self.logger.info(f"Initializing {split} dataset from {data_dir}")
        
        # 加载数据样本
        self.samples = self._load_samples()
        self.logger.info(f"Loaded {len(self.samples)} {split} samples")
        
    def _load_classes(self) -> List[str]:
        """从classes.txt加载类别列表"""
        classes_file = self.data_dir / "classes.txt"
        if not classes_file.exists():
            self.logger.error(f"Classes file not found: {classes_file}")
            raise FileNotFoundError(f"Classes file not found: {classes_file}")
            
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
            
        self.logger.debug(f"Loaded {len(classes)} classes: {classes}")
        return classes
    
    def _load_samples(self) -> List[Dict]:
        """加载数据集样本"""
        split_file = self.data_dir / f"{self.split}.txt"
        if not split_file.exists():
            self.logger.error(f"Split file not found: {split_file}")
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, 'r') as f:
            sample_paths = [line.strip() for line in f.readlines()]
            
        samples = []
        for sample_path in sample_paths:
            img_path = self.data_dir / "images" / f"{sample_path}.png"
            ann_path = self.data_dir / "labels" / f"{sample_path}.txt"
            
            if not img_path.exists():
                self.logger.warning(f"Image not found: {img_path}, skipping")
                continue
                
            if not ann_path.exists():
                self.logger.warning(f"Annotation not found: {ann_path}, skipping")
                continue
                
            samples.append({
                'image_path': img_path,
                'annotation_path': ann_path,
                'id': sample_path
            })
            
        return samples
    
    def _parse_annotation(self, annotation_path: Path) -> List[Dict]:
        """解析TXT标注文件"""
        annotations = []
        try:
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        self.logger.warning(f"Invalid annotation line: {line} in {annotation_path}")
                        continue
                    
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # 验证坐标范围 (0-1)
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        self.logger.warning(f"Invalid coordinates in {annotation_path}: {cx},{cy},{w},{h}")
                        continue
                    
                    annotations.append({
                        'class_id': class_id,
                        'cx': cx,
                        'cy': cy,
                        'width': w,
                        'height': h
                    })
        except Exception as e:
            self.logger.error(f"Error parsing {annotation_path}: {str(e)}")
            raise
            
        return annotations
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """获取样本及其标注"""
        sample = self.samples[idx]
        
        # 加载图像 (PNG格式)
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            orig_width, orig_height = image.size
            
            # 调整大小
            image = image.resize(self.target_size)
            image = np.array(image) / 255.0
            image = image.transpose(2, 0, 1)  # HWC to CHW
            image = torch.tensor(image, dtype=torch.float32)
        except Exception as e:
            self.logger.error(f"Error loading image {sample['image_path']}: {str(e)}")
            raise
        
        # 加载标注
        annotations = self._parse_annotation(sample['annotation_path'])
        
        # 准备目标张量
        boxes = []
        labels = []
        for ann in annotations:
            boxes.append([ann['cx'], ann['cy'], ann['width'], ann['height']])
            labels.append(ann['class_id'])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([orig_height, orig_width])
        }
        
        # 应用数据增强
        if self.transform:
            image, target = self.transform(image, target)
            
        return image, target

# 示例数据增强变换
class RandomHorizontalFlip:
    """随机水平翻转"""
    
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, image: torch.Tensor, target: Dict) -> Tuple:
        if torch.rand(1) < self.p:
            image = torch.flip(image, dims=[2])  # 沿宽度维度翻转
            boxes = target['boxes']
            boxes[:, 0] = 1.0 - boxes[:, 0]  # 翻转中心x坐标
            target['boxes'] = boxes
        return image, target
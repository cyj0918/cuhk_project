# Dataset Loading
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from cuhk_project.utils.logger import configure_logging

class YOLOMFDataset(Dataset):
    """YOLO格式目标检测数据集加载器，匹配您的数据结构"""
    
    def __init__(self, 
                 base_dir: str = "data/yolo_mf_dataset",
                 split: str = 'train', 
                 transform: Optional[callable] = None,
                 target_size: Tuple[int, int] = (416, 416)):
        """
        初始化数据集
        
        参数:
            base_dir: 数据集根目录
            split: 数据集分割 (train/val/test)
            transform: 数据增强变换
            target_size: 目标图像尺寸 (宽, 高)
        """
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # 初始化logger
        self.logger = configure_logging(module=f"YOLOMFDataset.{split}")
        self.logger.info(f"Initializing dataset from {self.base_dir}")
        
        # 加载类别和样本
        self.classes = self._load_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._scan_samples()
        self.logger.info(f"Loaded {len(self.samples)} samples with {len(self.classes)} classes")

    def _load_classes(self) -> List[str]:
        """从classes.txt加载类别列表"""
        classes_file = self.base_dir / "classes.txt"
        if not classes_file.exists():
            self.logger.error(f"Classes file not found: {classes_file}")
            raise FileNotFoundError(f"Classes file not found: {classes_file}")
            
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
            
        if not classes:
            self.logger.error("No classes found in classes.txt")
            raise ValueError("Empty classes.txt")
            
        return classes

    def _scan_samples(self) -> List[Dict]:
        """扫描图像和标注文件"""
        img_dir = self.base_dir / "images" / self.split
        label_dir = self.base_dir / "labels" / self.split
        
        if not img_dir.exists():
            self.logger.error(f"Image directory not found: {img_dir}")
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
            
        samples = []
        valid_extensions = {'.png', '.jpg', '.jpeg'}
        
        # 扫描图像目录
        for img_path in img_dir.glob('*'):
            if img_path.suffix.lower() not in valid_extensions:
                continue
                
            # 构建对应的标注路径
            label_path = label_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                self.logger.warning(f"Label not found for {img_path.name}, skipping")
                continue
                
            samples.append({
                'image_path': img_path,
                'annotation_path': label_path,
                'id': img_path.stem
            })
            
        if not samples:
            self.logger.error(f"No valid samples found in {img_dir}")
            raise ValueError(f"No valid samples found in {img_dir}")
            
        return samples

    def _parse_annotation(self, annotation_path: Path) -> List[Dict]:
        """解析YOLO格式标注文件"""
        annotations = []
        try:
            with open(annotation_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) != 5:
                        self.logger.warning(
                            f"Invalid annotation in {annotation_path.name} line {line_num}: {line}"
                        )
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # 验证数据有效性
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            self.logger.warning(
                                f"Invalid bbox in {annotation_path.name} line {line_num}: {line}"
                            )
                            continue
                            
                        annotations.append({
                            'class_id': class_id,
                            'cx': cx,
                            'cy': cy,
                            'width': w,
                            'height': h
                        })
                    except ValueError as e:
                        self.logger.warning(
                            f"Invalid number in {annotation_path.name} line {line_num}: {line}"
                        )
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error reading {annotation_path}: {str(e)}")
            raise
            
        return annotations

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """获取样本及其标注"""
        sample = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            orig_size = image.size  # (width, height)
            
            # 调整大小并归一化
            image = image.resize(self.target_size)
            image = np.array(image) / 255.0
            image = image.transpose(2, 0, 1)  # HWC to CHW
            image = torch.tensor(image, dtype=torch.float32)
        except Exception as e:
            self.logger.error(f"Error loading {sample['image_path']}: {str(e)}")
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
            'orig_size': torch.tensor([orig_size[1], orig_size[0]])  # (height, width)
        }
        
        # 应用数据增强
        if self.transform:
            image, target = self.transform(image, target)
            
        return image, target

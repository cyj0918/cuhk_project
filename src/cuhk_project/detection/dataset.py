import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from cuhk_project.utils.logger import logger 

class YOLOMFDataset(Dataset):
    """YOLO格式目标检测数据集加载器（修正版）"""
    
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
        
        # 初始化logger并启用传播
        logger.info(f"Initializing dataset from {self.base_dir}")
        
        # 加载类别和样本
        self.classes = self._load_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._scan_samples()
        
        # 新增：验证数据集完整性
        self._validate_dataset()
        logger.info(f"Loaded {len(self.samples)} samples with {len(self.classes)} classes")
    
    def __len__(self) -> int:
        """返回数据集中的样本数量"""
        return len(self.samples)

    def _validate_dataset(self):
        """验证数据集完整性"""
        for sample in self.samples[:min(10, len(self.samples))]:  # 检查前10个样本
            try:
                img = Image.open(sample['image_path'])
                annotations = self._parse_annotation(sample['annotation_path'])
                
                # 验证标注与图像尺寸匹配
                for ann in annotations:
                    if not (0 <= ann['cx'] <= 1 and 0 <= ann['cy'] <= 1 and 
                           0 < ann['width'] <= 1 and 0 < ann['height'] <= 1):
                        logger.warning(f"Invalid annotation in {sample['image_path'].name}: "
                                      f"cx={ann['cx']}, cy={ann['cy']}, "
                                      f"w={ann['width']}, h={ann['height']}")
            except Exception as e:
                logger.error(f"Validation failed for {sample['image_path'].name}: {str(e)}")
                raise

    def _parse_annotation(self, annotation_path: Path) -> List[Dict]:
        """解析YOLO格式标注文件（增强版）"""
        annotations = []
        try:
            with open(annotation_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) != 5:
                        logger.warning(
                            f"Invalid annotation in {annotation_path.name} line {line_num}: {line}"
                        )
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # 更严格的数据验证
                        if not (0 <= cx <= 1 and 0 <= cy <= 1):
                            logger.warning(
                                f"Invalid center in {annotation_path.name} line {line_num}: {line}"
                            )
                            continue
                            
                        if w <= 0 or h <= 0 or w > 1 or h > 1:
                            logger.warning(
                                f"Invalid dimensions in {annotation_path.name} line {line_num}: {line}"
                            )
                            continue
                            
                        # 检查边界框是否完全在图像内
                        x1, y1 = cx - w/2, cy - h/2
                        x2, y2 = cx + w/2, cy + h/2
                        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                            logger.warning(
                                f"Box out of bounds in {annotation_path.name} line {line_num}: {line}"
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
                        logger.warning(
                            f"Invalid number in {annotation_path.name} line {line_num}: {line}"
                        )
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading {annotation_path}: {str(e)}")
            raise
            
        logger.debug(f"Loaded {len(annotations)} valid boxes from {annotation_path}")
        return annotations

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """获取样本及其标注（修正版）"""
        sample = self.samples[idx]
        try:
            # 加载图像并获取原始尺寸
            image = Image.open(sample['image_path']).convert('RGB')
            orig_width, orig_height = image.size
            
            # 调整图像大小并归一化
            image = image.resize(self.target_size)
            image = np.array(image) / 255.0
            image = image.transpose(2, 0, 1)  # HWC to CHW
            image = torch.tensor(image, dtype=torch.float32)
            
            # 加载标注（不再需要调整坐标，因为YOLO格式已经是归一化的）
            annotations = self._parse_annotation(sample['annotation_path'])
            
            # 准备目标张量
            boxes = []
            labels = []
            for ann in annotations:
                boxes.append([ann['cx'], ann['cy'], ann['width'], ann['height']])
                labels.append(ann['class_id'])
            
            # 填充逻辑保持不变...
            
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'orig_size': torch.tensor([orig_height, orig_width]),
                'resized_size': torch.tensor(self.target_size[::-1]),  # (height, width)
                'num_boxes': torch.tensor(len(annotations))
            }
            
            # 应用数据增强
            if self.transform:
                image, target = self.transform(image, target)
            
            # 验证输出
            self._validate_output(image, target)
            return image, target
            
        except Exception as e:
            logger.error(f"Error processing sample {idx} ({sample['image_path']}): {str(e)}")
            raise

    def _validate_output(self, image: torch.Tensor, target: Dict):
        """验证输出数据有效性"""
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image must be torch.Tensor")
            
        boxes = target['boxes']
        if boxes.dim() != 2 or boxes.size(1) != 4:
            raise ValueError(f"Invalid boxes shape: {boxes.shape}")
            
        # 验证坐标范围
        for box in boxes:
            cx, cy, w, h = box.tolist()
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"Invalid box coordinates: {box.tolist()}")
    
    def _load_classes(self) -> List[str]:
        """从classes.txt加载类别列表"""
        classes_file = self.base_dir / "classes.txt"
        if not classes_file.exists():
            logger.error(f"Classes file not found: {classes_file}")
            raise FileNotFoundError(f"Classes file not found: {classes_file}")
            
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
            
        if not classes:
            logger.error("No classes found in classes.txt")
            raise ValueError("Empty classes.txt")
            
        return classes
    
    def _scan_samples(self) -> List[Dict]:
        """扫描图像和标注文件"""
        img_dir = self.base_dir / "images" / self.split
        label_dir = self.base_dir / "labels" / self.split
        
        if not img_dir.exists():
            logger.error(f"Image directory not found: {img_dir}")
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
            
        samples = []
        valid_extensions = {'.png', '.jpg', '.jpeg'}
        
        # 扫描图像目录
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in valid_extensions:
                continue
                
            # 构建对应的标注路径
            label_path = label_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                logger.warning(f"Label not found for {img_path.name}, skipping")
                continue
                
            samples.append({
                'image_path': img_path,
                'annotation_path': label_path,
                'id': img_path.stem
            })
            
        if not samples:
            logger.error(f"No valid samples found in {img_dir}")
            raise ValueError(f"No valid samples found in {img_dir}")
            
        return samples
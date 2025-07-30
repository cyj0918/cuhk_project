import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from cuhk_project.utils.logger import logger 

class YOLOMFDataset(Dataset):
    @staticmethod
    def collate_fn(batch):
        """处理不同数量boxes的自定义collate函数"""
        images = []
        targets = []
        
        max_boxes = max(len(item[1]['boxes']) for item in batch) if batch else 0
        
        for image, target in batch:
            images.append(image)
            
            num_boxes = len(target['boxes'])
            padded_boxes = torch.zeros((max_boxes, 4), dtype=torch.float32)
            if num_boxes > 0:
                padded_boxes[:num_boxes] = target['boxes']
            
            padded_labels = torch.zeros(max_boxes, dtype=torch.int64)
            if num_boxes > 0:
                padded_labels[:num_boxes] = target['labels']
            
            padded_target = {
                'boxes': padded_boxes,
                'labels': padded_labels,
                'image_id': target['image_id'],
                'orig_size': target['orig_size'],
                'resized_size': target['resized_size'],
                'num_boxes': torch.tensor(num_boxes),
                'grid_size': target['grid_size']
            }
            targets.append(padded_target)
        
        images = torch.stack(images)  # 要求所有image已是Tensor
        return images, targets

    def __init__(self, 
                 base_dir: str = "data/yolo_mf_dataset",
                 split: str = 'train', 
                 target_size: Tuple[int, int] = (96, 512),
                 grid_size: Tuple[int, int] = (6, 32)):
        self.base_dir = Path(base_dir)
        self.split = split
        self.target_height, self.target_width = target_size
        self.grid_height, self.grid_width = grid_size
        
        logger.info(f"Initializing dataset from {self.base_dir}")
        logger.info(f"Target size: {target_size}, Grid size: {grid_size}")
        
        self.classes = self._load_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._scan_samples()
        
        self._validate_dataset()
        logger.info(f"Loaded {len(self.samples)} samples with {len(self.classes)} classes")
    
    def __len__(self) -> int:
        return len(self.samples)

    def _validate_dataset(self):
        for sample in self.samples[:min(10, len(self.samples))]:
            try:
                img = Image.open(sample['image_path'])
                annotations = self._parse_annotation(sample['annotation_path'])
                
                for ann in annotations:
                    if not (0 <= ann['cx'] <= 1 and 0 <= ann['cy'] <= 1 and 
                           0 < ann['width'] <= 1 and 0 < ann['height'] <= 1):
                        logger.warning(f"Invalid annotation in {sample['image_path'].name}")
            except Exception as e:
                logger.error(f"Validation failed for {sample['image_path'].name}: {str(e)}")
                raise

    def _parse_annotation(self, annotation_path: Path) -> List[Dict]:
        annotations = []
        try:
            with open(annotation_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                    elif len(parts) == 4:
                        class_id = 0  # 默认类别
                        cx, cy, w, h = map(float, parts[0:4])
                    else:
                        logger.warning(f"Invalid annotation in {annotation_path.name} line {line_num}")
                        continue
                    
                    # 边界检查
                    x1, y1 = cx - w/2, cy - h/2
                    x2, y2 = cx + w/2, cy + h/2
                    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                        logger.warning(f"Box out of bounds in {annotation_path.name} line {line_num}")
                        continue
                        
                    annotations.append({
                        'class_id': class_id,
                        'cx': cx,
                        'cy': cy,
                        'width': w,
                        'height': h
                    })
        except Exception as e:
            logger.error(f"Error reading {annotation_path}: {str(e)}")
            raise
            
        return annotations

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        sample = self.samples[idx]
        try:
            # 1. 加载图像
            image = Image.open(sample['image_path']).convert('RGB')
            orig_width, orig_height = image.size
            
            # 2. 调整大小并转换为Tensor
            image = image.resize((self.target_width, self.target_height))
            image_np = np.array(image, dtype=np.float32) / 255.0  # 归一化 [0,1]
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # HWC -> CHW
            
            # 3. 加载标注并修正类别索引
            annotations = self._parse_annotation(sample['annotation_path'])
            boxes = []
            labels = []
            
            for ann in annotations:
                labels.append(int(ann['class_id']) + 1)  # 原始0->1, 1->2...
                
                # 直接使用cxcywh格式
                boxes.append([
                    ann['cx'], 
                    ann['cy'], 
                    ann['width'], 
                    ann['height']
                ])
            
            # 4. 创建目标字典
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
            
            target = {
                'boxes': boxes_tensor,
                'labels': torch.tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'orig_size': torch.tensor([orig_width, orig_height]),
                'resized_size': torch.tensor([self.target_width, self.target_height]),
                'num_boxes': torch.tensor(len(annotations)),
                'grid_size': torch.tensor([self.grid_height, self.grid_width])
            }
            
            # 5. 验证输出
            self._validate_output(image_tensor.unsqueeze(0), [target])  # 模拟batch
            
            return image_tensor, target
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            empty_image = torch.zeros((3, self.target_height, self.target_width), dtype=torch.float32)
            empty_target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'orig_size': torch.tensor([self.target_width, self.target_height]),
                'resized_size': torch.tensor([self.target_width, self.target_height]),
                'num_boxes': torch.tensor(0),
                'grid_size': torch.tensor([self.grid_height, self.grid_width])
            }
            return empty_image, empty_target

    # 新增：实际调用验证
    def _validate_output(self, image: torch.Tensor, targets: List[Dict]):
        """验证输出数据有效性"""
        _, C, H, W = image.shape
        assert (H, W) == (self.target_height, self.target_width), \
            f"Image size mismatch: {H}x{W} vs {self.target_height}x{self.target_width}"
            
        for target in targets:
            boxes = target['boxes']
            labels = target['labels']
            
            assert boxes.dim() == 2 and boxes.shape[1] == 4, \
                f"Invalid boxes shape: {boxes.shape}"
                
            assert len(boxes) == len(labels), \
                f"Boxes/Labels count mismatch: {len(boxes)} vs {len(labels)}"
                
            # 验证坐标范围 (cxcywh格式)
            for box in boxes:
                cx, cy, w, h = box.tolist()
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    logger.warning(f"Box out of bounds: cx={cx}, cy={cy}, w={w}, h={h}")
                    
                # 检查转换后的边界
                x_min = cx - w/2
                y_min = cy - h/2
                x_max = cx + w/2
                y_max = cy + h/2
                if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
                    logger.warning(f"Box bounds invalid after conversion: {x_min},{y_min},{x_max},{y_max}")

    def _load_classes(self) -> List[str]:
        classes_file = self.base_dir / "classes.txt"
        if not classes_file.exists():
            raise FileNotFoundError(f"Classes file not found: {classes_file}")
            
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
            
        if not classes:
            raise ValueError("Empty classes.txt")
            
        # 添加背景类作为索引0
        if classes[0] != "background":
            classes.insert(0, "background")
            logger.info("Added background class at index 0")
            
        return classes
    
    def _scan_samples(self) -> List[Dict]:
        img_dir = self.base_dir / "images" / self.split
        label_dir = self.base_dir / "labels" / self.split
        
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
            
        samples = []
        valid_extensions = {'.png', '.jpg', '.jpeg'}
        
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in valid_extensions:
                continue
                
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
            raise ValueError(f"No valid samples found in {img_dir}")
            
        return samples

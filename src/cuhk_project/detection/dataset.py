import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from cuhk_project.utils.logger import logger 

class YOLOMFDataset(Dataset):
    """YOLO格式目标检测数据集加载器（优化版）"""

    @staticmethod
    def collate_fn(batch):
        """处理不同数量boxes的自定义collate函数"""
        images = []
        targets = []
        
        # 找出batch中最多的boxes数量
        max_boxes = max(len(item[1]['boxes']) for item in batch) if batch else 0
        
        for image, target in batch:
            images.append(image)
            
            # 对boxes进行padding
            num_boxes = len(target['boxes'])
            padded_boxes = torch.zeros((max_boxes, 4), dtype=torch.float32)
            if num_boxes > 0:
                padded_boxes[:num_boxes] = target['boxes']
            
            # 对labels进行padding
            padded_labels = torch.zeros(max_boxes, dtype=torch.int64)
            if num_boxes > 0:
                padded_labels[:num_boxes] = target['labels']
            
            # 创建新的target字典
            padded_target = {
                'boxes': padded_boxes,
                'labels': padded_labels,
                'image_id': target['image_id'],
                'orig_size': target['orig_size'],
                'resized_size': target['resized_size'],
                'num_boxes': torch.tensor(num_boxes),  # 记录实际boxes数量
                'grid_size': target['grid_size']
            }
            targets.append(padded_target)
        
        # 堆叠图像
        images = torch.stack(images)
        return images, targets

    def __init__(self, 
                 base_dir: str = "data/yolo_mf_dataset",
                 split: str = 'train', 
                 transform: Optional[callable] = None,
                 target_size: Tuple[int, int] = (96, 512),  # (height, width)
                 grid_size: Tuple[int, int] = (6, 32)):     # (grid_height, grid_width)
        """
        初始化数据集
        
        参数:
            base_dir: 数据集根目录
            split: 数据集分割 (train/val/test)
            transform: 数据增强变换
            target_size: 目标图像尺寸 (高, 宽)
            grid_size: 网格尺寸 (高, 宽)
        """
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = transform
        self.target_height, self.target_width = target_size  # 分解为高和宽
        self.grid_height, self.grid_width = grid_size        # 分解为网格高和宽
        
        # 初始化logger并启用传播
        logger.info(f"Initializing dataset from {self.base_dir}")
        logger.info(f"Target size: {target_size}, Grid size: {grid_size}")
        
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
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                    elif len(parts) == 4:
                        class_id = 0  # Default class if not specified
                        cx, cy, w, h = map(float, parts[0:4])
                    else:
                        logger.warning(f"Invalid annotation in {annotation_path.name} line {line_num}: {line}")
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
        """获取样本及其标注（优化版）"""
        sample = self.samples[idx]
        try:
            # 加载图像并获取原始尺寸
            image = Image.open(sample['image_path']).convert('RGB')
            orig_width, orig_height = image.size
            
            # 调整图像大小并归一化
            image = image.resize((self.target_width, self.target_height))
            image = np.array(image) / 255.0
            image = image.transpose(2, 0, 1)  # HWC to CHW
            image = torch.tensor(image, dtype=torch.float32)
            
            # 加载标注
            annotations = self._parse_annotation(sample['annotation_path'])
            
            # 准备目标张量 - 确保所有值在有效范围内
            boxes = []
            labels = []
            for ann in annotations:
                try:
                    # 验证并规范化坐标值
                    cx = max(0.01, min(0.99, float(ann['cx'])))
                    cy = max(0.01, min(0.99, float(ann['cy'])))
                    w = max(0.01, min(0.99, float(ann['width'])))
                    h = max(0.01, min(0.99, float(ann['height'])))
                    
                    boxes.append([cx, cy, w, h])
                    labels.append(int(ann['class_id']))
                except Exception as e:
                    logger.error(f"Invalid annotation in {sample['image_path']}: {ann}, error: {str(e)}")
                    continue
            
            # 严格验证boxes
            if not boxes:
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            else:
                # 确保所有boxes都是4个值
                invalid_boxes = [box for box in boxes if len(box) != 4]
                if invalid_boxes:
                    raise ValueError(f"Invalid boxes in {sample['image_path']}: {invalid_boxes}")
                
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                if boxes_tensor.dim() != 2 or boxes_tensor.size(1) != 4:
                    raise ValueError(f"Boxes tensor has invalid shape: {boxes_tensor.shape}")
            
            # 创建目标字典
            target = {
                'boxes': boxes_tensor.reshape(-1, 4),  # 强制Nx4形状
                'labels': torch.tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'orig_size': torch.tensor([orig_height, orig_width]),
                'resized_size': torch.tensor([self.target_height, self.target_width]),
                'num_boxes': torch.tensor(len(annotations)),
                'grid_size': torch.tensor([self.grid_height, self.grid_width])
            }
            
            # 应用数据增强
            if self.transform:
                try:
                    image, target = self.transform(image, target)
                    # transform后验证
                    if target['boxes'].dim() != 2 or target['boxes'].size(1) != 4:
                        raise ValueError(f"Transform corrupted boxes shape: {target['boxes'].shape}")
                except Exception as e:
                    logger.error(f"Transform failed for {sample['image_path']}: {str(e)}")
                    raise
            
            return image, target
            
        except Exception as e:
            logger.error(f"Error processing sample {idx} ({sample['image_path']}): {str(e)}")
            # 返回空样本避免训练中断
            empty_image = torch.zeros((3, self.target_height, self.target_width), dtype=torch.float32)
            empty_target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'orig_size': torch.tensor([self.target_height, self.target_width]),
                'resized_size': torch.tensor([self.target_height, self.target_width]),
                'num_boxes': torch.tensor(0),
                'grid_size': torch.tensor([self.grid_height, self.grid_width])
            }
            return empty_image, empty_target

    def _validate_output(self, image: torch.Tensor, target: Dict):
        """验证输出数据有效性"""
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image must be torch.Tensor")
            
        # 验证图像尺寸
        _, channels, height, width = image.shape
        if (height, width) != (self.target_height, self.target_width):
            raise ValueError(f"Image size mismatch: expected ({self.target_height}, {self.target_width}), got ({height}, {width})")
            
        boxes = target['boxes']
        if boxes.dim() != 2 or boxes.size(1) != 4:
            raise ValueError(f"Invalid boxes shape: {boxes.shape}")
            
        # 验证坐标范围
        for box in boxes:
            cx, cy, w, h = box.tolist()
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            
            # 允许轻微超出范围（0-1），但需要记录警告
            if not (0 <= x1 < x2 <= 1) or not (0 <= y1 < y2 <= 1):
                if abs(x1) > 0.05 or abs(y1) > 0.05 or abs(x2-1) > 0.05 or abs(y2-1) > 0.05:
                    logger.warning(f"Box coordinates out of bounds: {box.tolist()}")
    
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
            
        # 添加背景类作为索引0
        if "background" not in classes:
            classes.insert(0, "background")
            logger.info("Added background class to class list")
            
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
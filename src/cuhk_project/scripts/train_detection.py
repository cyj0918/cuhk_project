# python -m src.cuhk_project.scripts.train_detection --data-dir data/yolo_mf_dataset --batch-size 4 --epochs 10 --out-channels 16 --num-anchors 3 --grid-size 6x32 --save-path models/test_5.pth

import argparse
import torch
from pathlib import Path
from cuhk_project.utils.logger import logger
from cuhk_project.detection.dataset import YOLOMFDataset
from cuhk_project.detection.model import SimpleDetectionModel
from cuhk_project.detection.trainer import DetectionTrainer

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="Train YOLO object detection model")
    
    # 數據相關參數
    parser.add_argument("--data-dir", type=str, default="data/yolo_mf_dataset",
                        help="Dataset directory path")
    
    # 訓練參數
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    
    # 模型參數
    parser.add_argument("--out-channels", type=int, default=16,
                        help="Number of output channels in backbone")
    parser.add_argument("--num-anchors", type=int, default=3,
                        help="Number of anchor boxes")
    parser.add_argument("--grid-size", type=str, default="6x32",
                        help="Grid size (format: height x width)")
    
    # 存儲參數
    parser.add_argument("--save-path", type=str, default="models/detection_model.pth",
                        help="Path to save the trained model")
    
    # 設備參數
    parser.add_argument("--device", type=str, default="cpu",
                        help="Training device (cpu/cuda)")
    
    # 調試參數
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation")
    
    return parser.parse_args()

def validate_arguments(args):
    """驗證命令行參數的有效性"""
    # 檢查數據目錄
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")
    
    # 檢查必要的子目錄
    required_dirs = ["images/train", "images/val", "labels/train", "labels/val"]
    for subdir in required_dirs:
        if not (data_path / subdir).exists():
            logger.warning(f"Directory not found: {data_path / subdir}")
    
    # 檢查classes.txt
    if not (data_path / "classes.txt").exists():
        raise FileNotFoundError(f"Classes file not found: {data_path / 'classes.txt'}")
    
    # 驗證網格尺寸格式
    try:
        grid_parts = args.grid_size.split('x')
        if len(grid_parts) != 2:
            raise ValueError("Grid size format should be 'height x width'")
        grid_h, grid_w = map(int, grid_parts)
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError("Grid dimensions must be positive")
    except ValueError as e:
        raise ValueError(f"Invalid grid size '{args.grid_size}': {str(e)}")
    
    # 驗證其他參數
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if args.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")
    if args.num_anchors <= 0:
        raise ValueError("Number of anchors must be positive")

def create_datasets(args, grid_size, image_size):
    """創建訓練和驗證數據集"""
    logger.info("Creating datasets...")
    
    try:
        # 創建訓練數據集
        train_dataset = YOLOMFDataset(
            base_dir=args.data_dir,
            split="train",
            target_size=image_size,     # (height, width)
            grid_size=grid_size         # (height, width)
        )
        
        # 創建驗證數據集
        val_dataset = YOLOMFDataset(
            base_dir=args.data_dir,
            split="val", 
            target_size=image_size,     # (height, width)
            grid_size=grid_size         # (height, width)
        )
        
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
        
        # 驗證數據集非空
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        if len(val_dataset) == 0:
            logger.warning("Validation dataset is empty")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        logger.error(f"Failed to create datasets: {str(e)}")
        raise

def create_model(args, grid_size, anchor_boxes=None):
    """創建檢測模型"""
    logger.info("Creating detection model...")
    
    try:
        model = SimpleDetectionModel(
            in_channels=3,
            out_channels=args.out_channels,
            kernel_size=3,
            num_anchors=args.num_anchors,
            grid_size=grid_size,
            anchor_boxes=anchor_boxes    # 傳入計算出的錨框
        )
        
        # 打印模型信息
        model_info = model.get_model_info()
        logger.info(f"Model created successfully:")
        logger.info(f"  - Total parameters: {model_info['total_params']:,}")
        logger.info(f"  - Trainable parameters: {model_info['trainable_params']:,}")
        logger.info(f"  - Grid size: {model_info['grid_size']}")
        logger.info(f"  - Anchor boxes: {model_info['anchor_boxes']}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to create model: {str(e)}")
        raise

def create_trainer(model, train_dataset, val_dataset, args, grid_size):
    """創建訓練器"""
    logger.info("Creating trainer...")
    
    try:
        trainer = DetectionTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            grid_size=grid_size,
            num_anchors=args.num_anchors,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            device=args.device
        )
        
        logger.info("Trainer created successfully")
        return trainer
        
    except Exception as e:
        logger.error(f"Failed to create trainer: {str(e)}")
        raise

def main():
    """主訓練流程"""
    # 初始化日誌
    logger.info("="*60)
    logger.info("Starting YOLO object detection training")
    logger.info("="*60)
    
    # 解析和驗證參數
    args = parse_arguments()
    validate_arguments(args)
    
    # 解析網格尺寸
    grid_height, grid_width = map(int, args.grid_size.split('x'))
    grid_size = (grid_height, grid_width)
    
    # 固定圖像尺寸為96x512
    image_size = (96, 512)  # (height, width)
    
    # 打印配置信息
    logger.info("Training configuration:")
    logger.info(f"  - Data directory: {args.data_dir}")
    logger.info(f"  - Image size: {image_size}")
    logger.info(f"  - Grid size: {grid_size}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.lr}")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Output channels: {args.out_channels}")
    logger.info(f"  - Number of anchors: {args.num_anchors}")
    logger.info(f"  - Device: {args.device}")
    logger.info(f"  - Save path: {args.save_path}")
    
    try:
        # 1. 創建數據集
        train_dataset, val_dataset = create_datasets(args, grid_size, image_size)
        
        # 2. 計算最優錨框
        logger.info("Computing optimal anchor boxes...")
        anchor_boxes = DetectionTrainer.compute_anchors(train_dataset, args.num_anchors)
        logger.info(f"Computed anchor boxes: {anchor_boxes.tolist()}")
        
        # 3. 創建模型
        model = create_model(args, grid_size, anchor_boxes)
        
        # 4. 處理恢復訓練
        start_epoch = 0
        if args.resume:
            logger.info(f"Resuming training from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                logger.info(f"Resumed from epoch {start_epoch}")
            else:
                model.load_state_dict(checkpoint)
                logger.info("Loaded model weights")
        
        # 5. 創建訓練器
        trainer = create_trainer(model, train_dataset, val_dataset, args, grid_size)
        
        # 6. 只驗證模式
        if args.validate_only:
            logger.info("Running validation only...")
            val_metrics = trainer.validate()
            logger.info("Validation results:")
            for key, value in val_metrics.items():
                logger.info(f"  - {key}: {value:.4f}")
            return
        
        # 7. 創建保存目錄
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 8. 開始訓練
        logger.info("Starting training process...")
        best_loss = trainer.train(save_path=str(save_path))
        
        # 9. 訓練完成
        logger.info("="*60)
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {best_loss:.4f}")
        logger.info(f"Model saved to: {args.save_path}")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        logger.info("Training process finished")

if __name__ == "__main__":
    main()
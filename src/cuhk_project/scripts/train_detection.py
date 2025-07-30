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
                        help="Grid size (format: heightxwidth)")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes including background")
    
    # 存儲參數
    parser.add_argument("--save-path", type=str, default="models/detection_model.pth",
                        help="Path to save the trained model")
    
    # 設備參數  
    parser.add_argument("--device", type=str, default=None,
                        help="Training device (cpu/cuda/auto)")
    
    # 可視化參數
    parser.add_argument("--enable-vis", action="store_true", default=True,
                        help="Enable visualization during training")
    parser.add_argument("--vis-freq", type=int, default=10,
                        help="Visualization frequency (every N batches)")
    parser.add_argument("--vis-dir", type=str, default="debug_visualizations",
                        help="Visualization output directory")
    
    # 調試參數
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with extra logging")
    
    return parser.parse_args()


def validate_arguments(args):
    """驗證命令行參數的有效性"""
    # 檢查數據目錄
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")
    
    # 檢查必要的子目錄
    required_dirs = ["images/train", "images/val", "labels/train", "labels/val"]
    missing_dirs = []
    for subdir in required_dirs:
        if not (data_path / subdir).exists():
            missing_dirs.append(subdir)
    
    if missing_dirs:
        logger.warning(f"Missing directories: {missing_dirs}")
        # 檢查是否至少有訓練數據
        if "images/train" in missing_dirs or "labels/train" in missing_dirs:
            raise FileNotFoundError("Training data directories are required")
    
    # 檢查classes.txt
    classes_file = data_path / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_file}")
    
    # 驗證網格尺寸格式
    try:
        grid_parts = args.grid_size.lower().split('x')
        if len(grid_parts) != 2:
            raise ValueError("Grid size format should be 'heightxwidth' (e.g., '6x32')")
        grid_h, grid_w = map(int, grid_parts)
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError("Grid dimensions must be positive")
        args._grid_parsed = (grid_h, grid_w)  # 暫存解析結果
    except ValueError as e:
        raise ValueError(f"Invalid grid size '{args.grid_size}': {str(e)}")
    
    # 驗證其他參數
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if args.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if args.lr <= 0 or args.lr > 1:
        raise ValueError("Learning rate must be positive and <= 1")
    if args.num_anchors <= 0 or args.num_anchors > 10:
        raise ValueError("Number of anchors must be positive and <= 10")
    if args.num_classes < 1:
        raise ValueError("Number of classes must be >= 1")
    
    # 驗證設備設置
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    
    logger.info(f"Using device: {args.device}")


def get_num_classes_from_dataset(data_dir):
    """從數據集中獲取類別數量"""
    try:
        classes_file = Path(data_dir) / "classes.txt"
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        
        # 如果第一個類別不是background，會在dataset中自動添加
        if classes and classes[0] != "background":
            num_classes = len(classes) + 1  # +1 for background
        else:
            num_classes = len(classes)
            
        logger.info(f"Detected {num_classes} classes from dataset")
        return max(num_classes, 2)  # 至少2個類別（背景+前景）
        
    except Exception as e:
        logger.warning(f"Failed to read classes from dataset: {e}, using default 2 classes")
        return 2


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
            logger.warning("Validation dataset is empty - will skip validation")
        
        # 獲取類別信息
        logger.info(f"Dataset classes: {train_dataset.classes}")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        logger.error(f"Failed to create datasets: {str(e)}")
        raise


def create_model(args, grid_size, num_classes, anchor_boxes=None):
    """創建檢測模型"""
    logger.info("Creating detection model...")
    
    try:
        model = SimpleDetectionModel(
            in_channels=3,
            out_channels=args.out_channels,
            kernel_size=3,
            num_anchors=args.num_anchors,
            grid_size=grid_size,
            num_classes=num_classes,  # 使用從數據集檢測到的類別數
            anchor_boxes=anchor_boxes
        )
        
        # 移動到指定設備
        model = model.to(args.device)
        
        # 打印模型信息
        model_info = model.get_model_info()
        logger.info(f"Model created successfully:")
        logger.info(f"  - Total parameters: {model_info['total_params']:,}")
        logger.info(f"  - Trainable parameters: {model_info['trainable_params']:,}")
        logger.info(f"  - Grid size: {model_info['grid_size']}")
        logger.info(f"  - Number of classes: {model_info['num_classes']}")
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
            device=args.device,
            enable_visualization=args.enable_vis,
            vis_output_dir=args.vis_dir,
            vis_frequency=args.vis_freq
        )
        
        logger.info("Trainer created successfully")
        return trainer
        
    except Exception as e:
        logger.error(f"Failed to create trainer: {str(e)}")
        raise


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """加載檢查點"""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state")
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_loss = checkpoint.get('best_val_loss', float('inf'))
            
            # 更新錨框（如果存在）
            if 'anchor_boxes' in checkpoint:
                model.update_anchor_boxes(checkpoint['anchor_boxes'])
                logger.info("Updated anchor boxes from checkpoint")
            
            logger.info(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.4f}")
            return start_epoch, best_loss
            
        else:
            # 舊格式的檢查點，只包含模型權重
            model.load_state_dict(checkpoint)
            logger.info("Loaded model weights (legacy format)")
            return 0, float('inf')
            
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        raise


def setup_logging(debug=False):
    """設置日誌級別"""
    if debug:
        import logging
        logging.getLogger('cuhk_project').setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")


def main():
    """主訓練流程"""
    try:
        # 解析和驗證參數
        args = parse_arguments()
        validate_arguments(args)
        
        # 設置日誌
        setup_logging(args.debug)
        
        # 初始化日誌
        logger.info("=" * 60)
        logger.info("Starting YOLO Object Detection Training")
        logger.info("=" * 60)
        
        # 解析網格尺寸
        grid_size = args._grid_parsed
        
        # 固定圖像尺寸為96x512
        image_size = (96, 512)  # (height, width)
        
        # 打印配置信息
        logger.info("Training Configuration:")
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
        logger.info(f"  - Visualization: {'Enabled' if args.enable_vis else 'Disabled'}")
        
        # 1. 創建數據集
        train_dataset, val_dataset = create_datasets(args, grid_size, image_size)
        
        # 2. 獲取類別數量
        num_classes = len(train_dataset.classes)
        logger.info(f"Using {num_classes} classes: {train_dataset.classes}")
        
        # 3. 計算最優錨框
        logger.info("Computing optimal anchor boxes...")
        anchor_boxes = DetectionTrainer.compute_anchors(train_dataset, args.num_anchors)
        logger.info(f"Computed anchor boxes: {anchor_boxes.tolist()}")
        
        # 4. 創建模型
        model = create_model(args, grid_size, num_classes, anchor_boxes)
        
        # 5. 創建訓練器
        trainer = create_trainer(model, train_dataset, val_dataset, args, grid_size)
        
        # 6. 處理恢復訓練
        start_epoch = 0
        best_loss = float('inf')
        if args.resume:
            start_epoch, best_loss = load_checkpoint(
                model, trainer.optimizer, args.resume, args.device
            )
        
        # 7. 只驗證模式
        if args.validate_only:
            logger.info("Running validation only...")
            val_metrics = trainer.validate()
            logger.info("Validation Results:")
            for key, value in val_metrics.items():
                logger.info(f"  - {key}: {value:.4f}")
            return
        
        # 8. 創建保存目錄
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 9. 開始訓練
        logger.info("Starting training process...")
        logger.info(f"Training will run for {args.epochs} epochs")
        
        final_loss = trainer.train(save_path=str(save_path))
        
        # 10. 訓練完成
        logger.info("=" * 60)
        logger.info("Training Completed Successfully!")
        logger.info(f"Final validation loss: {final_loss:.4f}")
        logger.info(f"Model saved to: {args.save_path}")
        logger.info("=" * 60)
        
        # 11. 測試模型加載
        try:
            logger.info("Testing model loading...")
            test_checkpoint = torch.load(args.save_path, map_location='cpu', weights_only=False)
            logger.info("Model checkpoint loads successfully")
        except Exception as e:
            logger.warning(f"Model checkpoint test failed: {e}")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("Training process finished")


if __name__ == "__main__":
    main()

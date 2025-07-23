# python -m src.cuhk_project.scripts.train_detection --data-dir data/yolo_mf_dataset --batch-size 4 --epochs 10 --out-channels 16 --save-path models/test_1.pth

import argparse
from cuhk_project.utils.logger import logger
from cuhk_project.detection.dataset import YOLOMFDataset
from cuhk_project.detection.model import SimpleDetectionModel
from cuhk_project.detection.trainer import DetectionTrainer

def main():
    # 配置日志
    logger.info("Starting object detection training")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train object detection model")
    parser.add_argument("--data-dir", type=str, default="data/yolo_mf_dataset",
                        help="Dataset directory")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--out-channels", type=int, default=16,
                        help="Number of output channels in convolution layer")
    parser.add_argument("--save-path", type=str, default="models/detection_model.pth",
                        help="Path to save trained model")
    
    args = parser.parse_args()
    
    logger.info(f"Training configuration: "
                f"data_dir={args.data_dir}, batch_size={args.batch_size}, "
                f"epochs={args.epochs}, lr={args.lr}, out_channels={args.out_channels}")
    
    # 创建数据集
    try:
        train_dataset = YOLOMFDataset(
            base_dir=args.data_dir, split="train", target_size=(416, 416)
        )
        val_dataset = YOLOMFDataset(
            base_dir=args.data_dir, split="val", target_size=(416, 416)
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    except Exception as e:
        logger.error(f"Failed to create datasets: {str(e)}")
        return
    
    # 创建模型
    try:
        model = SimpleDetectionModel(
            in_channels=3, 
            out_channels=args.out_channels,
            kernel_size=3
        )
        logger.info("Model created successfully")
    except Exception as e:
        logger.error(f"Failed to create model: {str(e)}")
        return
    
    # 创建训练器
    try:
        trainer = DetectionTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs
        )
        logger.info("Trainer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to create trainer: {str(e)}")
        return
    
    # 开始训练
    try:
        trainer.train(save_path=args.save_path)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.critical(f"Training failed: {str(e)}")

if __name__ == "__main__":
    main()

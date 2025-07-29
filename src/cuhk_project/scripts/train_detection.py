# python -m src.cuhk_project.scripts.train_detection --data-dir data/yolo_mf_dataset --batch-size 4 --epochs 10 --out-channels 16 --num-anchors 3 --grid-size 6x32 --save-path models/test_5.pth
import argparse
from cuhk_project.utils.logger import logger
from cuhk_project.detection.dataset import YOLOMFDataset
from cuhk_project.detection.model import SimpleDetectionModel
from cuhk_project.detection.trainer import DetectionTrainer
import torch

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
    parser.add_argument("--num-anchors", type=int, default=3,
                        help="Number of anchor boxes for YOLO-style training")
    parser.add_argument("--grid-size", type=str, default="6x32",
                        help="Grid size for YOLO-style training (format: height x width)")
    parser.add_argument("--save-path", type=str, default="models/detection_model.pth",
                        help="Path to save trained model")
    
    args = parser.parse_args()
    
    # 解析网格尺寸 (height x width)
    grid_height, grid_width = map(int, args.grid_size.split('x'))
    grid_size = (grid_height, grid_width)  # (height, width)
    
    # 固定输入图像尺寸为96x512 (height, width)
    image_size = (96, 512)  # (height, width)
    
    logger.info(f"Training configuration: "
                f"data_dir={args.data_dir}, batch_size={args.batch_size}, "
                f"epochs={args.epochs}, lr={args.lr}, out_channels={args.out_channels}, "
                f"num_anchors={args.num_anchors}, grid_size={grid_size}, image_size={image_size}")

    # 创建数据集
    try:
        train_dataset = YOLOMFDataset(
            base_dir=args.data_dir, 
            split="train", 
            target_size=image_size,  # (height, width)
            grid_size=grid_size      # (height, width)
        )
        val_dataset = YOLOMFDataset(
            base_dir=args.data_dir, 
            split="val", 
            target_size=image_size,  # (height, width)
            grid_size=grid_size      # (height, width)
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")

        # 初始化DataLoader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=YOLOMFDataset.collate_fn,
            num_workers=0
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=YOLOMFDataset.collate_fn,
            num_workers=0
        )
    except Exception as e:
        logger.error(f"Failed to create datasets: {str(e)}")
        return
    
    # 创建模型
    try:
        model = SimpleDetectionModel(
            in_channels=3, 
            out_channels=args.out_channels,
            kernel_size=3,
            num_anchors=args.num_anchors,
            grid_size=grid_size  # 传递网格尺寸给模型
        )
        logger.info(f"Model created successfully with {args.num_anchors} anchors")
        logger.info(f"Model output grid size: {grid_size}")
    except Exception as e:
        logger.error(f"Failed to create model: {str(e)}")
        return
    
    # 创建训练器 - 传递数据集而不是数据加载器
    try:
        trainer = DetectionTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_loader=train_loader,  # 传入train_loader
            val_loader=val_loader,      # 传递验证数据集
            grid_size=grid_size,
            num_anchors=args.num_anchors,
            batch_size=args.batch_size,    # 传递批次大小
            learning_rate=args.lr,
            num_epochs=args.epochs
        )
        logger.info("Trainer initialized successfully with grid-based training")
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
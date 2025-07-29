import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cuhk_project.detection.dataset import YOLOMFDataset
from cuhk_project.detection.model import SimpleDetectionModel
from pathlib import Path

def load_model(model_path, out_channels=16):
    """加载训练好的模型"""
    model = SimpleDetectionModel(out_channels=out_channels)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def visualize_sample(image, pred_boxes, true_boxes=None):
    """可视化单个样本的预测结果"""
    # 转换图像格式 [C,H,W] -> [H,W,C]
    image = image.permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots(1, figsize=(12, 6))
    ax.imshow(image)
    
    # 绘制预测框（红色）
    for box in pred_boxes:
        cx, cy, w, h = box
        x = (cx - w/2) * image.shape[1]
        y = (cy - h/2) * image.shape[0]
        rect = patches.Rectangle(
            (x, y), w*image.shape[1], h*image.shape[0],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    
    # 绘制真实框（绿色）
    if true_boxes is not None:
        for box in true_boxes:
            cx, cy, w, h = box
            x = (cx - w/2) * image.shape[1]
            y = (cy - h/2) * image.shape[0]
            rect = patches.Rectangle(
                (x, y), w*image.shape[1], h*image.shape[0],
                linewidth=2, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)
    
    plt.title("Red: Predictions | Green: Ground Truth")
    plt.axis('off')
    plt.show()

def main():
    # 配置参数
    data_dir = "/Users/jhen/Documents/CUHK-Project/cuhk_project/data/yolo_mf_dataset"
    model_path = "/Users/jhen/Documents/CUHK-Project/cuhk_project/models/test_5.pth"
    out_channels = 16
    num_samples = 5  # 要验证的样本数量
    
    # 初始化
    device = torch.device('cpu')  # 使用CPU保证兼容性
    dataset = YOLOMFDataset(base_dir=data_dir, split='test', target_size=(512, 96))
    model = load_model(model_path, out_channels).to(device)
    
    print(f"Loaded model from {model_path}")
    print(f"Testing on {len(dataset)} samples")
    
    # 验证循环
    for i in range(min(num_samples, len(dataset))):
        image, target = dataset[i]
        image_tensor = image.unsqueeze(0).to(device)  # 添加batch维度
        
        # 运行预测 (逐步调整阈值)
        with torch.no_grad():
            pred = model.predict(image_tensor, conf_thresh=0.4, iou_thresh=0.4)  # 降低置信度阈值
        
        # 获取真实标注
        true_boxes = target['boxes'].numpy() if 'boxes' in target else None
        
        print(f"\nSample {i}:")
        print(f"Predicted boxes: {len(pred['boxes'])}")
        print(f"True boxes: {len(true_boxes) if true_boxes is not None else 0}")
        
        # 可视化
        visualize_sample(image, pred['boxes'].cpu().numpy(), true_boxes)
        
        # 输出原始预测值用于调试
        print("原始预测值 (cx, cy, w, h):")
        for box in pred['boxes']:
            print(f"  {box.tolist()}")
        print(f"置信度分数: {pred['scores'].tolist()}")

if __name__ == "__main__":
    main()

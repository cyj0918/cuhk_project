import torch
import numpy as np
import matplotlib.pyplot as plt
from cuhk_project.detection.model import SimpleDetectionModel
from cuhk_project.detection.dataset import YOLOMFDataset

def inspect_wide_outputs(model_path, data_dir, sample_index=0):
    # 初始化模型 (匹配512x96输入)
    model = SimpleDetectionModel(
        out_channels=16,
        num_anchors=3,
        grid_size=(6, 32)  # 96/16=6, 512/16=32
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 加载数据集 (确保尺寸匹配)
    dataset = YOLOMFDataset(
        base_dir=data_dir,
        split='test',
        target_size=(96, 512),  # 高96，宽512
        grid_size=(6, 32)
    )
    
    # 获取样本并添加batch维度
    image, _ = dataset[sample_index]
    image_tensor = image.unsqueeze(0)

    # 原始前向传播
    with torch.no_grad():
        bbox_pred, obj_pred = model(image_tensor)
    
    print("=== 宽屏图像诊断输出 ===")
    print(f"输入图像尺寸: {image.shape}")
    print(f"bbox_pred形状: {bbox_pred.shape} (anchors×4×H×W)")
    print(f"obj_pred形状: {obj_pred.shape} (anchors×H×W)")
    
    # 可视化宽屏特征图
    plt.figure(figsize=(20, 6))
    
    # 显示原始图像
    plt.subplot(2, 1, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title("Original Image (512x96)")
    
    # 显示第一个锚点的dx特征
    plt.subplot(2, 1, 2)
    dx_map = bbox_pred[0, 0, 0].detach().numpy()  # 第一个anchor的dx通道
    plt.imshow(dx_map, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("dx Prediction (Grid 32x6)")
    
    plt.tight_layout()
    plt.savefig('wide_outputs.png', bbox_inches='tight')
    plt.close()
    
    # 打印关键数值统计
    print("\n=== 数值统计 ===")
    print(f"dx范围: [{dx_map.min():.3f}, {dx_map.max():.3f}]")
    print(f"obj置信度范围: [{obj_pred.min():.3f}, {obj_pred.max():.3f}]")
    
    # 保存原始输出为npy文件
    np.save('bbox_pred.npy', bbox_pred.detach().numpy())
    np.save('obj_pred.npy', obj_pred.detach().numpy())
    print("\n原始输出已保存为 bbox_pred.npy 和 obj_pred.npy")

if __name__ == '__main__':
    inspect_wide_outputs(
        model_path='models/test_5.pth',
        data_dir='data/yolo_mf_dataset'
    )

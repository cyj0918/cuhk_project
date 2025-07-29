import os
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

def calculate_optimal_anchors(data_dir, num_anchors=3):
    """
    基於數據集統計計算最優錨框尺寸
    
    參數:
        data_dir: 數據集根目錄
        num_anchors: 需要的錨框數量
    """
    # 收集所有邊界框的寬高
    bbox_sizes = []
    
    # 遍歷所有標籤文件
    labels_dir = Path(data_dir) / "labels"
    for split in ["train", "val"]:
        split_dir = labels_dir / split
        for label_file in split_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:  # 格式: class_id cx cy w h
                        _, _, _, w, h = map(float, parts)
                        bbox_sizes.append([w, h])
    
    if not bbox_sizes:
        print("錯誤: 未找到任何邊界框數據")
        return None
    
    # 轉換為numpy數組
    bbox_array = np.array(bbox_sizes)
    
    # 使用K-means聚類尋找最優錨框尺寸
    kmeans = KMeans(n_clusters=num_anchors, random_state=0)
    kmeans.fit(bbox_array)
    
    # 獲取聚類中心並排序
    anchors = kmeans.cluster_centers_
    anchors = sorted(anchors, key=lambda x: x[0]*x[1])  # 按面積排序
    
    print(f"基於 {len(bbox_sizes)} 個邊界框計算的錨框尺寸:")
    for i, (w, h) in enumerate(anchors):
        print(f"錨框 {i+1}: 寬={w:.4f}, 高={h:.4f}, 面積={w*h:.4f}")
    
    return anchors

if __name__ == "__main__":
    # 配置參數
    data_dir = "/Users/jhen/Documents/CUHK-Project/dataset/yolo_worker_training"
    num_anchors = 3
    
    optimal_anchors = calculate_optimal_anchors(data_dir, num_anchors)
    
    # 保存結果到文件
    if optimal_anchors is not None:
        output_file = Path(data_dir) / "optimal_anchors.txt"
        with open(output_file, 'w') as f:
            for w, h in optimal_anchors:
                f.write(f"{w:.6f} {h:.6f}\n")
        print(f"錨框尺寸已保存至: {output_file}")

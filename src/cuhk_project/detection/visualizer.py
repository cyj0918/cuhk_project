import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

class DetectionVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_sample(self, image, pred_boxes, true_boxes, filename):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(image.permute(1, 2, 0).numpy())
        
        # 绘制预测框（红色）
        for box in pred_boxes:
            cx, cy, w, h = box
            x = (cx - w/2) * image.shape[2]
            y = (cy - h/2) * image.shape[1]
            rect = Rectangle(
                (x, y), w*image.shape[2], h*image.shape[1],
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
        
        # 绘制真实框（绿色）
        for box in true_boxes:
            cx, cy, w, h = box
            x = (cx - w/2) * image.shape[2]
            y = (cy - h/2) * image.shape[1]
            rect = Rectangle(
                (x, y), w*image.shape[2], h*image.shape[1],
                linewidth=1, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)
        
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
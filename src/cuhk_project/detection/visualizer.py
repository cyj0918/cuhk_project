import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Tuple
import warnings
from cuhk_project.utils.logger import logger

class DetectionVisualizer:
    """目標檢測結果可視化器 - 適配新數據集格式和訓練架構"""
    
    def __init__(self, output_dir: str):
        """
        初始化可視化器
        
        參數:
            output_dir: 輸出目錄路徑
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設置matplotlib參數
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.dpi'] = 150
        
        # 定義顏色映射
        self.colors = {
            'prediction': '#FF0000',  # 紅色
            'ground_truth': '#00FF00',  # 綠色
            'matched': '#0000FF',  # 藍色
            'text_bg': '#FFFFFF'  # 白色背景
        }
        
        logger.info(f"DetectionVisualizer initialized with output directory: {self.output_dir}")

    def visualize_sample(self, 
                        image: Union[np.ndarray, torch.Tensor], 
                        pred_boxes: np.ndarray, 
                        true_boxes: np.ndarray, 
                        filename: str,
                        pred_scores: Optional[np.ndarray] = None,
                        conf_thresh: float = 0.0,
                        show_confidence: bool = True,
                        show_legend: bool = True,
                        title: Optional[str] = None) -> None:
        """
        可視化檢測樣本
        
        參數:
            image: 輸入圖像 (HWC numpy array 或 CHW torch tensor)
            pred_boxes: 預測邊界框 [N, 4] (cxcywh格式，歸一化)
            true_boxes: 真實邊界框 [M, 4] (cxcywh格式，歸一化)
            filename: 保存文件名
            pred_scores: 預測置信度分數 [N,]
            conf_thresh: 置信度閾值，用於過濾顯示
            show_confidence: 是否顯示置信度分數
            show_legend: 是否顯示圖例
            title: 圖像標題
        """
        try:
            # 1. 圖像格式標準化
            image_np = self._standardize_image(image)
            h, w = image_np.shape[:2]
            
            # 2. 創建圖形
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(image_np)
            
            # 3. 設置標題
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold')
            else:
                ax.set_title(f"Detection Results - {filename}", fontsize=12)
            
            # 4. 繪製真實邊界框
            if len(true_boxes) > 0:
                self._draw_boxes(ax, true_boxes, w, h, 
                               color=self.colors['ground_truth'],
                               label='Ground Truth',
                               linewidth=2)
            
            # 5. 繪製預測邊界框
            if len(pred_boxes) > 0:
                self._draw_prediction_boxes(ax, pred_boxes, pred_scores, w, h,
                                          conf_thresh, show_confidence)
            
            # 6. 添加統計信息
            self._add_statistics(ax, pred_boxes, true_boxes, pred_scores, conf_thresh)
            
            # 7. 添加圖例
            if show_legend:
                self._add_legend(ax)
            
            # 8. 設置坐標軸
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)  # 翻轉y軸以匹配圖像坐標
            ax.set_xlabel('Width (pixels)', fontsize=10)
            ax.set_ylabel('Height (pixels)', fontsize=10)
            
            # 9. 保存圖像
            output_path = self.output_dir / filename
            plt.savefig(output_path, bbox_inches='tight', dpi=150, 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            logger.debug(f"Visualization saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing sample {filename}: {str(e)}")
            plt.close('all')  # 確保清理資源
            
    def _standardize_image(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """標準化圖像格式為HWC numpy array"""
        # 處理Torch Tensor
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # BCHW
                image = image.squeeze(0)  # 移除batch維度
            
            if image.dim() == 3 and image.shape[0] == 3:  # CHW
                image = image.permute(1, 2, 0)  # 轉換為HWC
                
            image = image.cpu().numpy()
        
        # 確保numpy array
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # 處理數值範圍
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = np.clip(image / 255.0, 0, 1)
        else:
            image = np.clip(image, 0, 1)
        
        # 確保HWC格式
        if image.ndim == 2:  # 灰度圖
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {image.shape}")
        
        return image

    def _draw_boxes(self, ax, boxes: np.ndarray, w: int, h: int, 
                   color: str, label: str, linewidth: int = 2,
                   linestyle: str = '-') -> None:
        """繪製邊界框"""
        if len(boxes) == 0:
            return
            
        # 處理維度
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        
        for i, box in enumerate(boxes):
            try:
                # 從cxcywh格式轉換為像素坐標
                cx, cy, box_w, box_h = box[:4]
                
                # 轉換為左上角坐標和尺寸
                x = (cx - box_w/2) * w
                y = (cy - box_h/2) * h
                rect_w = box_w * w
                rect_h = box_h * h
                
                # 邊界檢查
                x = max(0, min(x, w))
                y = max(0, min(y, h))
                rect_w = max(1, min(rect_w, w - x))
                rect_h = max(1, min(rect_h, h - y))
                
                # 創建矩形
                rect = Rectangle(
                    (x, y), rect_w, rect_h,
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor='none',
                    linestyle=linestyle,
                    label=label if i == 0 else ""  # 只為第一個框添加標籤
                )
                ax.add_patch(rect)
                
            except Exception as e:
                logger.warning(f"Error drawing box {i}: {str(e)}")
                continue

    def _draw_prediction_boxes(self, ax, pred_boxes: np.ndarray, pred_scores: Optional[np.ndarray],
                             w: int, h: int, conf_thresh: float, show_confidence: bool) -> None:
        """繪製預測邊界框"""
        if len(pred_boxes) == 0:
            return
        
        # 過濾低置信度預測
        if pred_scores is not None and conf_thresh > 0:
            valid_indices = pred_scores >= conf_thresh
            pred_boxes = pred_boxes[valid_indices]
            pred_scores = pred_scores[valid_indices]
        
        if len(pred_boxes) == 0:
            return
        
        # 繪製預測框
        for i, box in enumerate(pred_boxes):
            try:
                # 從cxcywh格式轉換為像素坐標
                cx, cy, box_w, box_h = box[:4]
                
                x = (cx - box_w/2) * w
                y = (cy - box_h/2) * h
                rect_w = box_w * w
                rect_h = box_h * h
                
                # 邊界檢查
                x = max(0, min(x, w))
                y = max(0, min(y, h))
                rect_w = max(1, min(rect_w, w - x))
                rect_h = max(1, min(rect_h, h - y))
                
                # 根據置信度設置透明度
                alpha = 1.0
                if pred_scores is not None:
                    alpha = max(0.3, min(1.0, pred_scores[i]))
                
                # 創建矩形
                rect = Rectangle(
                    (x, y), rect_w, rect_h,
                    linewidth=2,
                    edgecolor=self.colors['prediction'],
                    facecolor='none',
                    alpha=alpha,
                    label='Prediction' if i == 0 else ""
                )
                ax.add_patch(rect)
                
                # 添加置信度標籤
                if show_confidence and pred_scores is not None:
                    confidence_text = f"{pred_scores[i]:.2f}"
                    ax.text(x, y - 5, confidence_text,
                           fontsize=9, color=self.colors['prediction'],
                           bbox=dict(boxstyle="round,pad=0.2",
                                   facecolor=self.colors['text_bg'],
                                   alpha=0.8, edgecolor='none'),
                           ha='left', va='bottom')
                
            except Exception as e:
                logger.warning(f"Error drawing prediction box {i}: {str(e)}")
                continue

    def _add_statistics(self, ax, pred_boxes: np.ndarray, true_boxes: np.ndarray,
                       pred_scores: Optional[np.ndarray], conf_thresh: float) -> None:
        """添加統計信息"""
        try:
            num_gt = len(true_boxes)
            num_pred = len(pred_boxes)
            
            # 過濾低置信度預測進行統計
            if pred_scores is not None and conf_thresh > 0:
                num_pred_filtered = np.sum(pred_scores >= conf_thresh)
            else:
                num_pred_filtered = num_pred
            
            stats_text = f"GT: {num_gt} | Pred: {num_pred_filtered}"
            if conf_thresh > 0:
                stats_text += f" (conf≥{conf_thresh:.2f})"
            
            # 添加平均置信度
            if pred_scores is not None and len(pred_scores) > 0:
                avg_conf = np.mean(pred_scores)
                stats_text += f" | Avg Conf: {avg_conf:.3f}"
            
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.5",
                           facecolor=self.colors['text_bg'],
                           alpha=0.9, edgecolor='gray'),
                   verticalalignment='top')
                   
        except Exception as e:
            logger.warning(f"Error adding statistics: {str(e)}")

    def _add_legend(self, ax) -> None:
        """添加圖例"""
        try:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                # 去重圖例項目
                by_label = dict(zip(labels, handles))
                legend = ax.legend(by_label.values(), by_label.keys(),
                                 loc='upper right', fontsize=10,
                                 framealpha=0.9, fancybox=True, shadow=True)
                legend.get_frame().set_facecolor(self.colors['text_bg'])
        except Exception as e:
            logger.warning(f"Error adding legend: {str(e)}")

    def visualize_batch(self, 
                       images: Union[List[np.ndarray], torch.Tensor],
                       pred_boxes_list: List[np.ndarray],
                       true_boxes_list: List[np.ndarray],
                       batch_filename: str,
                       pred_scores_list: Optional[List[np.ndarray]] = None,
                       conf_thresh: float = 0.0,
                       max_samples: int = 16) -> None:
        """
        可視化批次結果
        
        參數:
            images: 批次圖像
            pred_boxes_list: 批次預測框列表
            true_boxes_list: 批次真實框列表
            batch_filename: 批次文件名
            pred_scores_list: 批次置信度列表
            conf_thresh: 置信度閾值
            max_samples: 最大顯示樣本數
        """
        try:
            # 處理輸入數據
            if isinstance(images, torch.Tensor):
                if images.dim() == 4:  # BCHW
                    images = [images[i] for i in range(min(len(images), max_samples))]
                else:
                    images = [images]
            
            num_samples = min(len(images), max_samples)
            
            # 計算網格佈局
            cols = min(4, num_samples)
            rows = (num_samples + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if num_samples == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i in range(num_samples):
                ax = axes[i]
                
                # 獲取當前樣本數據
                image = images[i]
                pred_boxes = pred_boxes_list[i] if i < len(pred_boxes_list) else np.empty((0, 4))
                true_boxes = true_boxes_list[i] if i < len(true_boxes_list) else np.empty((0, 4))
                pred_scores = pred_scores_list[i] if pred_scores_list and i < len(pred_scores_list) else None
                
                # 標準化圖像
                image_np = self._standardize_image(image)
                h, w = image_np.shape[:2]
                
                # 顯示圖像
                ax.imshow(image_np)
                ax.set_title(f"Sample {i+1}", fontsize=10)
                
                # 繪製框
                if len(true_boxes) > 0:
                    self._draw_boxes(ax, true_boxes, w, h, 
                                   self.colors['ground_truth'], 'GT', linewidth=1)
                
                if len(pred_boxes) > 0:
                    self._draw_prediction_boxes(ax, pred_boxes, pred_scores, w, h,
                                              conf_thresh, show_confidence=False)
                
                # 設置坐標軸
                ax.set_xticks([])
                ax.set_yticks([])
            
            # 隱藏多餘的子圖
            for i in range(num_samples, len(axes)):
                axes[i].set_visible(False)
            
            # 添加總標題
            fig.suptitle(f"Batch Visualization - {batch_filename}", fontsize=14, fontweight='bold')
            
            # 保存
            output_path = self.output_dir / batch_filename
            plt.savefig(output_path, bbox_inches='tight', dpi=150,
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            logger.info(f"Batch visualization saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing batch {batch_filename}: {str(e)}")
            plt.close('all')

    def create_comparison_plot(self, 
                              image: Union[np.ndarray, torch.Tensor],
                              results_dict: dict,
                              filename: str,
                              title: str = "Model Comparison") -> None:
        """
        創建多模型比較圖
        
        參數:
            image: 輸入圖像
            results_dict: 結果字典 {'model_name': {'boxes': boxes, 'scores': scores}}
            filename: 文件名
            title: 圖表標題
        """
        try:
            image_np = self._standardize_image(image)
            h, w = image_np.shape[:2]
            
            num_models = len(results_dict)
            cols = min(3, num_models)
            rows = (num_models + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
            if num_models == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, (model_name, results) in enumerate(results_dict.items()):
                ax = axes[i]
                ax.imshow(image_np)
                ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
                
                pred_boxes = results.get('boxes', np.empty((0, 4)))
                pred_scores = results.get('scores', None)
                
                if len(pred_boxes) > 0:
                    self._draw_prediction_boxes(ax, pred_boxes, pred_scores, w, h,
                                              conf_thresh=0.0, show_confidence=True)
                
                ax.set_xticks([])
                ax.set_yticks([])
            
            # 隱藏多餘的子圖
            for i in range(num_models, len(axes)):
                axes[i].set_visible(False)
            
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            output_path = self.output_dir / filename
            plt.savefig(output_path, bbox_inches='tight', dpi=150,
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            logger.info(f"Comparison plot saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating comparison plot {filename}: {str(e)}")
            plt.close('all')

    def get_output_dir(self) -> Path:
        """獲取輸出目錄路徑"""
        return self.output_dir

    def clear_output_dir(self) -> None:
        """清空輸出目錄"""
        try:
            import shutil
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cleared output directory: {self.output_dir}")
        except Exception as e:
            logger.error(f"Error clearing output directory: {str(e)}")
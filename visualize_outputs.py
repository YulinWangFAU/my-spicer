# -*- coding: utf-8 -*-
"""
Created on 2025/7/20 14:22

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

# 设置路径
metrics_path = './outputs/metrics.csv'
image_dir = './outputs/images'
save_dir = './outputs/figures'
os.makedirs(save_dir, exist_ok=True)

# 加载指标数据
df = pd.read_csv(metrics_path)

# ✅ 1. 绘制每个 slice 的 PSNR、SSIM、NMSE 柱状图
def plot_metrics(df):
    for metric in ['PSNR', 'SSIM', 'NMSE']:
        plt.figure(figsize=(10, 4))
        plt.bar(df['Index'], df[metric], color='steelblue')
        plt.xlabel('Slice Index')
        plt.ylabel(metric)
        plt.title(f'{metric} per Slice')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{metric}_barplot.png', dpi=300)
        plt.close()

# ✅ 2. 可视化 recon vs target vs diff 图像对比
def plot_image_triplets(image_dir, indices, cmap='gray', diff_cmap='hot'):
    for i in indices:
        recon_path = os.path.join(image_dir, f'recon_{i}.png')
        target_path = os.path.join(image_dir, f'target_{i}.png')
        diff_path = os.path.join(image_dir, f'diff_{i}.png')

        if not (os.path.exists(recon_path) and os.path.exists(target_path) and os.path.exists(diff_path)):
            continue

        recon = np.array(Image.open(recon_path))
        target = np.array(Image.open(target_path))
        diff = np.array(Image.open(diff_path))

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(recon, cmap=cmap)
        axs[0].set_title('Reconstructed')
        axs[1].imshow(target, cmap=cmap)
        axs[1].set_title('Target')
        axs[2].imshow(diff, cmap=diff_cmap)
        axs[2].set_title('Difference')

        for ax in axs:
            ax.axis('off')

        plt.suptitle(f'Slice {i}', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(f'{save_dir}/compare_slice_{i}.png', dpi=300)
        plt.close()

# ✅ 执行主流程
if __name__ == '__main__':
    print("✅ Start visualization...")

    # 保存指标图
    plot_metrics(df)

    # 保存图像对比（默认前 5 个 slice）
    plot_image_triplets(image_dir, indices=df['Index'].values[:5])

    print(f"✅ Done! All figures saved to: {save_dir}")

# -*- coding: utf-8 -*-
"""
Created on 2025/9/29 19:18

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ==== 配置路径 ====
base_dir = "/Users/wangyulin/best_recon_test"
acc_factors = {"R2": 2, "R4": 4, "R6": 6, "R8": 8, "R10": 10}

# 存储结果
results = {}

# 遍历每个加速因子
for key, value in acc_factors.items():
    folder = [f for f in os.listdir(base_dir) if f"_{key}_" in f][0]
    csv_path = os.path.join(base_dir, folder, "metrics_overall.csv")

    df = pd.read_csv(csv_path)
    results[value] = df.iloc[0].to_dict()  # 用数字作索引

# 转成 DataFrame，索引是 [2,4,6,8,10]
metrics_df = pd.DataFrame(results).T.sort_index()
metrics_df.index.name = "Acceleration"

print(metrics_df)


# ==== 画图并保存 ====
def plot_metric(metric_name, ylabel):
    plt.figure(figsize=(6, 4))
    plt.plot(metrics_df.index, metrics_df[f"{metric_name}_zf_mean"], marker="o", label="Zero-filled")
    plt.plot(metrics_df.index, metrics_df[f"{metric_name}_trained_mean"], marker="o", label="Trained")
    plt.plot(metrics_df.index, metrics_df[f"{metric_name}_init_mean"], marker="o", label="Init")

    plt.xlabel("Acceleration Factor (R)")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name.upper()} vs Acceleration Factor")
    plt.xticks([2, 4, 6, 8, 10])  # 明确横坐标
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # === 保存为 PNG ===
    plt.savefig(f"{metric_name}_vs_acceleration.png", dpi=300)
    plt.close()  # 关闭，避免图像堆积


# 分别画三张图并保存
plot_metric("psnr", "PSNR (dB)")
plot_metric("ssim", "SSIM")
plot_metric("nmse", "NMSE")

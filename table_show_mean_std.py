# -*- coding: utf-8 -*-
"""
Created on 2025/9/30 12:38

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import pandas as pd
import glob
import os

# 假设所有 R 的 metrics_subjects.csv 都在各自的文件夹里
base_dir = "/Users/wangyulin/best_recon_test"
results = []

# 遍历 R2, R4, R6, R8, R10 文件夹
for folder in sorted(glob.glob(os.path.join(base_dir, "spicer_test_R*"))):
    accel = folder.split("_R")[-1].split("_")[0]  # 提取加速因子，例如 R2 -> 2
    csv_path = os.path.join(folder, "metrics_subjects.csv")
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)

    # 针对每个方法统计 mean ± std (跨 subject)
    for method in df['method'].unique():
        df_method = df[df['method'] == method]

        psnr_mean = df_method['psnr_mean'].mean()
        psnr_std = df_method['psnr_mean'].std()

        ssim_mean = df_method['ssim_mean'].mean()
        ssim_std = df_method['ssim_mean'].std()

        nmse_mean = df_method['nmse_mean'].mean()
        nmse_std = df_method['nmse_mean'].std()

        results.append({
            "Accel": f"R{accel}",
            "Method": method,
            "PSNR": f"{psnr_mean:.2f} ± {psnr_std:.2f}",
            "SSIM": f"{ssim_mean:.3f} ± {ssim_std:.3f}",
            "NMSE": f"{nmse_mean:.4f} ± {nmse_std:.4f}"
        })

# 转成 DataFrame 方便导出
summary_df = pd.DataFrame(results)

# 打印结果
print(summary_df)

# 保存成 CSV
summary_df.to_csv(os.path.join(base_dir, "metrics_summary.csv"), index=False)

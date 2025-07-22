# -*- coding: utf-8 -*-
"""
Created on 2025/7/18 15:07

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime

from networks.SPICER_fastmri_network import SPNet
from dataset.pmri_fastmri_brain import RealMeasurement
from dataset.pmri_fastmri_brain import fmult, ftran
from utils.util import *
from utils.measures import *

# ==== 设备设置 ====
local_rank = int(os.environ.get("SLURM_PROCID", 0))
device = f'cuda:{local_rank % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu'
print(f"🧠 Using device: {device}")

# ==== 验证函数 ====
def val(model, valloader, save_dir="recon_results"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    psnr_list, ssim_list, nmse_list = [], [], []

    for iteration, samples in enumerate(tqdm(valloader, desc="Evaluating")):
        x_hat, smps_hat, y, mask_m, mask_n = samples
        mask_m = mask_m.byte().to(device)
        y = y.to(device)
        y_m = y * mask_m

        ny = y_m.shape[-2]
        ACS_size = ((ny // 2) - (int(ny * 0.2 * (2 / 8)) // 2)) * 2
        output_m, smap_m = model(torch.view_as_real(y_m), mask_m, ACS_center=(ny // 2), ACS_size=ACS_size)

        # Coil 合并
        brain_mask = torch.view_as_real(smps_hat[0]).to(device)[0]
        brain_mask[brain_mask != 0] = 1
        output_m = complex_mul(output_m, complex_conj(brain_mask)).sum(dim=0, keepdim=True)

        output_show = output_m.cpu().detach()
        output_show = complex_abs(output_show)[0]
        target_show = torch.abs(x_hat.squeeze())

        output_show = normlize(center_crop(output_show, [400, 320])).squeeze()
        target_show = normlize(center_crop(target_show, [400, 320])).squeeze()
        error_map = np.abs(output_show.numpy() - target_show.numpy())
        print(f"[Debug] output_show shape: {output_show.shape}")
        print(f"[Debug] target_show shape: {target_show.shape}")
        # === 保存图像 ===
        plt.imsave(f"{save_dir}/recon_{iteration:03d}.png", output_show.numpy(), cmap='gray')
        plt.imsave(f"{save_dir}/gt_{iteration:03d}.png", target_show.numpy(), cmap='gray')
        plt.imsave(f"{save_dir}/error_{iteration:03d}.png", error_map, cmap='hot')

        # === 指标计算 ===
        psnr_list.append(compare_psnr(output_show, target_show))
        ssim_list.append(compare_ssim(output_show.unsqueeze(0), target_show.unsqueeze(0).unsqueeze(0)))
        nmse_list.append(compare_nmse(output_show, target_show))

    print(f"✅ Val PSNR: {np.mean(psnr_list):.4f}")
    print(f"✅ Val SSIM: {np.mean(ssim_list):.4f}")
    print(f"✅ Val NMSE: {np.mean(nmse_list):.4f}")

# ==== 主函数入口 ====
if __name__ == "__main__":
    acceleration_factor = 8
    model_path = "/home/vault/iwi5/iwi5325h/tmp_1113439/spicer_out/SPICER_fastmri_18-Jul-2025-15-18-45/N2N_050.pth"

    # === 验证集 ===
    val_dataset = RealMeasurement(
        idx_list=range(80, 90),
        acceleration_rate=acceleration_factor,
        is_return_y_smps_hat=True,
        mask_pattern='uniformly_cartesian',
        smps_hat_method='eps',
    )
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # === 模型加载 ===
    model = SPNet(num_cascades=6, pools=4, chans=18, sens_pools=4, sens_chans=8).to(device)
    print(f"🔍 Loading model: {model_path}")
    test_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(test_state_dict)

    # === 验证并保存图像 ===
    with torch.no_grad():
        val(model, valloader, save_dir="recon_results_n2n_050")

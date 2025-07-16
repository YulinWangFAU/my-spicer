# evaluate_spicer_model.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.pmri_fastmri_brain import RealMeasurement
from networks.SPICER_fastmri_network import SPNet
from utils.util import complex_abs, normlize
from utils.measures import compare_psnr, compare_ssim

# ==== 设置路径 ====
output_dir = "/home/rlvl/rlvl144v/spicer_outputs/SPICER_fastmri_<timestamp>"  # 替换为实际路径
checkpoint_path = os.path.join(output_dir, "best_model.pth")

# ==== 加载模型 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SPNet(num_cascades=6, pools=4, chans=18, sens_pools=4, sens_chans=8)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# ==== 验证数据 ====
val_dataset = RealMeasurement(
    idx_list=range(1355, 1377),
    acceleration_rate=8,
    is_return_y_smps_hat=True,
    mask_pattern='uniformly_cartesian',
    smps_hat_method='eps',
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

psnr_list = []
ssim_list = []

for i, (x_hat, _, y, _, mask) in enumerate(tqdm(val_loader, desc="Evaluating")):
    y, mask = y.to(device), mask.to(device)
    ny = y.shape[-2]
    ACS_size = ((ny // 2) - (int(ny * 0.2 * (2 / 8)) // 2)) * 2

    with torch.no_grad():
        output, _ = model(torch.view_as_real(y), mask, ACS_center=ny // 2, ACS_size=ACS_size)

    recon = normlize(complex_abs(output.squeeze().cpu()))
    target = normlize(torch.abs(x_hat.squeeze()))

    psnr_list.append(compare_psnr(recon, target))
    ssim_list.append(compare_ssim(recon[None, None], target[None, None]))

    # === 可视化前几张重建图像 ===
    if i < 5:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(target.numpy(), cmap="gray")
        plt.title("Target")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(recon.numpy(), cmap="gray")
        plt.title(f"Reconstruction\nPSNR={psnr_list[-1]:.2f}, SSIM={ssim_list[-1]:.3f}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"recon_{i:02d}.png"))
        plt.close()

# ==== 打印最终平均指标 ====
print("\n✅ Evaluation Completed.")
print(f"📈 Mean PSNR: {np.mean(psnr_list):.2f} dB")
print(f"📈 Mean SSIM: {np.mean(ssim_list):.4f}")

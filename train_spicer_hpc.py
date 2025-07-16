# -*- coding: utf-8 -*-
"""
Created on 2025/7/16 16:18

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import shutil

# ==== 项目模块 ====
from dataset.pmri_fastmri_brain import RealMeasurement, fmult
from networks.SPICER_fastmri_network import SPNet
from utils.util import complex_abs, normlize
from utils.loss_functions import gradient_loss
from utils.measures import compare_psnr, compare_ssim
from utils.early_stopping import EarlyStopping  # 自定义模块

# ==== 环境变量配置 ====
user = os.environ.get("USER", "unknown_user")
TMPDIR = os.environ.get("TMPDIR", f"/tmp/{user}")
HOME = os.path.expanduser("~")
os.environ['CUPY_CACHE_DIR'] = os.path.join(TMPDIR, "cupy")
os.environ['NUMBA_CACHE_DIR'] = os.path.join(TMPDIR, "numba")
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# ==== 设备选择（支持多 GPU） ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()

# ==== 日志与模型保存路径 ====
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_name = "SPICER_fastmri"
output_dir = os.path.join(HOME, "spicer_outputs", f"{model_name}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

# ==== 参数 ====
batch_size = 1
num_epochs = 200
acceleration_factor = 8
early_stop_patience = 15

# ==== 数据集 ====
train_dataset = RealMeasurement(
    idx_list=range(0, 15),
    acceleration_rate=acceleration_factor,
    is_return_y_smps_hat=True,
    mask_pattern='uniformly_cartesian',
    smps_hat_method='eps',
)
val_dataset = RealMeasurement(
    idx_list=range(15, 20),
    acceleration_rate=acceleration_factor,
    is_return_y_smps_hat=True,
    mask_pattern='uniformly_cartesian',
    smps_hat_method='eps',
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

# ==== 模型定义 ====
model = SPNet(num_cascades=6, pools=4, chans=18, sens_pools=4, sens_chans=8)
if n_gpus > 1:
    print(f"Using {n_gpus} GPUs with DataParallel")
    model = nn.DataParallel(model)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4)
early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True, path=os.path.join(output_dir, 'best_model.pth'))

# ==== 训练函数 ====
def train(epoch):
    model.train()
    train_psnr, train_ssim, train_loss = [], [], []

    for x_hat, smps_hat, y, mask_m, mask_n in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        y = y.to(device)
        mask_m, mask_n = mask_m.to(device), mask_n.to(device)

        y_m, y_n = y * mask_m, y * mask_n
        ny = y_m.shape[-2]
        ACS_size = ((ny // 2) - (int(ny * 0.2 * (2 / acceleration_factor)) // 2)) * 2

        output_m, smap_m = model(torch.view_as_real(y_m), mask_m, ACS_center=ny // 2, ACS_size=ACS_size)
        output_n, smap_n = model(torch.view_as_real(y_n), mask_n, ACS_center=ny // 2, ACS_size=ACS_size)

        smap_m_1 = torch.view_as_complex(smap_m.squeeze())
        smap_n_1 = torch.view_as_complex(smap_n.squeeze())

        h_output_m = fmult(torch.view_as_complex(output_n), smap_n_1, mask_m)
        h_output_n = fmult(torch.view_as_complex(output_m), smap_m_1, mask_n)

        smap_m_loss = smap_m.squeeze().permute(0, 3, 1, 2)
        loss = ((nn.functional.mse_loss(torch.view_as_real(h_output_m).float(), torch.view_as_real(y_m).float()) +
                 nn.functional.mse_loss(torch.view_as_real(h_output_n).float(), torch.view_as_real(y_n).float())) / 2
                + 0.001 * gradient_loss(smap_m_loss)).float()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_img = normlize(complex_abs(output_m.cpu().detach().squeeze()))
        gt_img = normlize(torch.abs(x_hat.squeeze()))
        train_psnr.append(compare_psnr(output_img, gt_img))
        train_ssim.append(compare_ssim(output_img[None, None], gt_img[None, None]))
        train_loss.append(loss.item())

    writer.add_scalars("Train", {"PSNR": np.mean(train_psnr), "SSIM": np.mean(train_ssim), "Loss": np.mean(train_loss)}, epoch)

# ==== 验证函数 ====
def validate(epoch):
    model.eval()
    val_psnr, val_ssim = [], []

    with torch.no_grad():
        for idx, (x_hat, _, y, _, mask) in enumerate(val_loader):
            y, mask = y.to(device), mask.to(device)
            ny = y.shape[-2]
            ACS_size = ((ny // 2) - (int(ny * 0.2 * (2 / acceleration_factor)) // 2)) * 2

            output, _ = model(torch.view_as_real(y), mask, ACS_center=ny // 2, ACS_size=ACS_size)
            output_img = normlize(complex_abs(output.cpu().squeeze()))
            gt_img = normlize(torch.abs(x_hat.squeeze()))

            val_psnr.append(compare_psnr(output_img, gt_img))
            val_ssim.append(compare_ssim(output_img[None, None], gt_img[None, None]))

            # 保存图像
            if epoch % 10 == 0 and idx < 3:
                img_stack = torch.stack([gt_img, output_img], dim=0)
                vutils.save_image(img_stack.unsqueeze(1), os.path.join(output_dir, f"val_epoch{epoch}_sample{idx}.png"))

    mean_psnr = np.mean(val_psnr)
    mean_ssim = np.mean(val_ssim)
    writer.add_scalars("Validation", {"PSNR": mean_psnr, "SSIM": mean_ssim}, epoch)
    return mean_psnr

# ==== 主循环 ====
for epoch in range(num_epochs):
    train(epoch)
    val_psnr = validate(epoch)
    early_stopping(-val_psnr, model)  # 注意：EarlyStopping 默认监控最小值，我们取负数使其监控最大 PSNR
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# ==== 结束后关闭 writer ====
writer.close()

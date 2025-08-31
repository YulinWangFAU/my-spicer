# -*- coding: utf-8 -*-
"""
SPICER training script (aligned with Hu et al., MRM 2024, Section 3.5)
@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import argparse
import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from networks.SPICER_fastmri_network import SPNet
from tqdm import tqdm
from datetime import datetime
from dataset.pmri_fastmri_brain_lunwen import RealMeasurement
from torch.utils.data import DataLoader
from utils.util import *
from utils.measures import *
from dataset.pmri_fastmri_brain import fmult
from utils.loss_functions import gradient_loss
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import shutil
import pandas as pd

# ========== HPC 环境设置 ==========
user = os.environ.get("USER", "unknown_user")
TMPDIR = os.environ.get("TMPDIR", f"/tmp/{user}")
HOME = os.path.expanduser("~")
os.environ['CUPY_CACHE_DIR'] = os.path.join(TMPDIR, "cupy")
os.environ['NUMBA_CACHE_DIR'] = os.path.join(TMPDIR, "numba")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# GPU 设置
local_rank = int(os.environ.get("SLURM_PROCID", 0))
device = f'cuda:{local_rank % torch.cuda.device_count()}'
print(f"🧠 Using device: {device}")

# 当前时间戳
now = datetime.now()
timestamp = now.strftime("%d-%b-%Y-%H-%M-%S")

# 模型保存路径
model_name = 'SPICER_fastmri'
save_root_tmp = os.path.join(TMPDIR, "spicer_out", f"{model_name}_{timestamp}")
save_root_final = os.path.join(HOME, "spicer_outputs", f"{model_name}_{timestamp}")
os.makedirs(save_root_tmp, exist_ok=True)
os.makedirs(save_root_final, exist_ok=True)
save_root = save_root_tmp

# TensorBoard 设置
log_dir = os.path.join(TMPDIR, "tensorboard_logs", timestamp)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# EarlyStopping 类
class EarlyStopping:
    def __init__(self, patience=20, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose
        self.best_epoch = None

    def __call__(self, val_score, epoch):
        if self.best_score is None:
            self.best_score = val_score
            self.best_epoch = epoch
        elif val_score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0

# ========== 论文超参 (3.5 Implementation) ==========
ACCELERATION = 8   # R = 8
EPOCHS = 200
LR_INIT = 1e-3
LR_DECAY = 1e-4
MILESTONES = [30]  # 前30 epoch 1e-3, 之后1e-4
LAMBDA_SMOOTH = 0.01

# 模型与优化器
model = SPNet(num_cascades=8, pools=4, chans=18, sens_pools=4, sens_chans=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)

# 日志变量
train_loss_log, val_loss_log = [], []
train_psnr_log, val_psnr_log = [], []
train_ssim_log, val_ssim_log = [], []

# 数据集划分
train_idx = range(564, 694)   # 130 subjects
val_idx   = range(694, 709)   # 15 subjects
test_idx  = range(709, 729)   # 20 subjects

dataset = RealMeasurement(idx_list=train_idx, acceleration_rate=ACCELERATION,
                          is_return_y_smps_hat=True,
                          mask_pattern='uniformly_cartesian',
                          smps_hat_method='eps')
trainloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

val_dataset = RealMeasurement(idx_list=val_idx, acceleration_rate=ACCELERATION,
                              is_return_y_smps_hat=True,
                              mask_pattern='uniformly_cartesian',
                              smps_hat_method='eps')
valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

# 训练函数
def train(epoch):
    model.train()
    psnrs, losses, ssims = [], [], []
    for iteration, samples in enumerate(tqdm(trainloader, desc=f"Train [{epoch:03d}]")):
        x_hat, smps_hat, y, mask_m, mask_n = samples
        x_hat = x_hat.to(device)
        mask_m = mask_m.byte().to(device)
        mask_n = mask_n.byte().to(device)
        y = y.to(device)
        y_m = y * mask_m
        y_n = y * mask_n

        # squeeze if needed
        y_m_input = y_m.squeeze(1) if y_m.shape[1] == 1 else y_m
        y_n_input = y_n.squeeze(1) if y_n.shape[1] == 1 else y_n

        ny = y_m.shape[-2]
        ACS_size = ((ny // 2) - (int(ny * 0.2 * (2 / ACCELERATION)) // 2)) * 2

        output_m, smap_m = model(torch.view_as_real(y_m_input), mask_m, ACS_center=(ny // 2), ACS_size=ACS_size)
        output_n, smap_n = model(torch.view_as_real(y_n_input), mask_n, ACS_center=(ny // 2), ACS_size=ACS_size)

        # ---- Loss (论文版) ----
        smap_m_1 = torch.view_as_complex(smap_m.squeeze())
        smap_n_1 = torch.view_as_complex(smap_n.squeeze())
        h_output_n = fmult(torch.view_as_complex(output_m), smap_m_1, mask_n)
        h_output_m = fmult(torch.view_as_complex(output_n), smap_n_1, mask_m)

        rec_loss = 0.5 * (
            F.mse_loss(torch.view_as_real(h_output_m).float().squeeze(),
                       torch.view_as_real(y_m).float().squeeze())
          + F.mse_loss(torch.view_as_real(h_output_n).float().squeeze(),
                       torch.view_as_real(y_n).float().squeeze())
        )

        smap_m_for_smooth = smap_m.squeeze().permute(0, 3, 1, 2)
        smap_n_for_smooth = smap_n.squeeze().permute(0, 3, 1, 2)
        smooth_loss = 0.5 * (
            gradient_loss(smap_m_for_smooth) +
            gradient_loss(smap_n_for_smooth)
        )

        loss = rec_loss + LAMBDA_SMOOTH * smooth_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # PSNR & SSIM
        output_show = complex_abs(output_m.cpu().detach().squeeze())
        output_show = normlize(output_show)
        target_show = normlize(torch.abs(x_hat.squeeze()))
        psnrs.append(compare_psnr(output_show.to(device), target_show.to(device)).cpu())
        ssims.append(compare_ssim(output_show[None, None].to(device), target_show[None, None].to(device)).cpu())
        losses.append(loss.item())

    train_loss_log.append(np.mean(losses))
    train_psnr_log.append(np.mean(psnrs))
    train_ssim_log.append(np.mean(ssims))
    writer.add_scalar("Loss_Train", train_loss_log[-1], epoch)
    writer.add_scalar("PSNR_Train", train_psnr_log[-1], epoch)
    writer.add_scalar("SSIM_Train", train_ssim_log[-1], epoch)

def val(epoch):
    model.eval()
    psnrs, ssims, losses = [], [], []
    with torch.no_grad():
        for iteration, samples in enumerate(valloader):
            x_hat, smps_hat, y, mask_m, mask_n = samples
            x_hat = x_hat.to(device)
            y = y.to(device)
            mask = mask_m.byte().to(device)  # 只用一张 mask

            y_in = y.squeeze(1) if y.shape[1] == 1 else y
            ny = y.shape[-2]
            ACS_size = ((ny // 2) - (int(ny * 0.2 * (2 / ACCELERATION)) // 2)) * 2

            output_m, _ = model(torch.view_as_real(y_in), mask, ACS_center=(ny // 2), ACS_size=ACS_size)

            output_show = normlize(complex_abs(output_m.cpu().detach().squeeze())).to(device)
            target_show = normlize(torch.abs(x_hat.squeeze())).to(device)
            psnrs.append(compare_psnr(output_show, target_show).cpu())
            ssims.append(compare_ssim(output_show[None, None], target_show[None, None]).cpu())
            losses.append(F.mse_loss(output_show, target_show).item())

    val_psnr_log.append(np.mean(psnrs))
    val_ssim_log.append(np.mean(ssims))
    val_loss_log.append(np.mean(losses))
    writer.add_scalar("Loss_Val", val_loss_log[-1], epoch)
    writer.add_scalar("PSNR_Val", val_psnr_log[-1], epoch)
    writer.add_scalar("SSIM_Val", val_ssim_log[-1], epoch)

    if epoch % 5 == 0 or val_psnr_log[-1] >= max(val_psnr_log):
        torch.save(model.state_dict(), os.path.join(save_root, f"N2N_{epoch:03d}.pth"))

# ========== 主训练循环 ==========
if __name__ == "__main__":
    early_stopper = EarlyStopping(patience=20, verbose=True)
    for epoch in range(EPOCHS):
        print(f"\n🔁 Epoch {epoch}/{EPOCHS}")
        train(epoch)
        val(epoch)
        writer.flush()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"🔹 Current learning rate: {current_lr:.6f}")

        # Early stopping
        early_stopper(val_psnr_log[-1], epoch)
        if early_stopper.early_stop:
            print(f"⛔ Early stopping at epoch {epoch}, best epoch {early_stopper.best_epoch}")
            break

    # 保存曲线 & 日志
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.plot(train_loss_log, label='Train'); plt.plot(val_loss_log, label='Val'); plt.title("Loss"); plt.legend(); plt.grid()
    plt.subplot(1, 3, 2); plt.plot(train_psnr_log, label='Train'); plt.plot(val_psnr_log, label='Val'); plt.title("PSNR"); plt.legend(); plt.grid()
    plt.subplot(1, 3, 3); plt.plot(train_ssim_log, label='Train'); plt.plot(val_ssim_log, label='Val'); plt.title("SSIM"); plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "training_curves.png"))

    # 复制到 HOME 目录
    for file in os.listdir(save_root):
        src = os.path.join(save_root, file)
        dst = os.path.join(save_root_final, file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    metrics_dict = {
        'epoch': list(range(len(train_loss_log))),
        'train_loss': train_loss_log,
        'val_loss': val_loss_log,
        'train_psnr': train_psnr_log,
        'val_psnr': val_psnr_log,
        'train_ssim': train_ssim_log,
        'val_ssim': val_ssim_log,
    }
    pd.DataFrame(metrics_dict).to_csv(os.path.join(save_root, "metrics.csv"), index=False)

# -*- coding: utf-8 -*-
"""
Created on 2025/8/2 20:34

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import pathlib
import argparse
from argparse import ArgumentParser
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
from utils.loss_functions import spicer_loss
# ========== HPC 环境设置 ==========
user = os.environ.get("USER", "unknown_user")
TMPDIR = os.environ.get("TMPDIR", f"/tmp/{user}")
HOME = os.path.expanduser("~")

os.environ['CUPY_CACHE_DIR'] = os.path.join(TMPDIR, "cupy")
os.environ['NUMBA_CACHE_DIR'] = os.path.join(TMPDIR, "numba")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# GPU 设置

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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

# Checkpoint 保存函数
def save_checkpoint(state, filename):
    torch.save(state, filename)

# EarlyStopping 类
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
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

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    return parser.parse_args()

args = parse_args()
epoch_number = args.epochs
patience = args.patience

# 模型与优化器初始化
model = SPNet(num_cascades=8, pools=4, chans=18, sens_pools=4, sens_chans=8).to(device)
# 优化器：Adam，初始 lr=0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

# 学习率调度器：30 epoch 后 lr ×0.1 → 0.0001
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

snr_best = []

# 日志变量
train_loss_log, val_loss_log = [], []
train_psnr_log, val_psnr_log = [], []
train_ssim_log, val_ssim_log = [], []

# 尝试恢复断点
resume_path = os.path.join(save_root, "checkpoint_last.pth")
start_epoch = 0
best_model_state = None
if os.path.exists(resume_path):
    print(f"🔁 恢复训练：{resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    train_loss_log = checkpoint['train_loss_log']
    val_loss_log = checkpoint['val_loss_log']
    train_psnr_log = checkpoint['train_psnr_log']
    val_psnr_log = checkpoint['val_psnr_log']
    train_ssim_log = checkpoint['train_ssim_log']
    val_ssim_log = checkpoint['val_ssim_log']

# 初始化 early stopping
early_stopper = EarlyStopping(patience=patience, verbose=True)

train_idx = range(564, 694)   # 130 subjects
val_idx   = range(694, 709)   # 15 subjects
test_idx  = range(709, 729)   # 20 subjects
# 数据集加载
dataset = RealMeasurement(idx_list=train_idx, acceleration_rate=8,
                          is_return_y_smps_hat=True,
                          mask_pattern='uniformly_cartesian',
                          smps_hat_method='eps')
trainloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

val_dataset = RealMeasurement(idx_list=val_idx, acceleration_rate=8,
                              is_return_y_smps_hat=True,
                              mask_pattern='uniformly_cartesian',
                              smps_hat_method='eps')
valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

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

        # 判断是否 squeeze(1)（有些 loader 会加一维 channel=1）
        if y_m.shape[1] == 1:
            y_m_input = y_m.squeeze(1)
            y_n_input = y_n.squeeze(1)
        else:
            y_m_input = y_m
            y_n_input = y_n

        ny = y_m.shape[-2]
        ACS_size = ((ny // 2) - (int(ny * 0.2 * (2 / 8)) // 2)) * 2

        output_m, smap_m = model(torch.view_as_real(y_m_input), mask_m, ACS_center=(ny // 2), ACS_size=ACS_size)
        output_n, smap_n = model(torch.view_as_real(y_n_input), mask_n, ACS_center=(ny // 2), ACS_size=ACS_size)

        smap_m_1 = torch.view_as_complex(smap_m.squeeze())
        smap_n_1 = torch.view_as_complex(smap_n.squeeze())

        h_output_n = fmult(torch.view_as_complex(output_m), smap_m_1, mask_n)
        h_output_m = fmult(torch.view_as_complex(output_n), smap_n_1, mask_m)

        loss = spicer_loss(h_output_m, y_m, h_output_n, y_n, smap_m, gamma=1.0, tau=0.1)

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
    print(f"[Train Debug] Epoch {epoch} - Loss: {train_loss_log[-1]:.6f}")


def val(epoch):
    model.eval()
    psnrs, ssims, losses = [], [], []
    with torch.no_grad():
        for iteration, samples in enumerate(valloader):
        #for samples in valloader:
            dicom, x0, y_input, smps_input, mask_input = samples
            dicom = dicom.to(device)  # ✅ 添加这一行
            y_m = y_input.to(device)
            mask_m = mask_input.byte().to(device)

            # ✅ 在第一步打印调试信息
            if iteration == 0:
                print(f"[Debug] y_m.shape = {y_m.shape}")  # e.g., [1, 16, 640, 368]
                print(f"[Debug] view_as_real(y_m.squeeze(1)).shape = {torch.view_as_real(y_m.squeeze(1)).shape}")  # e.g., [1, 16, 640, 368, 2]

            ny = y_m.shape[-2]
            ACS_size = ((ny // 2) - (int(ny * 0.2 * (2 / 8)) // 2)) * 2

            output_m, _ = model(torch.view_as_real(y_m.squeeze(1)), mask_m, ACS_center=(ny // 2), ACS_size=ACS_size)
            #output_m, _ = model(torch.view_as_real(y_m), mask_m, ACS_center=(ny // 2), ACS_size=ACS_size)
            output_show = normlize(complex_abs(output_m.cpu().detach().squeeze())).to(device)
            target_show = normlize(torch.abs(dicom.squeeze())).to(device)
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

# 主训练循环
if __name__ == "__main__":
    for epoch in range(start_epoch, epoch_number):
        print(f"\n🔁 Epoch {epoch}/{epoch_number} 开始")  # ✅ 打印 epoch 开始
        train(epoch)
        print(f"✅ Train epoch {epoch} completed. Loss: {train_loss_log[-1]:.4e}")  # ✅ 打印 train 结果
        val(epoch)
        print(f"✅ Val   epoch {epoch} completed. Loss: {val_loss_log[-1]:.4e}")  # ✅ 打印 val 结果
        writer.flush()  # ✅ 强制写入 TensorBoard 日志
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_log': train_loss_log,
            'val_loss_log': val_loss_log,
            'train_psnr_log': train_psnr_log,
            'val_psnr_log': val_psnr_log,
            'train_ssim_log': train_ssim_log,
            'val_ssim_log': val_ssim_log,
        }, os.path.join(save_root, "checkpoint_last.pth"))

        # 保存 best model
        if val_psnr_log[-1] == max(val_psnr_log):
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(save_root, "best_model.pth"))

        # Early stopping 检查
        early_stopper(val_psnr_log[-1], epoch)
        if early_stopper.early_stop:
            print(f"⛔️ Early stopping triggered at epoch {epoch}, best epoch was {early_stopper.best_epoch}")
            break
        # 🔹 每个 epoch 结束后更新学习率
        scheduler.step()

        # 🔹 可选：打印当前学习率
        current_lr = scheduler.get_last_lr()[0]
        print(f"🔹 Current learning rate: {current_lr:.6f}")


    # 训练完成后保存曲线图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.plot(train_loss_log, label='Train'); plt.plot(val_loss_log, label='Val'); plt.title("Loss"); plt.legend(); plt.grid()
    plt.subplot(1, 3, 2); plt.plot(train_psnr_log, label='Train'); plt.plot(val_psnr_log, label='Val'); plt.title("PSNR"); plt.legend(); plt.grid()
    plt.subplot(1, 3, 3); plt.plot(train_ssim_log, label='Train'); plt.plot(val_ssim_log, label='Val'); plt.title("SSIM"); plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "training_curves.png"))

    # 自动备份模型与图像
    print("\n✅ 拷贝模型与图像到:", save_root_final)
    os.makedirs(save_root_final, exist_ok=True)
    for file in os.listdir(save_root):
        src = os.path.join(save_root, file)
        dst = os.path.join(save_root_final, file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)  # 复制并保留时间戳
    print("✅ 模型与图像已复制完毕 ✅")

    # 将指标保存为 CSV 文件
    metrics_dict = {
        'epoch': list(range(start_epoch, start_epoch + len(train_loss_log))),
        'train_loss': train_loss_log,
        'val_loss': val_loss_log,
        'train_psnr': train_psnr_log,
        'val_psnr': val_psnr_log,
        'train_ssim': train_ssim_log,
        'val_ssim': val_ssim_log,
    }
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.to_csv(os.path.join(save_root, "metrics.csv"), index=False)

    #
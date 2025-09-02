# -*- coding: utf-8 -*-
"""
SPICER training (slice-level, aligned with paper & author's code)
- One optimizer.step per slice
- Train/Val loss: bidirectional k-space MSE + lambda * gradient_loss(CSM)
- LR: 1e-3 for first 30 epochs, then 1e-4
- Save init_model.pth at start for test-time "random init" outputs
"""
import os, random, shutil
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from networks.SPICER_fastmri_network import SPNet
from dataset.pmri_fastmri_brain_lunwen import RealMeasurement
from dataset.pmri_fastmri_brain import fmult
from utils.util import *
from utils.measures import *
from utils.loss_functions import gradient_loss

# ----------------- 固定随机种子（可复现） -----------------
SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------- 配置（与论文/作者对齐） -----------------
ACCELERATION = 8
EPOCHS = 200
LR_INIT = 1e-3
MILESTONES = [30]   # -> 1e-4
LAMBDA_SMOOTH = 0.001  # 与作者参考代码一致（论文文本提到的0.01用于另一处；作者实现这里用0.001）

# 设备
local_rank = int(os.environ.get("SLURM_PROCID", 0))
device = f'cuda:{local_rank % max(1, torch.cuda.device_count())}' if torch.cuda.is_available() else 'cpu'
print(f"🧠 Using device: {device}")

# 输出目录
user = os.environ.get("USER", "user")
TMPDIR = os.environ.get("TMPDIR", f"/tmp/{user}")
HOME = os.path.expanduser("~")
timestamp = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
save_root = os.path.join(TMPDIR, "spicer_out", f"SPICER_fastmri_{timestamp}")
save_root_final = os.path.join(HOME, "spicer_outputs", f"SPICER_fastmri_{timestamp}")
os.makedirs(save_root, exist_ok=True); os.makedirs(save_root_final, exist_ok=True)

# ----------------- 数据集（作者划分：130/15） -----------------
train_idx = range(564, 694)  # 130 subjects
val_idx   = range(694, 709)  # 15 subjects

train_set = RealMeasurement(train_idx, acceleration_rate=ACCELERATION,
                            is_return_y_smps_hat=True,
                            mask_pattern='uniformly_cartesian',
                            smps_hat_method='eps')
val_set   = RealMeasurement(val_idx,   acceleration_rate=ACCELERATION,
                            is_return_y_smps_hat=True,
                            mask_pattern='uniformly_cartesian',
                            smps_hat_method='eps')

trainloader = DataLoader(train_set, batch_size=1, shuffle=True,  num_workers=0)
valloader   = DataLoader(val_set,   batch_size=1, shuffle=False, num_workers=0)

# ----------------- 模型/优化器/调度器 -----------------
model = SPNet(num_cascades=8, pools=4, chans=18, sens_pools=4, sens_chans=8).to(device)

# —— 保存初始化权重（测试用） ——
init_ckpt_path = os.path.join(save_root, "init_model.pth")
torch.save(model.state_dict(), init_ckpt_path)
print(f"[INIT] Saved random-initialized weights to {init_ckpt_path}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)

# ----------------- 记录 -----------------
train_loss_log, val_loss_log = [], []
train_psnr_log, val_psnr_log = [], []
train_ssim_log, val_ssim_log = [], []
best_psnr = -1e9

# ----------------- 公共：损失计算（与训练/验证一致） -----------------
def compute_loss_pair(y_m, y_n, mask_m, mask_n, output_m, output_n, smap_m, smap_n, lambda_smooth=LAMBDA_SMOOTH):
    smap_m_1 = torch.view_as_complex(smap_m.squeeze())
    smap_n_1 = torch.view_as_complex(smap_n.squeeze())
    h_output_n = fmult(torch.view_as_complex(output_m), smap_m_1, mask_n)  # A_n(x_m)
    h_output_m = fmult(torch.view_as_complex(output_n), smap_n_1, mask_m)  # A_m(x_n)

    rec = 0.5 * (
        F.mse_loss(torch.view_as_real(h_output_m).float().squeeze(),
                   torch.view_as_real(y_m).float().squeeze())
      + F.mse_loss(torch.view_as_real(h_output_n).float().squeeze(),
                   torch.view_as_real(y_n).float().squeeze())
    )
    smap_m_for_smooth = smap_m.squeeze().permute(0, 3, 1, 2)
    smap_n_for_smooth = smap_n.squeeze().permute(0, 3, 1, 2)
    smooth = 0.5 * (gradient_loss(smap_m_for_smooth) + gradient_loss(smap_n_for_smooth))
    return rec + lambda_smooth * smooth

# ----------------- 训练/验证 -----------------
def train_one_epoch(epoch):
    model.train()
    psnrs, ssims, losses = [], [], []

    for samples in tqdm(trainloader, desc=f"Train [{epoch:03d}]"):
        x_hat, smps_hat, y, mask_m, mask_n = samples
        x_hat = x_hat.to(device)
        y = y.to(device)
        mask_m = mask_m.byte().to(device)
        mask_n = mask_n.byte().to(device)

        y_m, y_n = y * mask_m, y * mask_n
        y_m_in = y_m.squeeze(1) if y_m.shape[1] == 1 else y_m
        y_n_in = y_n.squeeze(1) if y_n.shape[1] == 1 else y_n

        ny = y_m.shape[-2]
        ACS = ((ny // 2) - (int(ny * 0.2 * (2 / ACCELERATION)) // 2)) * 2

        output_m, smap_m = model(torch.view_as_real(y_m_in), mask_m, ACS_center=(ny // 2), ACS_size=ACS)
        output_n, smap_n = model(torch.view_as_real(y_n_in), mask_n, ACS_center=(ny // 2), ACS_size=ACS)

        loss = compute_loss_pair(y_m, y_n, mask_m, mask_n, output_m, output_n, smap_m, smap_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 图像域监控
        out = normlize(complex_abs(output_m.detach().cpu().squeeze())).to(device)
        tgt = normlize(torch.abs(x_hat.squeeze())).to(device)
        psnrs.append(compare_psnr(out, tgt).cpu())
        ssims.append(compare_ssim(out[None, None], tgt[None, None]).cpu())
        losses.append(loss.item())

    train_loss_log.append(float(np.mean(losses)))
    train_psnr_log.append(float(np.mean(psnrs)))
    train_ssim_log.append(float(np.mean(ssims)))

def validate(epoch):
    model.eval()
    psnrs, ssims, losses_trainstyle = [], [], []

    with torch.no_grad():
        for x_hat, smps_hat, y, mask_m, mask_n in valloader:
            x_hat = x_hat.to(device)
            y = y.to(device)
            mask_m = mask_m.byte().to(device)
            mask_n = mask_n.byte().to(device)

            y_m, y_n = y * mask_m, y * mask_n
            y_m_in = y_m.squeeze(1) if y_m.shape[1] == 1 else y_m
            y_n_in = y_n.squeeze(1) if y_n.shape[1] == 1 else y_n

            ny = y_m.shape[-2]
            ACS = ((ny // 2) - (int(ny * 0.2 * (2 / ACCELERATION)) // 2)) * 2

            output_m, smap_m = model(torch.view_as_real(y_m_in), mask_m, ACS_center=(ny // 2), ACS_size=ACS)
            output_n, smap_n = model(torch.view_as_real(y_n_in), mask_n, ACS_center=(ny // 2), ACS_size=ACS)

            # 1) 与训练一致的损失
            loss_trainstyle = compute_loss_pair(y_m, y_n, mask_m, mask_n, output_m, output_n, smap_m, smap_n)
            losses_trainstyle.append(loss_trainstyle.item())

            # 2) 图像域指标
            out = normlize(complex_abs(output_m.detach().cpu().squeeze())).to(device)
            tgt = normlize(torch.abs(x_hat.squeeze())).to(device)
            psnrs.append(compare_psnr(out, tgt).cpu())
            ssims.append(compare_ssim(out[None, None], tgt[None, None]).cpu())

    val_loss_log.append(float(np.mean(losses_trainstyle)))
    val_psnr_log.append(float(np.mean(psnrs)))
    val_ssim_log.append(float(np.mean(ssims)))

# ----------------- 主训练循环 -----------------
if __name__ == "__main__":
    for epoch in range(EPOCHS):
        print(f"\n🔁 Epoch {epoch}/{EPOCHS}")
        train_one_epoch(epoch)
        validate(epoch)

        # 保存日志 + checkpoint
        np.savetxt(os.path.join(save_root, "train_loss.txt"), np.array(train_loss_log))
        np.savetxt(os.path.join(save_root, "val_loss.txt"),   np.array(val_loss_log))
        np.savetxt(os.path.join(save_root, "train_psnr.txt"), np.array(train_psnr_log))
        np.savetxt(os.path.join(save_root, "val_psnr.txt"),   np.array(val_psnr_log))
        np.savetxt(os.path.join(save_root, "train_ssim.txt"), np.array(train_ssim_log))
        np.savetxt(os.path.join(save_root, "val_ssim.txt"),   np.array(val_ssim_log))
        torch.save(model.state_dict(), os.path.join(save_root, "checkpoint_last.pth"))

        # best by Val PSNR
        if val_psnr_log[-1] > best_psnr:
            best_psnr = val_psnr_log[-1]
            torch.save(model.state_dict(), os.path.join(save_root, "best_model.pth"))

        # lr 调度
        scheduler.step()
        print(f"🔹 LR = {scheduler.get_last_lr()[0]:.6f}")

    # 曲线图
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.plot(train_loss_log,label='Train'); plt.plot(val_loss_log,label='Val'); plt.title("Loss"); plt.legend(); plt.grid()
    plt.subplot(1,3,2); plt.plot(train_psnr_log,label='Train'); plt.plot(val_psnr_log,label='Val'); plt.title("PSNR"); plt.legend(); plt.grid()
    plt.subplot(1,3,3); plt.plot(train_ssim_log,label='Train'); plt.plot(val_ssim_log,label='Val'); plt.title("SSIM"); plt.legend(); plt.grid()
    plt.tight_layout(); plt.savefig(os.path.join(save_root, "training_curves.png"))

    # 复制到 HOME
    for fn in os.listdir(save_root):
        src = os.path.join(save_root, fn)
        dst = os.path.join(save_root_final, fn)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f"✅ Artifacts copied to: {save_root_final}")

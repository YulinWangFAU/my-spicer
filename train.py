import os
import pathlib
from argparse import ArgumentParser
import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from networks.SPICER_fastmri_network import SPNet
from tqdm import tqdm
from datetime import datetime
from dataset.pmri_fastmri_brain import RealMeasurement
from torch.utils.data import DataLoader
from utils.util import *
from utils.measures import *
from dataset.pmri_fastmri_brain import fmult
from utils.loss_functions import gradient_loss
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

# ========== HPC 环境设置 ==========
user = os.environ.get("USER", "unknown_user")
TMPDIR = os.environ.get("TMPDIR", f"/tmp/{user}")
HOME = os.path.expanduser("~")

os.environ['CUPY_CACHE_DIR'] = os.path.join(TMPDIR, "cupy")
os.environ['NUMBA_CACHE_DIR'] = os.path.join(TMPDIR, "numba")
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# GPU 设置
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 当前时间戳
now = datetime.now()
timestamp = now.strftime("%d-%b-%Y-%H-%M-%S")

# 模型保存路径
model_name = 'SPICER_fastmri'
save_root_tmp = os.path.join(TMPDIR, "spicer_out", f"{model_name}_{timestamp}")
save_root_final = os.path.join(HOME, "spicer_outputs", model_name)
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

# 模型与优化器初始化
model = SPNet(num_cascades=6, pools=4, chans=18, sens_pools=4, sens_chans=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0)

# 训练参数
epoch_number = 200
acceleration_factor = 8
snr_best = []

# 日志变量
train_loss_log, val_loss_log = [], []
train_psnr_log, val_psnr_log = [], []
train_ssim_log, val_ssim_log = [], []

# 尝试恢复断点
resume_path = os.path.join(save_root, "checkpoint_last.pth")
start_epoch = 0
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

# 数据集加载
dataset = RealMeasurement(idx_list=range(0, 695), acceleration_rate=acceleration_factor,
                          is_return_y_smps_hat=True, mask_pattern='uniformly_cartesian', smps_hat_method='eps')
trainloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

val_dataset = RealMeasurement(idx_list=range(695, 714), acceleration_rate=acceleration_factor,
                              is_return_y_smps_hat=True, mask_pattern='uniformly_cartesian', smps_hat_method='eps')
valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

def train(epoch):
    model.train()
    psnrs, losses, ssims = [], [], []
    for samples in tqdm(trainloader, desc=f"Train [{epoch:03d}]"):
        x_hat, smps_hat, y, mask_m, mask_n = samples
        mask_m = mask_m.byte().to(device)
        mask_n = mask_n.byte().to(device)
        y = y.to(device)
        y_m = y * mask_m
        y_n = y * mask_n
        ny = y_m.shape[-2]
        ACS_size = ((ny // 2) - (int(ny * 0.2 * (2 / acceleration_factor)) // 2)) * 2

        output_m, smap_m = model(torch.view_as_real(y_m), mask_m, ACS_center=(ny // 2), ACS_size=ACS_size)
        output_n, smap_n = model(torch.view_as_real(y_n), mask_n, ACS_center=(ny // 2), ACS_size=ACS_size)
        smap_m_1 = torch.view_as_complex(smap_m.squeeze())
        smap_n_1 = torch.view_as_complex(smap_n.squeeze())
        h_output_n = fmult(torch.view_as_complex(output_m), smap_m_1, mask_n)
        h_output_m = fmult(torch.view_as_complex(output_n), smap_n_1, mask_m)

        loss = ((F.mse_loss(torch.view_as_real(h_output_m).float().squeeze(), torch.view_as_real(y_m).float().squeeze()) +
                 F.mse_loss(torch.view_as_real(h_output_n).float().squeeze(), torch.view_as_real(y_n).float().squeeze())) / 2)
        loss += 0.001 * gradient_loss(smap_m.squeeze().permute(0, 3, 1, 2))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_show = complex_abs(output_m.cpu().detach().squeeze())
        output_show = normlize(output_show)
        target_show = normlize(torch.abs(x_hat.squeeze()))

        psnrs.append(compare_psnr(output_show, target_show))
        ssims.append(compare_ssim(output_show[None, None], target_show[None, None]))
        losses.append(loss.item())

    train_loss_log.append(np.mean(losses))
    train_psnr_log.append(np.mean(psnrs))
    train_ssim_log.append(np.mean(ssims))

    writer.add_scalar("Loss/Train", train_loss_log[-1], epoch)
    writer.add_scalar("PSNR/Train", train_psnr_log[-1], epoch)
    writer.add_scalar("SSIM/Train", train_ssim_log[-1], epoch)

def val(epoch):
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for samples in valloader:
            dicom, x0, y_input, smps_input, mask_input = samples
            y_m = y_input.to(device)
            mask_m = mask_input.byte().to(device)
            ny = y_m.shape[-2]
            ACS_size = ((ny // 2) - (int(ny * 0.2 * (2 / acceleration_factor)) // 2)) * 2
            output_m, _ = model(torch.view_as_real(y_m), mask_m, ACS_center=(ny // 2), ACS_size=ACS_size)
            output_show = normlize(complex_abs(output_m.cpu().detach().squeeze()))
            target_show = normlize(torch.abs(dicom.squeeze()))
            psnrs.append(compare_psnr(output_show, target_show))
            ssims.append(compare_ssim(output_show[None, None], target_show[None, None]))

    val_psnr_log.append(np.mean(psnrs))
    val_ssim_log.append(np.mean(ssims))
    writer.add_scalar("PSNR/Val", val_psnr_log[-1], epoch)
    writer.add_scalar("SSIM/Val", val_ssim_log[-1], epoch)

    if epoch % 5 == 0 or val_psnr_log[-1] >= max(val_psnr_log):
        torch.save(model.state_dict(), os.path.join(save_root, f"N2N_{epoch:03d}.pth"))

# 主训练循环
for epoch in range(start_epoch, epoch_number):
    train(epoch)
    val(epoch)
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

# 训练完成后保存曲线图
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.plot(train_loss_log); plt.title("Train Loss"); plt.grid()
plt.subplot(1, 3, 2); plt.plot(train_psnr_log, label='Train'); plt.plot(val_psnr_log, label='Val'); plt.title("PSNR"); plt.legend(); plt.grid()
plt.subplot(1, 3, 3); plt.plot(train_ssim_log, label='Train'); plt.plot(val_ssim_log, label='Val'); plt.title("SSIM"); plt.legend(); plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_root, "training_curves.png"))
print("\n✅ 模型训练与保存完成，图像与模型保存在:", save_root_final)

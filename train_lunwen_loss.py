# -*- coding: utf-8 -*-
"""
SPICER training (slice-level, aligned with paper & author's code)
- 保留你的整体结构、EarlyStopping 和 TensorBoard
- 增强：可续训(恢复 optimizer/scheduler/日志/随机数)、捕获 SIGTERM、保存 init_model.pth
- 修正：val() 的样本解包顺序、ACS 使用实际加速因子
"""
import os, sys, argparse, random, signal, shutil
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from networks.SPICER_fastmri_network import SPNet
from dataset.pmri_fastmri_brain_lunwen import RealMeasurement
from dataset.pmri_fastmri_brain import fmult
from utils.util import *
from utils.measures import *
from utils.loss_functions import gradient_loss, spicer_loss
import pandas as pd
import torch.optim as optim

# ----------------- 解析参数（新增 resume 选项） -----------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--acceleration', type=int, default=4, help='undersampling factor (e.g., 4 or 8)')
    # 续训相关（可选）
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
    parser.add_argument('--resume_path', type=str, default='', help='path to a checkpoint_xxx.pth')
    parser.add_argument('--resume_last_dir', type=str, default='', help='directory that contains checkpoint_last.pth')
    return parser.parse_args()

args = parse_args()
epoch_number = args.epochs
patience = args.patience
ACCELERATION = int(args.acceleration)

# ----------------- HPC 环境 & 设备 -----------------
user = os.environ.get("USER", "unknown_user")
TMPDIR = os.environ.get("TMPDIR", f"/tmp/{user}")
HOME = os.path.expanduser("~")
os.environ['CUPY_CACHE_DIR'] = os.path.join(TMPDIR, "cupy")
os.environ['NUMBA_CACHE_DIR'] = os.path.join(TMPDIR, "numba")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

local_rank = int(os.environ.get("SLURM_PROCID", 0))
device = f'cuda:{local_rank % max(1, torch.cuda.device_count())}' if torch.cuda.is_available() else 'cpu'
print(f"🧠 Using device: {device}")

# 固定随机种子（可复现续训）
SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------- 输出路径 -----------------
now = datetime.now()
timestamp = now.strftime("%d-%b-%Y-%H-%M-%S")
model_name = 'SPICER_fastmri'
save_root_tmp = os.path.join(TMPDIR, "spicer_out", f"{model_name}_{timestamp}")
save_root_final = os.path.join(HOME, "spicer_outputs", f"{model_name}_{timestamp}")
os.makedirs(save_root_tmp, exist_ok=True)
os.makedirs(save_root_final, exist_ok=True)
save_root = save_root_tmp

# TensorBoard
log_dir = os.path.join(TMPDIR, "tensorboard_logs", timestamp)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# ----------------- EarlyStopping -----------------
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

# ----------------- 数据 -----------------
train_idx = range(564, 694)   # 130 subjects
val_idx   = range(694, 709)   # 15 subjects
# test_idx  = range(709, 729)   # 20 subjects （测试脚本使用）

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

# ----------------- 模型/优化器/调度器 -----------------
model = SPNet(num_cascades=8, pools=4, chans=18, sens_pools=4, sens_chans=8).to(device)

# 保存“随机初始化”权重，测试时可用来生成“初始化输出”
init_weight_path = os.path.join(save_root, "init_model.pth")
torch.save(model.state_dict(), init_weight_path)
print(f"[INIT] Saved random-initialized weights -> {init_weight_path}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

# ----------------- 训练日志 -----------------
snr_best = []
train_loss_log, val_loss_log = [], []
train_psnr_log, val_psnr_log = [], []
train_ssim_log, val_ssim_log = [], []
best_model_state = None
best_psnr = -1e9

# ----------------- 断点保存/恢复（增强） -----------------
def save_checkpoint_full(epoch, tag="last"):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss_log': train_loss_log,
        'val_loss_log': val_loss_log,
        'train_psnr_log': train_psnr_log,
        'val_psnr_log': val_psnr_log,
        'train_ssim_log': train_ssim_log,
        'val_ssim_log': val_ssim_log,
        'rng': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        'meta': {
            'acceleration': ACCELERATION,
        }
    }
    path = os.path.join(save_root, f"checkpoint_{tag}.pth")
    torch.save(ckpt, path)
    if tag == "last":
        torch.save(ckpt, os.path.join(save_root, "checkpoint_last.pth"))
    print(f"[CKPT] saved -> {path}")

def try_resume():
    # 1) 显式 --resume_path
    if args.resume and args.resume_path and os.path.exists(args.resume_path):
        path = args.resume_path
    # 2) 指定目录下的 checkpoint_last.pth
    elif args.resume_last_dir and os.path.exists(os.path.join(args.resume_last_dir, "checkpoint_last.pth")):
        path = os.path.join(args.resume_last_dir, "checkpoint_last.pth")
    # 3) 当前 save_root 下的 checkpoint_last.pth
    else:
        path = os.path.join(save_root, "checkpoint_last.pth")
        if not os.path.exists(path):
            return 0  # 不恢复

    print(f"🔁 恢复训练：{path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    # 恢复日志
    train_loss_log[:] = ckpt.get('train_loss_log', [])
    val_loss_log[:]   = ckpt.get('val_loss_log', [])
    train_psnr_log[:] = ckpt.get('train_psnr_log', [])
    val_psnr_log[:]   = ckpt.get('val_psnr_log', [])
    train_ssim_log[:] = ckpt.get('train_ssim_log', [])
    val_ssim_log[:]   = ckpt.get('val_ssim_log', [])

    # 恢复随机数状态（兼容旧格式）
    rng = ckpt.get('rng', {})
    if rng.get('python') is not None:
        random.setstate(rng['python'])
    if rng.get('numpy') is not None:
        np.random.set_state(rng['numpy'])
    if rng.get('torch') is not None:
        torch.set_rng_state(torch.as_tensor(rng['torch']))
    if torch.cuda.is_available() and rng.get('torch_cuda') is not None:
        cuda_states = []
        for st in rng['torch_cuda']:
            cuda_states.append(torch.as_tensor(st, dtype=torch.uint8))
        torch.cuda.set_rng_state_all(cuda_states)

    start_epoch = ckpt.get('epoch', -1) + 1
    print(f"[CKPT] resume start_epoch = {start_epoch}")
    return start_epoch

# 捕获 SLURM SIGTERM，紧急保存
CURRENT_EPOCH = 0
def _handle_sigterm(signum, frame):
    print("[CKPT] Caught SIGTERM, saving checkpoint_last and exiting...")
    save_checkpoint_full(CURRENT_EPOCH, tag="last")
    sys.exit(0)
signal.signal(signal.SIGTERM, _handle_sigterm)

# ----------------- 训练/验证 -----------------
def train(epoch):
    model.train()
    psnrs, losses, ssims = [], [], []
    for iteration, samples in enumerate(tqdm(trainloader, desc=f"Train [{epoch:03d}]")):
        # 数据解包：与你的数据集 RealMeasurement 对齐
        x_hat, smps_hat, y, mask_m, mask_n = samples
        x_hat = x_hat.to(device)
        mask_m = mask_m.byte().to(device)
        mask_n = mask_n.byte().to(device)
        y = y.to(device)
        y_m = y * mask_m
        y_n = y * mask_n

        # squeeze 以适配 model 的输入
        y_m_input = y_m.squeeze(1) if y_m.shape[1] == 1 else y_m
        y_n_input = y_n.squeeze(1) if y_n.shape[1] == 1 else y_n

        ny = y_m.shape[-2]
        ACS_size = ((ny // 2) - (int(ny * 0.2 * (2 / ACCELERATION)) // 2)) * 2

        output_m, smap_m = model(torch.view_as_real(y_m_input), mask_m, ACS_center=(ny // 2), ACS_size=ACS_size)
        output_n, smap_n = model(torch.view_as_real(y_n_input), mask_n, ACS_center=(ny // 2), ACS_size=ACS_size)

        smap_m_1 = torch.view_as_complex(smap_m.squeeze())
        smap_n_1 = torch.view_as_complex(smap_n.squeeze())

        h_output_n = fmult(torch.view_as_complex(output_m), smap_m_1, mask_n)
        h_output_m = fmult(torch.view_as_complex(output_n), smap_n_1, mask_m)

        # 与作者实现一致的 SPICER loss
        loss = spicer_loss(h_output_m, y_m, h_output_n, y_n, smap_m, gamma=1.0, tau=0.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # PSNR & SSIM（图像域监控）
        output_show = complex_abs(output_m.detach().cpu().squeeze())
        output_show = normlize(output_show)
        target_show = normlize(torch.abs(x_hat.squeeze()))
        psnrs.append(compare_psnr(output_show.to(device), target_show.to(device)).cpu())
        ssims.append(compare_ssim(output_show[None, None].to(device), target_show[None, None].to(device)).cpu())
        losses.append(loss.item())

    train_loss_log.append(float(np.mean(losses)))
    train_psnr_log.append(float(np.mean(psnrs)))
    train_ssim_log.append(float(np.mean(ssims)))

    writer.add_scalar("Loss_Train", train_loss_log[-1], epoch)
    writer.add_scalar("PSNR_Train", train_psnr_log[-1], epoch)
    writer.add_scalar("SSIM_Train", train_ssim_log[-1], epoch)
    print(f"[Train Debug] Epoch {epoch} - Loss: {train_loss_log[-1]:.6f}")

def val(epoch):
    model.eval()
    psnrs, ssims, losses = [], [], []
    with torch.no_grad():
        for iteration, samples in enumerate(valloader):
            # 和 train() 一致的数据解包
            x_hat, smps_hat, y, mask_m, mask_n = samples
            x_hat = x_hat.to(device)
            mask_m = mask_m.byte().to(device)
            mask_n = mask_n.byte().to(device)
            y = y.to(device)
            y_m = y * mask_m
            y_n = y * mask_n

            # squeeze 适配模型输入
            y_m_input = y_m.squeeze(1) if y_m.shape[1] == 1 else y_m
            y_n_input = y_n.squeeze(1) if y_n.shape[1] == 1 else y_n

            ny = y_m.shape[-2]
            ACS_size = ((ny // 2) - (int(ny * 0.2 * (2 / ACCELERATION)) // 2)) * 2

            # 前向传播
            output_m, smap_m = model(torch.view_as_real(y_m_input), mask_m, ACS_center=(ny // 2), ACS_size=ACS_size)
            output_n, smap_n = model(torch.view_as_real(y_n_input), mask_n, ACS_center=(ny // 2), ACS_size=ACS_size)

            smap_m_1 = torch.view_as_complex(smap_m.squeeze())
            smap_n_1 = torch.view_as_complex(smap_n.squeeze())

            h_output_n = fmult(torch.view_as_complex(output_m), smap_m_1, mask_n)
            h_output_m = fmult(torch.view_as_complex(output_n), smap_n_1, mask_m)

            # === 关键改动：验证也用 SPICER loss ===
            loss = spicer_loss(h_output_m, y_m, h_output_n, y_n, smap_m, gamma=1.0, tau=0.1)

            # 图像域监控指标
            output_show = normlize(complex_abs(output_m.detach().cpu().squeeze())).to(device)
            target_show = normlize(torch.abs(x_hat.squeeze())).to(device)
            psnrs.append(compare_psnr(output_show, target_show).cpu())
            ssims.append(compare_ssim(output_show[None, None], target_show[None, None]).cpu())
            losses.append(loss.item())

    val_psnr_log.append(float(np.mean(psnrs)))
    val_ssim_log.append(float(np.mean(ssims)))
    val_loss_log.append(float(np.mean(losses)))

    writer.add_scalar("Loss_Val", val_loss_log[-1], epoch)
    writer.add_scalar("PSNR_Val", val_psnr_log[-1], epoch)
    writer.add_scalar("SSIM_Val", val_ssim_log[-1], epoch)

    # 保存模型
    if epoch % 5 == 0 or val_psnr_log[-1] >= max(val_psnr_log):
        torch.save(model.state_dict(), os.path.join(save_root, f"N2N_{epoch:03d}.pth"))

    print(f"[Val Debug] Epoch {epoch} - Loss: {val_loss_log[-1]:.6f}, PSNR: {val_psnr_log[-1]:.4f}, SSIM: {val_ssim_log[-1]:.4f}")


# ----------------- 主训练循环（含续训 & SIGTERM 保护） -----------------
start_epoch = try_resume()  # 若找不到断点则返回 0

early_stopper = EarlyStopping(patience=patience, verbose=True)

for epoch in range(start_epoch, epoch_number):
    CURRENT_EPOCH = epoch  # 供 SIGTERM handler 使用
    print(f"\n🔁 Epoch {epoch}/{epoch_number} 开始")
    train(epoch)
    print(f"✅ Train epoch {epoch} completed. Loss: {train_loss_log[-1]:.4e}")
    val(epoch)
    print(f"✅ Val   epoch {epoch} completed. Loss: {val_loss_log[-1]:.4e}")

    # 刷新 TensorBoard
    writer.flush()

    # 保存“last”断点（全量）
    save_checkpoint_full(epoch, tag="last")

    # 保存 best model（以 Val PSNR 为准）
    #if val_psnr_log[-1] > best_psnr:
    #    best_psnr = val_psnr_log[-1]
    #    torch.save(model.state_dict(), os.path.join(save_root, "best_model.pth"))
    if val_psnr_log[-1] > best_psnr:
        best_psnr = val_psnr_log[-1]
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'psnr': best_psnr,
        }, os.path.join(save_root, "best_model.pth"))
        print(f"[BEST] Updated best_model at epoch {epoch}, PSNR={best_psnr:.4f}")

    # Early stopping
    early_stopper(val_psnr_log[-1], epoch)
    if early_stopper.early_stop:
        print(f"⛔️ Early stopping triggered at epoch {epoch}, best epoch was {early_stopper.best_epoch}")
        break

    # 学习率调度
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"🔹 Current learning rate: {current_lr:.6f}")

# ----------------- 训练收尾：曲线 + 复制产物 -----------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.plot(train_loss_log, label='Train'); plt.plot(val_loss_log, label='Val'); plt.title("Loss"); plt.legend(); plt.grid()
plt.subplot(1, 3, 2); plt.plot(train_psnr_log, label='Train'); plt.plot(val_psnr_log, label='Val'); plt.title("PSNR"); plt.legend(); plt.grid()
plt.subplot(1, 3, 3); plt.plot(train_ssim_log, label='Train'); plt.plot(val_ssim_log, label='Val'); plt.title("SSIM"); plt.legend(); plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_root, "training_curves.png"))

print("\n✅ 拷贝模型与图像到:", save_root_final)
os.makedirs(save_root_final, exist_ok=True)
for file in os.listdir(save_root):
    src = os.path.join(save_root, file)
    dst = os.path.join(save_root_final, file)
    #if not os.path.exists(dst):
    shutil.copy2(src, dst)
print("✅ 模型与图像已复制完毕 ✅")

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


#import torch

#ckpt = torch.load("spicer_out/.../best_model.pth", map_location="cpu")
#print("Best model at epoch:", ckpt['epoch'])
#print("Val PSNR:", ckpt['psnr'])
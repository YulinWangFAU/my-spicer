# -*- coding: utf-8 -*-
"""
SPICER test script (slice-level, four-panel figures)
Outputs per-slice:
- GT (magnitude of x_hat)
- Zero-filled (ftran(y*mask_m))
- Trained model output
- Random-initialized model output (using init_model.pth from training)

Also saves metrics (PSNR/SSIM/NMSE) per-slice, per-subject, and overall.
"""

import os, argparse, math
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from networks.SPICER_fastmri_network import SPNet
from dataset.pmri_fastmri_brain_lunwen import RealMeasurement
from dataset.pmri_fastmri_brain import ftran
from utils.util import *
from utils.measures import *

# ----------------- args -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--acceleration', type=int, default=4, help='undersampling R (e.g., 4 or 8)')
    p.add_argument('--test_begin', type=int, default=709, help='test index begin (inclusive)')
    p.add_argument('--test_end',   type=int, default=729, help='test index end (exclusive)')
    p.add_argument('--model_dir',  type=str, required=True,
                   help='the training output dir that contains best_model.pth and init_model.pth')
    p.add_argument('--use_best',   action='store_true',
                   help='use best_model.pth (if not set, uses checkpoint_last.pth)')
    p.add_argument('--save_root',  type=str, default='recon_results',
                   help='root to save test outputs')
    p.add_argument('--batch_size', type=int, default=1)
    return p.parse_args()

args = parse_args()
ACCELERATION = args.acceleration

# ----------------- device -----------------
local_rank = int(os.environ.get("SLURM_PROCID", 0))
device = f'cuda:{local_rank % max(1, torch.cuda.device_count())}' if torch.cuda.is_available() else 'cpu'
print(f"🧠 Using device: {device}")

# ----------------- data -----------------
test_idx = range(args.test_begin, args.test_end)
test_set = RealMeasurement(
    idx_list=test_idx,
    acceleration_rate=ACCELERATION,
    is_return_y_smps_hat=True,
    mask_pattern='uniformly_cartesian',
    smps_hat_method='eps',
)
testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

# ----------------- model -----------------
def build_model():
    return SPNet(num_cascades=8, pools=4, chans=18, sens_pools=4, sens_chans=8).to(device)

# trained model
model = build_model()
ckpt_path = os.path.join(args.model_dir, "best_model.pth" if args.use_best else "checkpoint_last.pth")
print(f"🔍 Load trained model: {ckpt_path}")
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.eval()

# random-init model (for “初始化输出”)
model_init = build_model()
init_path = os.path.join(args.model_dir, "init_model.pth")
print(f"🔍 Load random-initialized weights: {init_path}")
state_init = torch.load(init_path, map_location=device)
model_init.load_state_dict(state_init)
model_init.eval()

# ----------------- output dirs -----------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(args.save_root, f"spicer_test_R{ACCELERATION}_{timestamp}")
os.makedirs(save_dir, exist_ok=True)
print(f"💾 Saving to: {os.path.abspath(save_dir)}")

# ----------------- helpers -----------------
def to_mag_norm(x):
    """x: torch.Tensor [H,W] or [1,H,W] or [C,H,W]; return [H,W] (cpu) normalized [0,1]"""
    if x.dim() == 3:
        x = x.squeeze(0)
    x = normlize(x).cpu()
    return x

def save_four_panel(figpath, gt, zf, trained, init_out, suptitle=None):
    plt.figure(figsize=(10, 8))

    plt.subplot(2,2,1); plt.imshow(gt, cmap='gray'); plt.title('GT'); plt.axis('off')
    plt.subplot(2,2,2); plt.imshow(zf, cmap='gray'); plt.title('Zero-filled'); plt.axis('off')
    plt.subplot(2,2,3); plt.imshow(trained, cmap='gray'); plt.title('Trained Output'); plt.axis('off')
    plt.subplot(2,2,4); plt.imshow(init_out, cmap='gray'); plt.title('Init-Model Output'); plt.axis('off')

    if suptitle:
        plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(figpath, dpi=200)
    plt.close()

def compute_ACS(ny, R):
    # 和训练一致：acs_percentage=0.2
    return ((ny // 2) - (int(ny * 0.2 * (2 / R)) // 2)) * 2

# ----------------- run -----------------
rows = []  # per-slice metrics
subject_rows = {}  # accumulate per subject

with torch.no_grad():
    slice_counter = 0
    for batch in tqdm(testloader, desc="Testing"):
        # DataLoader(batch_size=1) → 返回的都是 size=1 的批
        # RealMeasurement 返回: x_hat, smps_hat, y, mask1, mask2
        x_hat, smps_hat, y, mask_m, mask_n = batch

        # ---- move ----
        x_hat = x_hat.to(device)              # [B,H,W] complex
        smps_hat = smps_hat.to(device)        # [B,C,H,W] complex
        y = y.to(device)                      # [B,C,H,W] complex (kspace)
        mask_m = mask_m.byte().to(device)     # [B,H,W]
        # 输入模型的 kspace：y_m
        y_m = y * mask_m

        # ---- shapes / ACS ----
        ny = y_m.shape[-2]
        ACS_size = compute_ACS(ny, ACCELERATION)
        ACS_center = (ny // 2)

        # ---- GT magnitude ----
        gt_mag = to_mag_norm(torch.abs(x_hat.squeeze()))  # [H,W]

        # ---- Zero-filled (用 ftran 与 smps_hat、mask_m) ----
        # ftran expects torch Tensors of complex dtype with shapes aligned:
        # y_m: [B,C,H,W] complex, smps_hat: [B,C,H,W] complex, mask: [B,H,W] float/bool
        zf_cplx = ftran(y_m.squeeze(0), smps_hat.squeeze(0), mask_m.squeeze(0))  # [H,W] complex
        zf_mag = to_mag_norm(torch.abs(zf_cplx))

        # ---- 模型输入 (view_as_real) ----
        # SPNet forward expects: (B,C,H,W,2) real-imag split
        y_m_in = y_m.squeeze(1) if y_m.shape[1] == 1 else y_m  # 适配是否有多余维
        y_m_in = torch.view_as_real(y_m_in)  # [B,C,H,W,2]

        # ---- Trained model output ----
        out_trained, _ = model(y_m_in, mask_m, ACS_center=ACS_center, ACS_size=ACS_size)  # image domain complex
        trained_mag = to_mag_norm(complex_abs(out_trained.squeeze().cpu()))

        # ---- Init model output ----
        out_init, _ = model_init(y_m_in, mask_m, ACS_center=ACS_center, ACS_size=ACS_size)
        init_mag = to_mag_norm(complex_abs(out_init.squeeze().cpu()))

        # ---- metrics (与论文常用一致) ----
        # 注意 compare_* 接收 torch.Tensor
        gt_t   = torch.from_numpy(gt_mag.numpy())
        zf_t   = torch.from_numpy(zf_mag.numpy())
        tr_t   = torch.from_numpy(trained_mag.numpy())
        init_t = torch.from_numpy(init_mag.numpy())

        psnr_zf   = float(compare_psnr(zf_t,   gt_t))
        ssim_zf   = float(compare_ssim(zf_t[None,None],   gt_t[None,None]))
        nmse_zf   = float(compare_nmse(zf_t,   gt_t))

        psnr_tr   = float(compare_psnr(tr_t,   gt_t))
        ssim_tr   = float(compare_ssim(tr_t[None,None],   gt_t[None,None]))
        nmse_tr   = float(compare_nmse(tr_t,   gt_t))

        psnr_init = float(compare_psnr(init_t, gt_t))
        ssim_init = float(compare_ssim(init_t[None,None], gt_t[None,None]))
        nmse_init = float(compare_nmse(init_t, gt_t))

        # ---- subject name（从 dataset 里取文件名） ----
        # RealMeasurement 内部记录了 ret['x_hat'] 的路径；我们从 batch 无法直接拿到，
        # 这里用 test_set._RealMeasurement__index_maps[slice_counter] 取（私有属性，但简单可用）。
        # 也可以在 dataloader 外层使用 enumerate(test_set) 逐样本迭代拿到路径，这里为简便：
        ret, s = test_set._RealMeasurement__index_maps[slice_counter]
        fname = os.path.basename(ret['x_hat']).replace('.h5','')

        # ---- save four panel ----
        panel_dir = os.path.join(save_dir, fname)
        os.makedirs(panel_dir, exist_ok=True)
        fig_path = os.path.join(panel_dir, f"{fname}_slice{s:03d}.png")
        save_four_panel(fig_path, gt_mag, zf_mag, trained_mag, init_mag,
                        suptitle=f"{fname} | slice {s}")

        # 也顺带各自单图保存（可选）
        plt.imsave(os.path.join(panel_dir, f"{fname}_slice{s:03d}_gt.png"), gt_mag.numpy(), cmap='gray', dpi=200)
        plt.imsave(os.path.join(panel_dir, f"{fname}_slice{s:03d}_zf.png"), zf_mag.numpy(), cmap='gray', dpi=200)
        plt.imsave(os.path.join(panel_dir, f"{fname}_slice{s:03d}_trained.png"), trained_mag.numpy(), cmap='gray', dpi=200)
        plt.imsave(os.path.join(panel_dir, f"{fname}_slice{s:03d}_init.png"), init_mag.numpy(), cmap='gray', dpi=200)

        # ---- record per-slice ----
        rows.append({
            "subject": fname,
            "slice": int(s),
            "psnr_zf": psnr_zf, "ssim_zf": ssim_zf, "nmse_zf": nmse_zf,
            "psnr_trained": psnr_tr, "ssim_trained": ssim_tr, "nmse_trained": nmse_tr,
            "psnr_init": psnr_init, "ssim_init": ssim_init, "nmse_init": nmse_init,
        })

        # accumulate per-subject
        if fname not in subject_rows:
            subject_rows[fname] = []
        subject_rows[fname].append(rows[-1])

        slice_counter += 1

# ----------------- save metrics -----------------
df_slices = pd.DataFrame(rows)
df_slices.sort_values(by=["subject","slice"], inplace=True)
df_slices.to_csv(os.path.join(save_dir, "metrics_slices.csv"), index=False)

# per subject mean±std
subj_recs = []
for subj, lst in subject_rows.items():
    df = pd.DataFrame(lst)
    def stat_pair(cols):
        v = df[cols].mean(), df[cols].std()
        return float(v[0]), float(v[1])
    for tag in ["zf", "trained", "init"]:
        m_psnr, s_psnr = stat_pair(f"psnr_{tag}")
        m_ssim, s_ssim = stat_pair(f"ssim_{tag}")
        m_nmse, s_nmse = stat_pair(f"nmse_{tag}")
    # 写三行（每个方法一行）
    subj_recs.append({"subject":subj, "method":"zf", "psnr_mean":m_psnr, "psnr_std":s_psnr,
                      "ssim_mean":m_ssim, "ssim_std":s_ssim, "nmse_mean":m_nmse, "nmse_std":s_nmse})
    subj_recs.append({"subject":subj, "method":"trained", "psnr_mean":stat_pair("psnr_trained")[0], "psnr_std":stat_pair("psnr_trained")[1],
                      "ssim_mean":stat_pair("ssim_trained")[0], "ssim_std":stat_pair("ssim_trained")[1],
                      "nmse_mean":stat_pair("nmse_trained")[0], "nmse_std":stat_pair("nmse_trained")[1]})
    subj_recs.append({"subject":subj, "method":"init", "psnr_mean":stat_pair("psnr_init")[0], "psnr_std":stat_pair("psnr_init")[1],
                      "ssim_mean":stat_pair("ssim_init")[0], "ssim_std":stat_pair("ssim_init")[1],
                      "nmse_mean":stat_pair("nmse_init")[0], "nmse_std":stat_pair("nmse_init")[1]})

df_subj = pd.DataFrame(subj_recs)
df_subj.to_csv(os.path.join(save_dir, "metrics_subjects.csv"), index=False)

# overall mean
overall = {}
for tag in ["zf","trained","init"]:
    overall[f"psnr_{tag}_mean"] = float(df_slices[f"psnr_{tag}"].mean())
    overall[f"ssim_{tag}_mean"] = float(df_slices[f"ssim_{tag}"].mean())
    overall[f"nmse_{tag}_mean"] = float(df_slices[f"nmse_{tag}"].mean())
pd.DataFrame([overall]).to_csv(os.path.join(save_dir, "metrics_overall.csv"), index=False)

print(f"✅ Done. Outputs saved under: {os.path.abspath(save_dir)}")

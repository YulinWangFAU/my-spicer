# -*- coding: utf-8 -*-
"""
SPICER test (slice-level)
- Save 4 images per subject (limited): GT, ZF, model_init, model_trained
- Export slice-level & subject-level metrics (CSV + curves)
"""
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from datetime import datetime

from networks.SPICER_fastmri_network import SPNet
from dataset.pmri_fastmri_brain_lunwen import RealMeasurement
from utils.util import *
from utils.measures import *

# 固定随机种子（可复现）
SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ACCELERATION = 8
SAVE_TOPK_FIGS_PER_SUBJ = 1  # 每个subject保存几套“四图”

# 设备
local_rank = int(os.environ.get("SLURM_PROCID", 0))
device = f'cuda:{local_rank % max(1, torch.cuda.device_count())}' if torch.cuda.is_available() else 'cpu'
print(f"🧠 Using device: {device}")

def forward_once(model, y, mask):
    y_m = y * mask
    y_in = y_m.squeeze(1) if y_m.shape[1] == 1 else y_m
    ny = y_m.shape[-2]
    ACS = ((ny // 2) - (int(ny * 0.2 * (2 / ACCELERATION)) // 2)) * 2
    out, smap = model(torch.view_as_real(y_in), mask, ACS_center=(ny // 2), ACS_size=ACS)
    return out, smap, y_m

def save_four_images(save_dir, x_hat, y_m, out_init, out_trained, tag):
    gt = normlize(torch.abs(x_hat.squeeze())).cpu().numpy()
    zf = normlize(torch.abs(ifft2c(y_m).squeeze())).cpu().numpy()
    pred_init = normlize(complex_abs(out_init.detach().cpu().squeeze())).cpu().numpy()
    pred_curr = normlize(complex_abs(out_trained.detach().cpu().squeeze())).cpu().numpy()
    os.makedirs(save_dir, exist_ok=True)
    plt.imsave(os.path.join(save_dir, f"{tag}_1_gt.png"), gt, cmap='gray')
    plt.imsave(os.path.join(save_dir, f"{tag}_2_undersampled_zf.png"), zf, cmap='gray')
    plt.imsave(os.path.join(save_dir, f"{tag}_3_model_init.png"), pred_init, cmap='gray')
    plt.imsave(os.path.join(save_dir, f"{tag}_4_model_trained.png"), pred_curr, cmap='gray')

def evaluate(model_trained, model_init, dataset, save_dir):
    model_trained.eval()
    model_init.eval()

    slice_rows = []
    subj_bucket = {}
    saved_count = {}

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Testing"):
            x_hat, smps_hat, y, mask_m, mask_n = dataset[i]
            x_hat = x_hat.to(device); y = y.to(device)
            mask = mask_m.byte().to(device)

            # subject key
            ret_i, s_i = dataset._RealMeasurement__index_maps[i]
            subject_key = os.path.basename(ret_i['x_hat']).replace(".h5", "")

            # forward
            out_init, _, y_m = forward_once(model_init,    y, mask)
            out_tr,   _, _  = forward_once(model_trained, y, mask)

            # 图像域 & 中心裁剪（作者 400x320）
            out_mag = normlize(center_crop(complex_abs(out_tr)[0], [400, 320])).squeeze()
            tgt_mag = normlize(center_crop(torch.abs(x_hat.squeeze()), [400, 320])).squeeze()

            psnr = float(compare_psnr(out_mag, tgt_mag))
            ssim = float(compare_ssim(out_mag[None, None], tgt_mag[None, None]))
            nmse = float(compare_nmse(out_mag, tgt_mag))
            slice_rows.append({"index": i, "subject": subject_key, "psnr": psnr, "ssim": ssim, "nmse": nmse})
            subj_bucket.setdefault(subject_key, []).append((psnr, ssim, nmse))

            # 保存“四图”（每个 subject 限量保存）
            saved_count.setdefault(subject_key, 0)
            if saved_count[subject_key] < SAVE_TOPK_FIGS_PER_SUBJ:
                save_four_images(os.path.join(save_dir, "fig4"),
                                 x_hat, y_m, out_init, out_tr,
                                 tag=f"{subject_key}_slice{s_i:03d}")
                saved_count[subject_key] += 1

    # slice-level CSV & 曲线
    df_slice = pd.DataFrame(slice_rows)
    df_slice.to_csv(os.path.join(save_dir, "metrics_slices.csv"), index=False)
    for name in ["psnr", "ssim", "nmse"]:
        plt.figure(); plt.plot(df_slice[name].values, marker='o'); plt.title(f"{name.upper()} per Slice")
        plt.xlabel("Slice idx"); plt.ylabel(name.upper()); plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{name}_curve_slices.png")); plt.close()

    # subject-level 聚合
    rows = []
    for subj, vals in subj_bucket.items():
        arr = np.array(vals)
        rows.append({
            "subject": subj,
            "psnr_mean": float(arr[:,0].mean()), "psnr_std": float(arr[:,0].std(ddof=1)) if len(arr)>1 else 0.,
            "ssim_mean": float(arr[:,1].mean()), "ssim_std": float(arr[:,1].std(ddof=1)) if len(arr)>1 else 0.,
            "nmse_mean": float(arr[:,2].mean()), "nmse_std": float(arr[:,2].std(ddof=1)) if len(arr)>1 else 0.,
            "num_slices": int(len(arr))
        })
    df_subj = pd.DataFrame(rows).sort_values("subject")
    df_subj.to_csv(os.path.join(save_dir, "metrics_subjects.csv"), index=False)

    for name in ["psnr_mean", "ssim_mean", "nmse_mean"]:
        plt.figure(); plt.plot(df_subj[name].values, marker='o'); plt.title(f"{name.upper()} per Subject")
        plt.xlabel("Subject idx"); plt.ylabel(name.upper()); plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{name}_curve_subjects.png")); plt.close()

    print("✅ Subject-level summary:")
    print(f"  PSNR mean = {df_subj['psnr_mean'].mean():.4f} ± {df_subj['psnr_mean'].std(ddof=1):.4f}")
    print(f"  SSIM mean = {df_subj['ssim_mean'].mean():.4f} ± {df_subj['ssim_mean'].std(ddof=1):.4f}")
    print(f"  NMSE mean = {df_subj['nmse_mean'].mean():.4f} ± {df_subj['nmse_mean'].std(ddof=1):.4f}")

if __name__ == "__main__":
    # 测试集：20 subjects（作者）
    test_set = RealMeasurement(range(709, 729), acceleration_rate=ACCELERATION,
                               is_return_y_smps_hat=True,
                               mask_pattern='uniformly_cartesian',
                               smps_hat_method='eps')

    # 路径（改成你训练的输出目录）
    best_path = "/path/to/best_model.pth"   # ← 改这里：训练好的 best
    init_path = "/path/to/init_model.pth"   # ← 改这里：训练脚本保存的 init_model.pth

    # 模型
    model_trained = SPNet(num_cascades=8, pools=4, chans=18, sens_pools=4, sens_chans=8).to(device)
    model_trained.load_state_dict(torch.load(best_path, map_location=device))
    model_init = SPNet(num_cascades=8, pools=4, chans=18, sens_pools=4, sens_chans=8).to(device)
    model_init.load_state_dict(torch.load(init_path, map_location=device))
    print(f"🔍 Loaded trained from {best_path}")
    print(f"🔍 Loaded init    from {init_path}")

    # 输出目录
    root = "recon_results"; os.makedirs(root, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    outdir = os.path.join(root, f"test_job{job}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)

    with torch.no_grad():
        evaluate(model_trained, model_init, test_set, save_dir=outdir)

    print(f"✅ Results saved to: {os.path.abspath(outdir)}")

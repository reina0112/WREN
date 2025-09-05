#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retinex E2E  —  Evaluation script (fixed-fixed) with metrics summary
==============================================
Usage:
    python allUNete2e_test.py --data dataset/test \
                              --weights retinex_e2e_best.pth \
                              --out outputs_test [--use-ema]
"""
from __future__ import annotations
from pathlib import Path
import argparse
import math
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from matplotlib.image import imsave
import torch.nn.functional as F

# モデル定義と EMA（学習スクリプトから import）
from allUNete2e5 import RetinexAttnUNet, EnhancementAttnUNet, pad16, EMA

# SSIM 計算用
from pytorch_msssim import ssim


class PairDatasetTest(Dataset):
    """
    Class for loading paired test image datasets.
    If an error occurs, the filename and error message are recorded in a CSV file,
    and None is returned instead of the sample.
    """
    # ★★★ log_fileを引数に追加 ★★★
    def __init__(self, root: str | Path, log_file: Path):
        super().__init__()
        root = Path(root)
        self.low_dir = root / "low"
        self.high_dir = root / "high"
        self.log_file = log_file

        extensions = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
        self.fnames = []
        for ext in extensions:
            self.fnames.extend(p.name for p in self.low_dir.glob(ext))
        
        if not self.fnames:
            print(f"Warning: No images were found in the 'low' folder.: {self.low_dir}")

        self.fnames.sort()
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int):
        fname = self.fnames[idx]
        low_img_path = self.low_dir / fname
        high_img_path = self.high_dir / fname

        try:
            low_img = Image.open(low_img_path).convert("RGB")

            if not high_img_path.exists():
                 raise FileNotFoundError(f"c: {high_img_path}")
            high_img = Image.open(high_img_path).convert("RGB")
            
            # resize
            max_size = 1280
            if low_img.width > max_size or low_img.height > max_size:
                low_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                high_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            orig_h, orig_w = low_img.size[1], low_img.size[0]
            low_t_orig = self.to_tensor(low_img); high_t_orig = self.to_tensor(high_img)
            low_t_padded = pad16(low_t_orig); high_t_padded = pad16(high_t_orig)
            
            return low_t_padded, high_t_padded, fname, (orig_h, orig_w)

        except Exception as e:
            # ★★★ エラー処理：CSVに記録し、Noneを返す ★★★
            print(f"\nData loading error: '{fname}' will be skipped. Check the CSV log for details.")
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([fname, e])
            return None
        
def collate_fn_skip_corrupted(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


@torch.inference_mode()
def evaluate(
    dl: DataLoader,
    decomp: RetinexAttnUNet,
    enh: EnhancementAttnUNet,
    device: torch.device | str,
    out_dir: Path,
):
    decomp.eval(); enh.eval()
    # --- 出力フォルダを準備 ---
    out_recon = out_dir / "recon"
    out_high  = out_dir / "high"
    out_low   = out_dir / "low"
    out_recon.mkdir(parents=True, exist_ok=True)
    out_high.mkdir(parents=True, exist_ok=True)
    out_low.mkdir(parents=True, exist_ok=True)

    psnr_list, ssim_list, mae_list, mse_list = [], [], [], []

    for batch_data in dl:
        if batch_data is None:
            continue
        
        lo, hi, names, sizes = batch_data
        lo, hi = lo.to(device), hi.to(device)
        R, L, _   = decomp(lo)
        Lh     = enh(torch.cat([lo, L, R], 1))
        recon  = (R * Lh).clamp(0.0, 1.0)

        hs, ws = sizes
        for i, fname in enumerate(names):
            orig_h, orig_w = int(hs[i]), int(ws[i])
            pad_h = (16 - orig_h % 16) % 16; pad_w = (16 - orig_w % 16) % 16
            pad_t = pad_h // 2; pad_l = pad_w // 2

            pred = recon[i, :, pad_t : pad_t + orig_h, pad_l : pad_l + orig_w]
            gt   = hi[i,    :, pad_t : pad_t + orig_h, pad_l : pad_l + orig_w]
            lo_i = lo[i,    :, pad_t : pad_t + orig_h, pad_l : pad_l + orig_w]

            mse_val = F.mse_loss(pred, gt).item()
            mse_list.append(mse_val)
            psnr_val = 10.0 * math.log10(1.0 / (mse_val + 1e-9))
            psnr_list.append(psnr_val)
            ssim_val = ssim(gt.unsqueeze(0), pred.unsqueeze(0), data_range=1.0).item()
            ssim_list.append(ssim_val)
            mae_val = F.l1_loss(pred, gt).item()
            mae_list.append(mae_val)

            # --- 保存処理 ---
            rec_img = pred.detach().cpu().permute(1,2,0).numpy()
            gt_img  = gt.detach().cpu().permute(1,2,0).numpy()
            lo_img  = lo_i.detach().cpu().permute(1,2,0).numpy()

            imsave(out_recon / fname.replace(".jpg", "_recon.png"), rec_img)
            imsave(out_high  / fname.replace(".jpg", "_high.png"),  gt_img)
            imsave(out_low   / fname.replace(".jpg", "_low.png"),   lo_img)

            print(f"{fname}: PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}  MAE={mae_val:.4f}")


    if not psnr_list:
        print("\nNo valid images could be processed."
)
        return

    import numpy as np
    print("\n--- Results ---")
    print(f"The number of images: {len(psnr_list)}")
    print(f"Average PSNR: {np.mean(psnr_list):.3f} dB")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")
    print(f"Average MSE : {np.mean(mse_list):.6f}")
    print(f"Average MAE : {np.mean(mae_list):.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    required=True, help="test folder (low / high)")
    ap.add_argument("--weights", required=True, help="retinex_e2e_best.pth")
    ap.add_argument("--out",     default="outputs_test", help="save folder")
    ap.add_argument("--device",  default="cuda", help="cpu / cuda")
    ap.add_argument("--use-ema", action="store_true", help="Apply EMA weights for eval")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ★★★ CSV ★★★
    log_file = Path("corrupted_files.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "error_message"])
    print(f"Error log will be saved to {log_file}.")

    # ★★★ DataLoader ★★★
    dataset = PairDatasetTest(args.data, log_file=log_file)
    dl = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=collate_fn_skip_corrupted
    )

    ckpt = torch.load(args.weights, map_location=device)
    decomp = RetinexAttnUNet().to(device)
    enh    = EnhancementAttnUNet().to(device)
    decomp.load_state_dict(ckpt["decomp"])
    enh.load_state_dict(ckpt["enh"])

    if args.use_ema and "ema_shadow_decomp" in ckpt:
        ema_dec = EMA(decomp); ema_dec.shadow = ckpt["ema_shadow_decomp"]; ema_dec.apply_to(decomp)
        print("Applied EMA weights for Decomp model.")
    if args.use_ema and "ema_shadow_enh" in ckpt:
        ema_enh = EMA(enh); ema_enh.shadow = ckpt["ema_shadow_enh"]; ema_enh.apply_to(enh)
        print("Applied EMA weights for Enh model.")

    evaluate(dl, decomp, enh, device, Path(args.out))

if __name__ == "__main__":
    main()

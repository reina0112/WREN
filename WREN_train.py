#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WREN
PyTorch 1.13+
"""

import random
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from matplotlib.image import imsave
import kornia.color as kc
from torchvision.models import vgg16
import torchvision.transforms.functional as TF
from pytorch_msssim import ms_ssim

# ------------------------------------------------------------
# 1. utility
# ------------------------------------------------------------
def scale_invariant_log_loss(pred, target, mask=None):
    eps  = 1e-6
    d    = torch.log(pred.clamp_min(eps)) - torch.log(target.clamp_min(eps))
    if mask is not None:
        d = d * mask
        N = mask.sum()
    else:
        N = d.numel()
    return (d.pow(2).sum()/N) - (d.sum()/N)**2

def tv_iso(x):
    dx = x[..., 1:, :] - x[..., :-1, :]
    dy = x[..., :, 1:] - x[..., :, :-1]
    dx_c = dx[..., :, :-1]
    dy_c = dy[..., :-1, :]
    return torch.sqrt(dx_c.pow(2)+dy_c.pow(2)+1e-6).sum() / x.size(0)

def gradient_magnitude(img):
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],
                           dtype=img.dtype, device=img.device).view(1,1,3,3)/8.0
    sobel_y = sobel_x.transpose(2,3)
    gx = F.conv2d(img, sobel_x.repeat(img.size(1),1,1,1),
                  padding=1, groups=img.size(1))
    gy = F.conv2d(img, sobel_y.repeat(img.size(1),1,1,1),
                  padding=1, groups=img.size(1))
    return torch.sqrt(gx.pow(2)+gy.pow(2)+1e-6).mean(1,keepdim=True)

# ---------- SSIM ----------
def _gaussian_window(ch: int, size: int = 11, sigma: float = 1.5):
    coords = torch.arange(size).float() - size//2
    g = torch.exp(-(coords**2)/(2*sigma**2))
    g = (g/g.sum()).unsqueeze(0)
    window = g.t() @ g
    window = window.unsqueeze(0).unsqueeze(0)
    return window.repeat(ch,1,1,1)

def ssim(img1: torch.Tensor, img2: torch.Tensor,
         window_size: int = 11, sigma: float = 1.5, C1: float = 0.01**2, C2: float = 0.03**2):
    ch = img1.size(1)
    window = _gaussian_window(ch, window_size, sigma).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=ch)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=ch)
    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1*mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=ch) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=ch) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding=window_size//2, groups=ch) - mu12
    ssim_map = ((2*mu12 + C1)*(2*sigma12 + C2)) / ((mu1_sq+mu2_sq + C1)*(sigma1_sq+sigma2_sq + C2))
    return ssim_map.mean()

def lab_ab_loss(pred, gt):
    p_lab = kc.rgb_to_lab(pred)[:,1:]   # a,b
    g_lab = kc.rgb_to_lab(gt)[:,1:]
    return F.l1_loss(p_lab, g_lab)

class AttentionBlock(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g,  F_int, 1, bias=False),
                                 nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False),
                                 nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)            # (B,1,H,W)
        return x * psi                 

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.relu  = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        shortcut = self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.relu(out + shortcut)
        return out

class AttentionDecoder(nn.Module):
    def __init__(self, out_ch, base_ch=128):
        super().__init__()
        self.up1   = nn.ConvTranspose2d(base_ch*8, base_ch*8, 2, stride=2)
        self.att1  = AttentionBlock(base_ch*8, base_ch*8,  base_ch*4)
        self.conv1 = ResidualBlock(base_ch*16, base_ch*4)

        self.up2   = nn.ConvTranspose2d(base_ch*4, base_ch*4, 2, stride=2)
        self.att2  = AttentionBlock(base_ch*4, base_ch*4,  base_ch*2)
        self.conv2 = ResidualBlock(base_ch*8,  base_ch*2)

        self.up3   = nn.ConvTranspose2d(base_ch*2, base_ch*2, 2, stride=2)
        self.att3  = AttentionBlock(base_ch*2, base_ch*2,  base_ch)
        self.conv3 = ResidualBlock(base_ch*4,  base_ch)

        self.up4   = nn.ConvTranspose2d(base_ch,   base_ch,   2, stride=2)
        self.att4  = AttentionBlock(base_ch, base_ch, base_ch//2)
        self.conv4 = ResidualBlock(base_ch*2,  base_ch)

        self.out   = nn.Sequential(nn.Conv2d(base_ch, out_ch, 1), nn.Sigmoid())

    def forward(self, feats: List[torch.Tensor]):
        x0,x1,x2,x3,x4 = feats
        g = self.up1(x4);   x3 = self.att1(g, x3)
        g = self.conv1(torch.cat([g, x3], 1))

        g = self.up2(g);     x2 = self.att2(g, x2)
        g = self.conv2(torch.cat([g, x2], 1))

        g = self.up3(g);     x1 = self.att3(g, x1)
        g = self.conv3(torch.cat([g, x1], 1))

        g = self.up4(g);     x0 = self.att4(g, x0)
        g = self.conv4(torch.cat([g, x0], 1))

        return self.out(g)

# ------------------------------------------------------------
# 2. Network ― Stage-1
# ------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, base_ch=128):
        super().__init__()
        self.inc   = ResidualBlock(3, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_ch,   base_ch*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_ch*2, base_ch*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_ch*4, base_ch*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_ch*8, base_ch*8))
    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0,x1,x2,x3,x4]

class RetinexAttnUNet(nn.Module):
    def __init__(self, base_ch=128):
        super().__init__()
        self.encoder   = Encoder(base_ch)
        self.decoder_R = AttentionDecoder(3, base_ch)
        self.decoder_L = AttentionDecoder(1, base_ch)

    def forward(self, x):
        feats = self.encoder(x)
        R = self.decoder_R(feats)
        x0,x1,x2,x3,x4 = feats
        g4 = self.decoder_L.up1(x4);   x3_att = self.decoder_L.att1(g4, x3)
        g3 = self.decoder_L.conv1(torch.cat([g4, x3_att], 1))
        g3_up = self.decoder_L.up2(g3);  x2_att = self.decoder_L.att2(g3_up, x2)
        g2 = self.decoder_L.conv2(torch.cat([g3_up, x2_att], 1))
        g2_up = self.decoder_L.up3(g2);  x1_att = self.decoder_L.att3(g2_up, x1)
        g1 = self.decoder_L.conv3(torch.cat([g2_up, x1_att], 1))
        g1_up = self.decoder_L.up4(g1);  x0_att = self.decoder_L.att4(g1_up, x0)
        g0 = self.decoder_L.conv4(torch.cat([g1_up, x0_att], 1))
        L = self.decoder_L.out(g0)
        return R, L, [g2, g1, g0]

# ------------------------------------------------------------
# 3. Stage-1 Loss
# ------------------------------------------------------------
class RetinexLoss(nn.Module):
    def __init__(self, epsilon=1.0, lambda_rec=1.0,
                 lambda_tv_R=0, lambda_tv_WL=0):
        super().__init__()
        self.epsilon, self.lambda_rec = epsilon, lambda_rec
        self.lambda_tv_R, self.lambda_tv_WL = lambda_tv_R, lambda_tv_WL
    def forward(self, I_low, I_high, R, L):
        sR = scale_invariant_log_loss(R, I_high)
        sL = scale_invariant_log_loss(L, I_high.max(1,keepdim=True).values)
        rec= (I_low - R*L).pow(2).mean()
        tvR= tv_iso(R)
        W  = torch.exp(-self.epsilon*gradient_magnitude(I_low))
        tvWL= tv_iso(W*L)
        total = 2*sR+2*sL + self.lambda_rec*rec + self.lambda_tv_R*tvR + self.lambda_tv_WL*tvWL
        return total, {"sR":sR.detach(),"sL":sL.detach(),"rec":rec.detach()}

class GuidedFilterLayer(nn.Module):
    def __init__(self, ch_guide:int, ch_target:int, radius:int=1, eps:float=1e-4, ch_reduce:int=4):
        super().__init__()
        self.reduce_g = nn.Conv2d(ch_guide,  ch_reduce, 1, bias=False)
        self.reduce_t = nn.Conv2d(ch_target, ch_reduce, 1, bias=False)
        self.up_a = nn.Conv2d(ch_reduce, ch_target, 1, bias=False)
        self.up_b = nn.Conv2d(ch_reduce, ch_target, 1, bias=False)
        self.radius = radius
        self.eps    = nn.Parameter(torch.tensor(eps))
        k = 2 * radius + 1
        self.register_buffer("box_kernel", torch.ones(1, 1, k, k) / (k * k))

    def _boxfilter(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.box_kernel.expand(x.size(1), -1, -1, -1)
        return F.conv2d(x, kernel, padding=self.radius, groups=x.size(1))

    def forward(self, F_enc: torch.Tensor, F_dec: torch.Tensor) -> torch.Tensor:
        B, _, H, W = F_enc.shape
        Fg_lr = F.interpolate(self.reduce_g(F_enc), scale_factor=0.25,
                              mode='bilinear', align_corners=False)
        Ft_lr = F.interpolate(self.reduce_t(F_dec), scale_factor=0.25,
                              mode='bilinear', align_corners=False)
        mean_g = self._boxfilter(Fg_lr)
        mean_t = self._boxfilter(Ft_lr)
        var_g  = self._boxfilter(Fg_lr * Fg_lr) - mean_g * mean_g
        cov_gt = self._boxfilter(Fg_lr * Ft_lr) - mean_g * mean_t
        a = cov_gt / (var_g + self.eps)
        b = mean_t - a * mean_g
        a = F.interpolate(a, size=(H, W), mode='bilinear', align_corners=False)
        b = F.interpolate(b, size=(H, W), mode='bilinear', align_corners=False)
        return F_dec + self.up_a(a) * F_enc + self.up_b(b)

# ------------------------------------------------------------
# 4. Dataset
# ------------------------------------------------------------
class PairDataset(Dataset):
    """random crop + Synchronized horizontal flip"""
    def __init__(self, root: Union[str, Path], crop: int = 384):
        super().__init__()
        self.crop = crop
        root      = Path(root)
        self.low_files  = sorted((root / "low").glob("*.png"))
        self.high_files = [ (root / "high" / f.name.replace("_low", "_high"))
                            for f in self.low_files ]
        self.to_tensor = transforms.ToTensor()
        self.hflip     = transforms.RandomHorizontalFlip(0.5)

    def __len__(self) -> int:
        return len(self.low_files)

    def _rand_coords(self, w: int, h: int) -> Tuple[int, int]:
        if w < self.crop or h < self.crop:
            return 0, 0
        x = random.randint(0, w - self.crop)
        y = random.randint(0, h - self.crop)
        return x, y

    def __getitem__(self, idx):
        p_low  = self.low_files[idx]
        p_high = self.high_files[idx]
        low_img  = Image.open(p_low ).convert("RGB")
        high_img = Image.open(p_high).convert("RGB")
        w, h = low_img.size
        if w >= self.crop and h >= self.crop:
            x, y = self._rand_coords(w, h)
            low_img  = TF.crop(low_img , y, x, self.crop, self.crop)
            high_img = TF.crop(high_img, y, x, self.crop, self.crop)
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed); low_img  = self.hflip(low_img)
        torch.manual_seed(seed); high_img = self.hflip(high_img)
        low_t  = self.to_tensor(low_img)
        high_t = self.to_tensor(high_img)
        return pad16(low_t), pad16(high_t), (low_img.height, low_img.width)

def pad16(t: torch.Tensor):
    _, h, w = t.shape
    ph, pw = (16-h%16)%16, (16-w%16)%16
    pad_l = pw // 2; pad_r = pw - pad_l
    pad_t = ph // 2; pad_b = ph - pad_t
    return F.pad(t, (pad_l, pad_r, pad_t, pad_b), mode='reflect') if (ph or pw) else t

# ------------------------------------------------------------
# 5. Stage-2 Enhancement UNet
# ------------------------------------------------------------
class BasicTransformerBlock(nn.Module):
    """(B,C,H,W) -> (B,C,H,W)"""
    def __init__(self, dim, n_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        shortcut = x
        x = x.view(b, c, h * w).permute(2, 0, 1)  # (HW,B,C)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        x = x.permute(1, 2, 0).view(b, c, h, w)
        return x + shortcut

class EnhancementAttnUNet(nn.Module):
    def __init__(self, in_ch=7, base_ch=128):
        super().__init__()
        # --- Encoder ---
        self.enc1  = ResidualBlock(in_ch, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_ch,   base_ch*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_ch*2, base_ch*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_ch*4, base_ch*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_ch*8, base_ch*8))
        # --- Transformer bottleneck ---
        self.transformer_bottleneck = nn.Sequential(
            BasicTransformerBlock(dim=base_ch*8, n_heads=8),
            BasicTransformerBlock(dim=base_ch*8, n_heads=8),
            BasicTransformerBlock(dim=base_ch*8, n_heads=8),
            BasicTransformerBlock(dim=base_ch*8, n_heads=8)
        )
        # --- Decoder (Attention + GuidedFilter + Conv) ---
        self.up1   = nn.ConvTranspose2d(base_ch*8, base_ch*8, 2, stride=2)
        self.att1  = AttentionBlock(base_ch*8, base_ch*8, base_ch*4)
        self.gf1   = GuidedFilterLayer(ch_guide=base_ch*8, ch_target=base_ch*8)
        self.conv1 = ResidualBlock(base_ch*16, base_ch*4)

        self.up2   = nn.ConvTranspose2d(base_ch*4, base_ch*4, 2, stride=2)
        self.att2  = AttentionBlock(base_ch*4, base_ch*4, base_ch*2)
        self.gf2   = GuidedFilterLayer(ch_guide=base_ch*4, ch_target=base_ch*4)
        self.conv2 = ResidualBlock(base_ch*8,  base_ch*2)

        self.up3   = nn.ConvTranspose2d(base_ch*2, base_ch*2, 2, stride=2)
        self.att3  = AttentionBlock(base_ch*2, base_ch*2, base_ch)
        self.gf3   = GuidedFilterLayer(ch_guide=base_ch*2, ch_target=base_ch*2)
        self.conv3 = ResidualBlock(base_ch*4,  base_ch)

        self.up4   = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.att4  = AttentionBlock(base_ch, base_ch, base_ch//2)
        self.gf4   = GuidedFilterLayer(ch_guide=base_ch, ch_target=base_ch)
        self.conv4 = ResidualBlock(base_ch*2, base_ch)

        self.out = nn.Sequential(nn.Conv2d(base_ch, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # --- Encoder ---
        x0 = self.enc1(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        # --- Transformer ---
        x4 = self.transformer_bottleneck(x4)
        # --- Decoder ---
        g  = self.up1(x4);  x3a= self.att1(g, x3); g = self.gf1(x3a, g); g = self.conv1(torch.cat([g, x3a], dim=1))
        g  = self.up2(g);   x2a= self.att2(g, x2); g = self.gf2(x2a, g); g = self.conv2(torch.cat([g, x2a], dim=1))
        g  = self.up3(g);   x1a= self.att3(g, x1); g = self.gf3(x1a, g); g = self.conv3(torch.cat([g, x1a], dim=1))
        g  = self.up4(g);   x0a= self.att4(g, x0); g = self.gf4(x0a, g); g = self.conv4(torch.cat([g, x0a], dim=1))
        return self.out(g)

@torch.no_grad()
def _evaluate(dec, enh, dl, device):
    dec.eval(); enh.eval()
    tot_psnr, tot_ssim, cnt = 0.0, 0.0, 0
    for lo, hi, _ in dl:
        lo, hi = lo.to(device), hi.to(device)
        R, L, _ = dec(lo)
        Lh = enh(torch.cat([lo, L, R], 1))
        hat = (R * Lh).clamp(0.0, 1.0)
        mse = F.mse_loss(hat, hi)
        psnr_val = 10 * torch.log10(1.0 / mse).item()
        ssim_val = ssim(hat, hi).item()
        tot_psnr += psnr_val * lo.size(0)
        tot_ssim += ssim_val * lo.size(0)
        cnt += lo.size(0)
    return tot_psnr / max(cnt, 1e-8), tot_ssim / max(cnt, 1e-8)

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        self.decay = decay
        self.shadow = {}
        self.device = device
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone().to(device or p.device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad: 
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1-d)

    def apply_to(self, model: nn.Module):
        backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad: 
                continue
            backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: dict):
        for name, p in model.named_parameters():
            if not p.requires_grad: 
                continue
            p.data.copy_(backup[name])

# ------------------------------------------------------------
# 6. End-to-End
# ------------------------------------------------------------
def train_end_to_end(root, *, val_root=None,
                     epochs=150, batch=8, lr=1e-4,
                     lambda_mse=2.0, lambda_ssim=0.5,
                     device='cuda'):
    out=Path("outputs_WREN"); out.mkdir(exist_ok=True)
    dl = DataLoader(PairDataset(root), batch, True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(PairDataset(val_root), batch, False, num_workers=4, pin_memory=True) if val_root is not None else None

    dec, enh = RetinexAttnUNet().to(device), EnhancementAttnUNet().to(device)
    accumulation_steps = 4
    loss_dec = RetinexLoss()
    opt = torch.optim.AdamW(list(dec.parameters())+list(enh.parameters()), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(dl), epochs=epochs)
    best=-1.0
    ema_dec  = EMA(dec,  decay=0.995)
    ema_enh  = EMA(enh,  decay=0.995)

    vgg = vgg16(weights="IMAGENET1K_V1").features[:16].eval().to(device)
    for p in vgg.parameters(): p.requires_grad_(False)

    for ep in range(1,epochs+1):
        dec.train(); enh.train(); tot=0.0
        opt.zero_grad()
        for i,(lo,hi,hw) in enumerate(dl):
            lo,hi=lo.to(device),hi.to(device)
            R, L, _ = dec(lo)
            Lh = enh(torch.cat([lo, L, R], 1))
            recon = R * Lh

            # loss
            l_perc = F.l1_loss(vgg(recon*2-1), vgg(hi*2-1))
            l_lab  = lab_ab_loss(recon, hi)
            ms_ssim_loss = 1.0 - ms_ssim(hi, recon, data_range=1.0, size_average=True)
            mse_loss = F.mse_loss(recon, hi)
            gradient_loss = F.l1_loss(gradient_magnitude(recon), gradient_magnitude(hi))
            l_bright = F.relu(Lh - 1.0).mean()
            l_dec, _ = loss_dec(lo, hi, R, L)

            # weight
            if ep > 350:
                loss = (1.0 * l_dec +
                        (2.0 * mse_loss + 1.0 * ms_ssim_loss + 0.0 * gradient_loss))
            else:
                loss = (1.0 * l_dec +
                        (10.0 * mse_loss + 1.0 * ms_ssim_loss + 0.0 * gradient_loss +
                         0.5 * l_perc + 0.5 * l_lab + 0.5 * l_bright))

            
            loss = loss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(list(dec.parameters())+list(enh.parameters()), 1.0)
                opt.step()
                opt.zero_grad()
                scheduler.step()

            tot += loss.item() * lo.size(0) * accumulation_steps
            ema_dec.update(dec); ema_enh.update(enh)
            
            # Intermediate output image
            # if ep % 1 == 0 and i == 0:
            #     heights, widths = hw
            #     oh = int(heights[0].item()); ow = int(widths[0].item())
            #     rec_img = recon[0, :, :oh, :ow].detach().cpu().clamp(0, 1).permute(1,2,0).numpy()
            #     Lh_img  = Lh   [0, :, :oh, :ow].detach().cpu().clamp(0, 1).squeeze().numpy()
            #     R_img   = R    [0, :, :oh, :ow].detach().cpu().clamp(0, 1).permute(1,2,0).numpy()
            #     imsave(out / f"ep{ep:03d}_recon.png", rec_img)
            #     imsave(out / f"ep{ep:03d}_Lh.png",   Lh_img, cmap="gray", vmin=0, vmax=1)
            #     imsave(out / f"ep{ep:03d}_R.png",    R_img)

        epoch_loss = tot / len(dl.dataset)
        msg = f"Ep{ep:03d} train-loss={epoch_loss:.4f}"

        if dl_val is not None:
            bdec = ema_dec.apply_to(dec); benh = ema_enh.apply_to(enh)
            psnr_v, ssim_v = _evaluate(dec, enh, dl_val, device)
            ema_dec.restore(dec, bdec); ema_enh.restore(enh, benh)
            msg += f" | val-PSNR(EMA)={psnr_v:.2f}  SSIM(EMA)={ssim_v:.4f}"
            if psnr_v > best:
                best = psnr_v
                bdec = ema_dec.apply_to(dec); benh = ema_enh.apply_to(enh)
                torch.save({"decomp": dec.state_dict(), "enh": enh.state_dict()},
                           "retinex_e2e_best_test_gf.pth")
                ema_dec.restore(dec, bdec); ema_enh.restore(enh, benh)

        print(msg)

    # final save
    bdec = ema_dec.apply_to(dec); benh = ema_enh.apply_to(enh)
    torch.save({"decomp": dec.state_dict(), "enh": enh.state_dict()},
               "retinex_e2e_final_test_gf.pth")
    ema_dec.restore(dec, bdec); ema_enh.restore(enh, benh)
    print("✓ training finished – final weights saved -> retinex_e2e_final_test_gf.pth")

# ------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_end_to_end(
        root="dataset/train",
        val_root="dataset/testlol",
        epochs=500, batch=4, lr=3e-4,
        lambda_mse=10.0, lambda_ssim=0.2,
        device=device
    )

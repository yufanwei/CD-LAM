"""Synthetic camera transforms for LAM v1 same-z finetune.

All transforms operate on float tensors (B, 3, H, W) in [0, 1], output same shape.
Uses F.grid_sample with reflection padding -> no black borders, no wrap-around.
align_corners=False matches the LAM input pipeline (F.interpolate(... align_corners=False)).
"""
from __future__ import annotations
import math
import random
from typing import Dict

import torch
import torch.nn.functional as F


def _affine_grid(theta: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """theta: (B, 2, 3) affine matrix. Returns sampling grid (B, H, W, 2)."""
    return F.affine_grid(theta, size=(theta.shape[0], 3, H, W), align_corners=False)


def _apply(img: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """img: (B, 3, H, W) float. theta: (B, 2, 3) affine matrix in normalized coords."""
    H, W = img.shape[-2:]
    grid = _affine_grid(theta, H, W)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def identity_theta(B: int, device, dtype=torch.float32) -> torch.Tensor:
    th = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    th[:, 0, 0] = 1.0
    th[:, 1, 1] = 1.0
    return th


def shift_theta(dx_px: torch.Tensor, dy_px: torch.Tensor, H: int, W: int, device) -> torch.Tensor:
    """dx_px / dy_px: (B,) tensor of pixel shifts (positive = right / down).

    grid_sample uses normalized coords [-1, +1]. With align_corners=False, output pixel (i, j)
    samples input at center (i+0.5, j+0.5) mapped to the [-1,+1] range. To shift the OUTPUT
    image right by dx pixels, the sampling grid must read from x - dx, i.e. theta[:, 0, 2] = -2*dx/W.
    """
    B = dx_px.shape[0]
    th = identity_theta(B, device, dtype=torch.float32)
    th[:, 0, 2] = -2.0 * dx_px.to(device).float() / W
    th[:, 1, 2] = -2.0 * dy_px.to(device).float() / H
    return th


def zoom_theta(scale: torch.Tensor, device) -> torch.Tensor:
    """scale: (B,) tensor. >1 zooms in (object larger), <1 zooms out. Center-preserving."""
    B = scale.shape[0]
    th = identity_theta(B, device, dtype=torch.float32)
    s = (1.0 / scale.to(device).float())  # sample compressed grid -> output zoomed
    th[:, 0, 0] = s
    th[:, 1, 1] = s
    return th


def rotation_theta(angle_deg: torch.Tensor, device) -> torch.Tensor:
    """angle_deg: (B,) tensor of degrees. Center-preserving rotation."""
    B = angle_deg.shape[0]
    a = angle_deg.to(device).float() * math.pi / 180.0
    cos = torch.cos(a)
    sin = torch.sin(a)
    th = torch.zeros(B, 2, 3, device=device, dtype=torch.float32)
    th[:, 0, 0] = cos
    th[:, 0, 1] = -sin
    th[:, 1, 0] = sin
    th[:, 1, 1] = cos
    return th


def sample_transform_metadata(
    B: int,
    distribution: Dict[str, float],
    shift_pixels=(1, 2, 3),
    zoom_scales=(0.985, 0.99, 1.01, 1.015),
    rotation_degrees=(1, 2),
    perspective_pixels=(2, 4),
    blur_sigmas=(0.5, 0.8),
    brightness_pcts=(0.05, 0.10),
    seed: int | None = None,
) -> Dict:
    """Sample per-batch-item transform descriptors. Returns dict of per-item tensors.

    distribution: keys in {'identity','shift','zoom','rotation','perspective','blur','brightness'}.
    Type codes:
      0=id, 1=shift, 2=zoom, 3=rot, 4=perspective, 5=blur (o_j only), 6=brightness (o_j only)
    """
    rng = random.Random(seed) if seed is not None else random
    types = []
    dx = torch.zeros(B)
    dy = torch.zeros(B)
    zoom = torch.ones(B)
    rot = torch.zeros(B)
    persp = torch.zeros(B)
    blur = torch.zeros(B)
    bright = torch.zeros(B)

    keys = ["identity", "shift", "zoom", "rotation", "perspective", "blur", "brightness"]
    cum = []
    s = 0.0
    for k in keys:
        s += float(distribution.get(k, 0.0))
        cum.append(s)

    for i in range(B):
        u = rng.random() * cum[-1]
        if u < cum[0]:
            tcode = 0
        elif u < cum[1]:
            tcode = 1
        elif u < cum[2]:
            tcode = 2
        elif u < cum[3]:
            tcode = 3
        elif u < cum[4]:
            tcode = 4
        elif u < cum[5]:
            tcode = 5
        else:
            tcode = 6
        types.append(tcode)

        if tcode == 1:
            mag = rng.choice(shift_pixels) * (1 if rng.random() < 0.5 else -1)
            if rng.random() < 0.5:
                dx[i] = mag
            else:
                dy[i] = mag
        elif tcode == 2:
            zoom[i] = rng.choice(zoom_scales)
        elif tcode == 3:
            mag = rng.choice(rotation_degrees) * (1 if rng.random() < 0.5 else -1)
            rot[i] = mag
        elif tcode == 4:
            mag = rng.choice(perspective_pixels) * (1 if rng.random() < 0.5 else -1)
            persp[i] = mag
        elif tcode == 5:
            blur[i] = rng.choice(blur_sigmas)
        elif tcode == 6:
            bright[i] = rng.choice(brightness_pcts) * (1 if rng.random() < 0.5 else -1)

    return {
        "type": torch.tensor(types, dtype=torch.long),
        "dx": dx, "dy": dy, "zoom": zoom, "rot_deg": rot,
        "persp_px": persp, "blur_sigma": blur, "bright_pct": bright,
    }


def apply_transform(img: torch.Tensor, meta: Dict) -> torch.Tensor:
    """img: (B, 3, H, W) float. meta from sample_transform_metadata. Returns transformed (B,3,H,W).

    All-GPU implementation. For type ∈ {identity, shift, zoom, rotation, perspective}, builds a
    single per-item homography (affine for 0-3, perspective row added for 4) + one grid_sample.
    For type ∈ {blur, brightness}, applies the photometric op AFTER grid_sample (selectively).
    """
    B, _, H, W = img.shape
    device = img.device

    typ = meta["type"].to(device)
    dx = meta["dx"].to(device).float()
    dy = meta["dy"].to(device).float()
    zoom = meta["zoom"].to(device).float()
    rot_rad = meta["rot_deg"].to(device).float() * (math.pi / 180.0)
    persp = meta.get("persp_px", torch.zeros(B)).to(device).float()
    blur = meta.get("blur_sigma", torch.zeros(B)).to(device).float()
    bright = meta.get("bright_pct", torch.zeros(B)).to(device).float()

    one = torch.ones_like(zoom)
    scale = torch.where(typ == 2, 1.0 / zoom, one)
    angle = torch.where(typ == 3, rot_rad, torch.zeros_like(rot_rad))
    cos = torch.cos(angle) * scale
    sin = torch.sin(angle) * scale
    is_shift = (typ == 1).float()
    tx = -2.0 * (dx * is_shift) / W
    ty = -2.0 * (dy * is_shift) / H

    theta = torch.zeros(B, 2, 3, device=device, dtype=torch.float32)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    out = _apply(img, theta)

    # Perspective (type==4): warp corners by `persp_px`. Implemented per-item.
    persp_idx = (typ == 4).nonzero(as_tuple=False).squeeze(-1)
    if persp_idx.numel() > 0:
        out_p = out.clone()
        for k_i in persp_idx.tolist():
            m = float(persp[k_i].item())
            startpoints = [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]
            endpoints = [[m, m], [W - 1 - m, m], [W - 1 - m, H - 1 - m], [m, H - 1 - m]]
            try:
                from torchvision.transforms.v2.functional import perspective as tv_persp
                out_p[k_i:k_i + 1] = tv_persp(
                    out[k_i:k_i + 1], startpoints=startpoints, endpoints=endpoints,
                    interpolation=2, fill=None,
                )
            except Exception:
                pass
        out = out_p

    # Blur (type==5): per-item Gaussian blur with sigma=blur[i]. Apply only on selected items.
    blur_idx = (typ == 5).nonzero(as_tuple=False).squeeze(-1)
    if blur_idx.numel() > 0:
        try:
            from torchvision.transforms.v2.functional import gaussian_blur
            out_b = out.clone()
            for k_i in blur_idx.tolist():
                sig = float(blur[k_i].item())
                if sig > 0:
                    ksize = max(3, int(sig * 4) | 1)
                    out_b[k_i:k_i + 1] = gaussian_blur(out[k_i:k_i + 1], kernel_size=[ksize, ksize], sigma=[sig, sig])
            out = out_b
        except Exception:
            pass

    # Brightness (type==6): multiply by (1 + bright_pct).
    bright_idx = (typ == 6).nonzero(as_tuple=False).squeeze(-1)
    if bright_idx.numel() > 0:
        out_br = out.clone()
        factor = (1.0 + bright[bright_idx]).view(-1, 1, 1, 1)
        out_br[bright_idx] = (out[bright_idx] * factor).clamp(0, 1)
        out = out_br

    return out


def apply_specific(img: torch.Tensor, kind: str, magnitude: float) -> torch.Tensor:
    """Used for evaluation: apply one specific transform to entire batch.

    kind: 'identity'|'shift_x'|'shift_y'|'zoom'|'rotation'
    magnitude: pixels (shift), scale (zoom), degrees (rotation), ignored for identity.
    """
    B, _, H, W = img.shape
    device = img.device
    if kind == "identity":
        return img
    elif kind == "shift_x":
        th = shift_theta(torch.full((B,), magnitude), torch.zeros(B), H, W, device)
    elif kind == "shift_y":
        th = shift_theta(torch.zeros(B), torch.full((B,), magnitude), H, W, device)
    elif kind == "zoom":
        th = zoom_theta(torch.full((B,), magnitude), device)
    elif kind == "rotation":
        th = rotation_theta(torch.full((B,), magnitude), device)
    else:
        raise ValueError(f"unknown transform kind: {kind}")
    return _apply(img, th)

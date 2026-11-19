"""Per-frame and per-segment quality metrics (PSNR / SSIM).

Declared conventions (recorded in the ledger):
  - pixel range: full [0,255] 8-bit planes, BT.601 matrix (see colors.py)
  - channel weights: configurable, default Y=0.8 U=0.1 V=0.1
  - chroma downsampling: 2x2 box average for 4:2:0 (see colors.box_downsample2)
  - SSIM: 11x11 Gaussian window, sigma=1.5, K1=0.01, K2=0.03, edge padding
  - PSNR is capped at 99.0 dB when MSE == 0 (declared, keeps JSON finite)
"""
from __future__ import annotations

import numpy as np

from . import util

PSNR_CAP = 99.0


def psnr_plane(ref: np.ndarray, dec: np.ndarray) -> float:
    if ref.shape != dec.shape:
        raise ValueError(f"plane shape mismatch: {ref.shape} vs {dec.shape}")
    diff = ref.astype(np.float64) - dec.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return PSNR_CAP
    return 10.0 * np.log10(255.0 * 255.0 / mse)


def _gauss1d(size: int, sigma: float) -> np.ndarray:
    x = np.arange(size, dtype=np.float64) - (size // 2)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def _filter1d(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    pad = len(k) // 2
    pads = [(0, 0), (0, 0)]
    pads[axis] = (pad, pad)
    p = np.pad(img, pads, mode="edge")
    out = np.zeros_like(img, dtype=np.float64)
    for i, w in enumerate(k):
        sl = [slice(None), slice(None)]
        sl[axis] = slice(i, i + img.shape[axis])
        out += w * p[tuple(sl)]
    return out


def _gauss2d(img: np.ndarray, size: int, sigma: float) -> np.ndarray:
    k = _gauss1d(size, sigma)
    return _filter1d(_filter1d(img, k, 0), k, 1)


def ssim_plane(ref: np.ndarray, dec: np.ndarray, size: int = 11, sigma: float = 1.5) -> float:
    if ref.shape != dec.shape:
        raise ValueError(f"plane shape mismatch: {ref.shape} vs {dec.shape}")
    x = ref.astype(np.float64)
    y = dec.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mux = _gauss2d(x, size, sigma)
    muy = _gauss2d(y, size, sigma)
    xx = _gauss2d(x * x, size, sigma) - mux * mux
    yy = _gauss2d(y * y, size, sigma) - muy * muy
    xy = _gauss2d(x * y, size, sigma) - mux * muy
    num = (2.0 * mux * muy + c1) * (2.0 * xy + c2)
    den = (mux * mux + muy * muy + c1) * (xx + yy + c2)
    return float(np.mean(num / den))


def frame_metrics(ref_planes, dec_planes, weights, ssim_cfg) -> dict:
    psnr = {ch: psnr_plane(ref_planes[ch], dec_planes[ch]) for ch in ("Y", "U", "V")}
    wsum = weights["Y"] + weights["U"] + weights["V"]
    weighted = sum(psnr[ch] * weights[ch] for ch in ("Y", "U", "V")) / wsum
    return {
        "psnr_y": util.round6(psnr["Y"]),
        "psnr_u": util.round6(psnr["U"]),
        "psnr_v": util.round6(psnr["V"]),
        "psnr_weighted": util.round6(weighted),
        "ssim_y": util.round6(ssim_plane(ref_planes["Y"], dec_planes["Y"],
                                         ssim_cfg["window"], ssim_cfg["sigma"])),
    }


def segment_aggregate(per_frame, src_indices, shots) -> list:
    """Aggregate per-frame metrics by shot. Boundaries come from the ground
    truth shot list itself (never inferred from record counts)."""
    by_src = {s: m for s, m in zip(src_indices, per_frame)}
    segments = []
    for shot in shots:
        present = [by_src[i] for i in range(shot["start"], shot["end"]) if i in by_src]
        missing = [i for i in range(shot["start"], shot["end"]) if i not in by_src]
        seg = {
            "shot": shot["index"],
            "start": shot["start"],
            "end": shot["end"],
            "frames_present": len(present),
            "frames_missing": missing,
        }
        if present:
            for key in ("psnr_y", "psnr_weighted", "ssim_y"):
                vals = [m[key] for m in present]
                seg[f"{key}_mean"] = util.round6(float(np.mean(vals)))
                seg[f"{key}_min"] = util.round6(float(np.min(vals)))
        segments.append(seg)
    return segments

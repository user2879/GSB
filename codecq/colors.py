"""Deterministic RGB <-> YUV conversion (BT.601, full range, declared in ledger)."""
from __future__ import annotations

import numpy as np

MATRIX_BT601_FULL = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ]
)

SUPPORTED_PIX_FMTS = ("yuv420p", "yuv444p")


def rgb_to_yuv444(rgb: np.ndarray):
    """uint8 (H,W,3) RGB -> (Y,U,V) uint8 planes, BT.601 full range, rint rounding."""
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3).astype(np.float64)
    yuv = flat @ MATRIX_BT601_FULL.T
    yuv[:, 1:] += 128.0
    yuv = np.clip(np.rint(yuv), 0, 255).astype(np.uint8)
    return yuv[:, 0].reshape(h, w), yuv[:, 1].reshape(h, w), yuv[:, 2].reshape(h, w)


def box_downsample2(plane: np.ndarray) -> np.ndarray:
    """2x2 box average downsample (declared chroma downsampling for 4:2:0)."""
    p = plane.astype(np.float64)
    d = (p[0::2, 0::2] + p[1::2, 0::2] + p[0::2, 1::2] + p[1::2, 1::2]) / 4.0
    return np.rint(d).astype(np.uint8)


def rgb_to_planes(rgb: np.ndarray, pix_fmt: str):
    """Return {'Y','U','V'} planes at the native resolution of pix_fmt."""
    y, u, v = rgb_to_yuv444(rgb)
    if pix_fmt == "yuv444p":
        return {"Y": y, "U": u, "V": v}
    if pix_fmt == "yuv420p":
        return {"Y": y, "U": box_downsample2(u), "V": box_downsample2(v)}
    raise ValueError(f"unsupported pix_fmt: {pix_fmt}")


def stack_planes(planes, pix_fmt: str) -> np.ndarray:
    """Stack planes into the single ndarray layout PyAV from_ndarray expects."""
    y, u, v = planes["Y"], planes["U"], planes["V"]
    h, w = y.shape
    if pix_fmt == "yuv444p":
        return np.stack([y, u, v], axis=0)  # PyAV yuv444p layout: (3, H, W)
    if pix_fmt == "yuv420p":
        return np.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)]).reshape(
            h + h // 2, w
        )
    raise ValueError(f"unsupported pix_fmt: {pix_fmt}")


def unstack_planes(arr: np.ndarray, pix_fmt: str):
    """Inverse of stack_planes for decoded frames."""
    if pix_fmt == "yuv444p":
        return {"Y": arr[0].copy(), "U": arr[1].copy(), "V": arr[2].copy()}
    if pix_fmt == "yuv420p":
        h = (arr.shape[0] * 2) // 3
        w = arr.shape[1]
        y = arr[0:h, :].copy()
        chroma = arr[h:, :].reshape(-1)
        half = chroma.size // 2
        return {
            "Y": y,
            "U": chroma[:half].reshape(h // 2, w // 2).copy(),
            "V": chroma[half:].reshape(h // 2, w // 2).copy(),
        }
    raise ValueError(f"unsupported pix_fmt: {pix_fmt}")

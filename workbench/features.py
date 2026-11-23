"""Per-window spectral feature extraction (float64, deterministic)."""
from __future__ import annotations

import numpy as np


def _local_range_3x3(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="edge")
    acc_max = np.full_like(gray, -np.inf)
    acc_min = np.full_like(gray, np.inf)
    for dr in range(3):
        for dc in range(3):
            window = padded[dr:dr + gray.shape[0], dc:dc + gray.shape[1]]
            acc_max = np.maximum(acc_max, window)
            acc_min = np.minimum(acc_min, window)
    return acc_max - acc_min


def compute_features(bands: dict) -> dict:
    """bands: dict with R,G,B,ELEV float64-convertible arrays of equal shape."""
    red = bands["R"].astype(np.float64)
    green = bands["G"].astype(np.float64)
    blue = bands["B"].astype(np.float64)
    elev = bands["ELEV"].astype(np.float64)
    veg = (green - red) / (green + red + 1.0)
    water = (blue - green) / (blue + green + 1.0)
    gray = (red + green + blue) / 3.0
    texture = _local_range_3x3(gray) / 64.0
    grad_r, grad_c = np.gradient(elev)
    slope = np.hypot(grad_r, grad_c) / 32.0
    return {"veg": veg, "water": water, "texture": texture, "slope": slope}

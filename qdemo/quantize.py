"""Per-layer quantization / dequantization core.

Symmetric and asymmetric schemes, optional per-channel scales, group
granularity along the input dimension (or the flattened tensor when
per-channel scaling is off). Scales are computed in float64 and rounded to a
configured number of significant digits (a recorded rounding parameter).
"""
from __future__ import annotations

import numpy as np

from .common import BusinessError, E_CONFIG_INVALID, array_fingerprint, round_sig


def _group_view(w: np.ndarray, group_size: int, per_channel: bool) -> np.ndarray:
    if per_channel:
        out_f, in_f = w.shape
        if in_f % group_size != 0:
            raise BusinessError(
                E_CONFIG_INVALID,
                f"group_size {group_size} does not divide input dim {in_f}",
            )
        return w.reshape(out_f, in_f // group_size, group_size)
    flat = w.reshape(-1)
    if flat.size % group_size != 0:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"group_size {group_size} does not divide tensor size {flat.size}",
        )
    return flat.reshape(1, flat.size // group_size, group_size)


def quantize_dequantize(
    w: np.ndarray,
    bits: int,
    group_size: int,
    scheme: str,
    per_channel: bool,
    sig_digits: int,
) -> dict:
    """Quantize then dequantize one layer; return params + error summary."""
    w = np.asarray(w, dtype=np.float64)
    g = _group_view(w, group_size, per_channel)

    if scheme == "symmetric":
        qmax = float(2 ** (bits - 1) - 1)
        maxabs = np.max(np.abs(g), axis=-1, keepdims=True)
        scale = np.where(maxabs == 0.0, 1.0, maxabs / qmax)
        scale = round_sig(scale, sig_digits)
        zero_point = np.zeros_like(scale)
        q = np.clip(np.round(g / scale), -qmax, qmax)
        deq = q * scale
        qmin, qmax_i = -int(qmax), int(qmax)
    elif scheme == "asymmetric":
        qmax = float(2**bits - 1)
        mn = np.min(g, axis=-1, keepdims=True)
        mx = np.max(g, axis=-1, keepdims=True)
        span = mx - mn
        scale = np.where(span == 0.0, 1.0, span / qmax)
        scale = round_sig(scale, sig_digits)
        zero_point = np.clip(np.round(-mn / scale), 0.0, qmax)
        q = np.clip(np.round(g / scale) + zero_point, 0.0, qmax)
        deq = (q - zero_point) * scale
        qmin, qmax_i = 0, int(qmax)
    else:
        raise BusinessError(E_CONFIG_INVALID, f"unknown scheme {scheme!r}")

    dequant = deq.reshape(w.shape)
    err = dequant - w
    return {
        "scheme": scheme,
        "bits": bits,
        "group_size": group_size,
        "per_channel": per_channel,
        "qmin": qmin,
        "qmax": qmax_i,
        "scale_sig_digits": sig_digits,
        "scales": scale.reshape(-1),
        "zero_points": zero_point.reshape(-1).astype(np.int64),
        "dequant": dequant,
        "dequant_digest": array_fingerprint(dequant),
        "error": {
            "max_abs": float(np.max(np.abs(err))),
            "mse": float(np.mean(err**2)),
        },
    }


def quantize_both_schemes(w, bits, group_size, per_channel, sig_digits) -> dict:
    """Run both schemes so their error summaries can be compared side by side."""
    return {
        s: quantize_dequantize(w, bits, group_size, s, per_channel, sig_digits)
        for s in ("symmetric", "asymmetric")
    }

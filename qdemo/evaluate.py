"""Calibration-set evaluation and lightweight distillation correction."""
from __future__ import annotations

import numpy as np


def forward(weighted_layers, x: np.ndarray) -> dict:
    """Sequential linear model: tanh activation on all but the last layer.

    weighted_layers: list of (name, W) with W of shape (out, in).
    Returns {layer_name: output} with outputs of shape (samples, out).
    """
    acts = np.asarray(x, dtype=np.float64)
    outs = {}
    last = len(weighted_layers) - 1
    for i, (name, w) in enumerate(weighted_layers):
        acts = acts @ np.asarray(w, dtype=np.float64).T
        if i != last:
            acts = np.tanh(acts)
        outs[name] = acts
    return outs


def error_stats(student: np.ndarray, teacher: np.ndarray) -> dict:
    diff = np.asarray(student, dtype=np.float64) - np.asarray(teacher, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "mse": float(np.mean(diff**2)),
        "max_abs": float(np.max(np.abs(diff))),
    }


def per_sample_errors(student: np.ndarray, teacher: np.ndarray) -> list:
    diff = np.asarray(student, dtype=np.float64) - np.asarray(teacher, dtype=np.float64)
    return [float(v) for v in np.sqrt(np.mean(diff**2, axis=1))]


def softmax_temperature(z: np.ndarray, temperature: float) -> np.ndarray:
    zt = np.asarray(z, dtype=np.float64) / temperature
    zt = zt - np.max(zt, axis=1, keepdims=True)
    e = np.exp(zt)
    return e / np.sum(e, axis=1, keepdims=True)


def distill_correct(teacher: np.ndarray, student: np.ndarray, temperature: float, alpha: float) -> dict:
    """One gradient step on the temperature-scaled soft-label KL objective.

    s_corr = s - alpha * T**2 * (softmax(s/T) - softmax(t/T))

    Large alpha * T**2 can overshoot; the caller compares errors and marks
    the layer unstable when the correction increases the error.
    """
    t = np.asarray(teacher, dtype=np.float64)
    s = np.asarray(student, dtype=np.float64)
    p = softmax_temperature(t, temperature)
    q = softmax_temperature(s, temperature)
    corrected = s - alpha * temperature**2 * (q - p)
    before = error_stats(s, t)
    after = error_stats(corrected, t)
    correction = corrected - s
    return {
        "corrected": corrected,
        "error_before": before,
        "error_after": after,
        "stable": bool(after["mse"] <= before["mse"]),
        "per_sample_correction_l2": [
            float(v) for v in np.sqrt(np.sum(correction**2, axis=1))
        ],
    }

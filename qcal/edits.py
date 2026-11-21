"""Declarative edit helpers for deterministic incremental experiments."""

from __future__ import annotations

from typing import Any

import numpy as np


def replacement_weights(*, layer: str, shape: tuple[int, ...], seed: int, scale: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, float(scale), size=shape)
    return np.asarray(values, dtype=np.float64)


def load_edit_sequence(path: str | Any) -> list[dict[str, Any]]:
    from .io_utils import load_yaml

    payload = load_yaml(path)
    if isinstance(payload, dict):
        steps = payload.get("steps", [])
    elif isinstance(payload, list):
        steps = payload
    else:
        steps = []
    return [step for step in steps if isinstance(step, dict)]

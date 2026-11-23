"""Shared deterministic test fixtures."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from hotblock_diag.config import (FixedPointConfig, PrefilterConfig,
                                  validate_layout)
from hotblock_diag.blocks import build_blocks
from hotblock_diag.errors import FailureList
from hotblock_diag.injection import InjectionPlan
from hotblock_diag.tensorpack import write_pack


def make_prefilter(**overrides) -> PrefilterConfig:
    data = dict(q_block=4, k_block=4, candidates=2, score_floor_ratio=0.5,
                temperature=1.25, padding="last_zero")
    data.update(overrides)
    return PrefilterConfig(**data)


def make_fixedpoint(**overrides) -> FixedPointConfig:
    data = dict(frac_bits=8, quant_gear="q8", acc_bits=32,
                saturation="saturate", rounding="nearest_even",
                hot_capacity=4, hot_keep=3, max_rounds=6, threads=3,
                batch_budget=16, qblock_quota=8, snapshot_interval=1,
                tolerance=1e-9)
    data.update(overrides)
    return FixedPointConfig(**data)


def make_pack(directory: str, seq_q: int = 16, seq_k: int = 16,
              dim: int = 8, seed: int = 777) -> tuple[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    tensors = {
        "query": rng.normal(0, 1, (seq_q, dim)).astype(np.float32),
        "key": rng.normal(0, 0.8, (seq_k, dim)).astype(np.float32),
        "value": rng.normal(0, 0.6, (seq_k, dim)).astype(np.float32),
    }
    path = os.path.join(directory, "pack.bin")
    write_pack(path, tensors)
    return path, tensors


def layout_for(tensors: dict[str, np.ndarray], pre: PrefilterConfig,
               failures: FailureList | None = None):
    failures = failures or FailureList()
    return validate_layout(
        tensors["query"].shape[0], tensors["key"].shape[0],
        tensors["value"].shape[0], tensors["query"].shape[1],
        tensors["key"].shape[1], tensors["value"].shape[1], pre, failures,
    )


def blocks_for(tensors, layout, pre):
    return build_blocks(tensors, layout, pre)


def temp_workspace() -> str:
    return tempfile.mkdtemp(prefix="hbdiag-")


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, sort_keys=True, indent=2)
        handle.write("\n")

"""持久化快照：临时文件 + 原子替换，仅完整替换的快照可被恢复。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SNAPSHOT_JSON = "snapshot.json"
BLOCKS_NPZ = "blocks.npz"


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"不可序列化类型: {type(obj)!r}")


def save_snapshot(state_dir, state, c_matrix):
    """先写临时文件再原子替换；snapshot.json 最后落位作为提交标记。"""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp_npz = state_dir / (BLOCKS_NPZ + ".tmp")
    tmp_json = state_dir / (SNAPSHOT_JSON + ".tmp")
    with open(tmp_npz, "wb") as f:
        np.savez(f, C=np.asarray(c_matrix, dtype=np.float32))
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, sort_keys=True, default=_json_default)
    os.replace(tmp_npz, state_dir / BLOCKS_NPZ)
    os.replace(tmp_json, state_dir / SNAPSHOT_JSON)


def load_snapshot(state_dir):
    """加载最近一次完整提交的快照；缺失或半成品（含残留 tmp）一律忽略。"""
    state_dir = Path(state_dir)
    json_path = state_dir / SNAPSHOT_JSON
    npz_path = state_dir / BLOCKS_NPZ
    if not json_path.exists() or not npz_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    with np.load(npz_path) as z:
        c_matrix = z["C"].copy()
    return state, c_matrix

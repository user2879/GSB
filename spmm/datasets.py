"""本地确定性样本生成与加载：三种非零元分布的 CSR 与稠密矩阵。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .csr import DISTRIBUTIONS, CSRMatrix, generate_csr

SAMPLE_M = 64
SAMPLE_K = 48
SAMPLE_N = 32
SEEDS = {"uniform": 20241101, "powerlaw": 20241102, "extreme": 20241103}


def generate_all(data_dir, m=SAMPLE_M, k=SAMPLE_K, n=SAMPLE_N):
    """按固定种子生成三种分布的样本，幂等覆盖写入。"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    metas = {}
    for dist in DISTRIBUTIONS:
        seed = SEEDS[dist]
        a = generate_csr(m, k, dist, seed)
        rng = np.random.default_rng(seed + 1)
        b = rng.standard_normal((k, n)).astype(np.float32)
        c = rng.standard_normal((m, n)).astype(np.float32)
        np.savez(
            data_dir / f"{dist}.npz",
            indptr=a.indptr, indices=a.indices, values=a.values,
            a_shape=np.asarray(a.shape, dtype=np.int64), B=b, C=c,
        )
        metas[dist] = {
            "distribution": dist,
            "seed": seed,
            "M": m, "K": k, "N": n,
            "nnz": a.nnz,
            "row_nnz_min": int(np.diff(a.indptr).min()),
            "row_nnz_max": int(np.diff(a.indptr).max()),
        }
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    return metas


def load_sample(data_dir, distribution):
    """加载指定分布的 (A, B, C, meta)。"""
    data_dir = Path(data_dir)
    path = data_dir / f"{distribution}.npz"
    if not path.exists():
        generate_all(data_dir)
    with np.load(path) as z:
        a = CSRMatrix(
            indptr=z["indptr"].copy(),
            indices=z["indices"].copy(),
            values=z["values"].copy(),
            shape=tuple(int(x) for x in z["a_shape"]),
        )
        b = z["B"].copy()
        c = z["C"].copy()
    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)[distribution]
    return a, b, c, meta


def random_row(rng, k, max_nnz=None):
    """生成一个确定性的随机 CSR 行 (cols, vals)，列索引严格递增。"""
    if max_nnz is None:
        max_nnz = max(1, k // 3)
    cnt = int(rng.integers(1, max_nnz + 1))
    cols = np.sort(rng.choice(k, size=cnt, replace=False)).astype(np.int64)
    vals = rng.standard_normal(cnt).astype(np.float32)
    return cols, vals

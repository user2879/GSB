"""本地确定性样本生成：清单 v1/v2/v3、节点注册表、样本数值数据。

样本数据由键名 sha256 派生种子经 NumPy PCG64 生成，完全离线、
跨次运行逐字节一致。
"""

import hashlib
import json
import os

import numpy as np

SERIES_LEN = 64
ALL_KEYS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta",
            "eta", "rho", "sigma"]


def generate_series(key, n=SERIES_LEN):
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8],
                          "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    return [int(x) for x in rng.integers(-5000, 5001, size=n)]


def _write(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")


def generate_samples(out_dir):
    """生成全部样本文件，返回文件路径字典。"""
    os.makedirs(out_dir, exist_ok=True)
    mdir = os.path.join(out_dir, "manifests")
    os.makedirs(mdir, exist_ok=True)

    data = {"series": {k: generate_series(k) for k in ALL_KEYS}}
    data_path = os.path.join(out_dir, "data.json")
    _write(data_path, data)

    registry = {
        "nodes": [
            {"id": "node-a", "port": 39211, "view": "s2", "profile": "sum"},
            {"id": "node-b", "port": 39212, "view": "s1", "profile": "mean"},
            {"id": "node-c", "port": 39213, "view": "s2", "profile": "wsum"},
            {"id": "node-d", "port": 39214, "view": "s2", "profile": "sum"},
        ]
    }
    registry_path = os.path.join(out_dir, "registry.json")
    _write(registry_path, registry)

    v1 = {
        "format": "v1",
        "entries": [
            {"key": "alpha", "value": 1500},
            {"key": "beta", "value": -250},
            {"key": "gamma", "value": "enabled"},
            {"key": "delta", "value": True},
        ],
    }
    _write(os.path.join(mdir, "app_v1.json"), v1)

    v2 = {
        "format": "v2",
        "manifest_version": 3,
        "entries": [
            {"key": "alpha", "value": 1500, "targets": ["node-a", "node-b"]},
            {"key": "beta", "value": -250, "targets": ["*"]},
            {"key": "gamma", "value": "enabled", "targets": ["node-c"]},
            {"key": "delta", "value": True, "targets": ["node-d"]},
            {"key": "epsilon", "value": 42, "targets": ["node-a", "node-c"]},
        ],
    }
    _write(os.path.join(mdir, "app_v2.json"), v2)

    v3 = {
        "format": "v3",
        "manifest_version": 5,
        "strategy": {"timeout_ms": 800, "max_retries": 2,
                     "global_concurrency": 3, "node_quota": 2},
        "entries": [
            {"key": "alpha", "value": 1500, "targets": ["*"], "priority": 30},
            {"key": "beta", "value": -250, "targets": ["node-a", "node-b"],
             "priority": 10},
            {"key": "gamma", "value": "enabled", "targets": ["node-c"],
             "priority": 30},
            {"key": "delta", "value": True, "targets": ["node-d"],
             "priority": 20},
            {"key": "epsilon", "value": 42, "targets": ["node-a", "node-c"],
             "priority": 10},
            {"key": "zeta", "value": 7, "targets": ["*"], "priority": 0},
        ],
        "derived": [
            {"key": "rho", "op": "mul", "inputs": ["alpha", "beta"],
             "targets": ["node-a", "node-c"], "priority": 40},
            {"key": "sigma", "op": "sub", "inputs": ["epsilon", "zeta"],
             "targets": ["node-b"], "priority": 5},
        ],
    }
    _write(os.path.join(mdir, "app_v3.json"), v3)

    return {"dir": out_dir, "manifest_dir": mdir, "data": data_path,
            "registry": registry_path}


def load_data(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["series"]

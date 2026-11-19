"""固定种子下按用途派生独立子种子。

派生路径（账本 seed_tree 逐条记录，可独立复算）：
  master(seed)
    └─ purpose  (sha256(f"{seed}|{purpose}|{region_id?}|{index?}") 低 64 位)

每个用途使用独立 numpy PCG64 发生器，互不共享流；
所有随机决策只来自这些发生器，不依赖 dict 遍历顺序或线程调度。
"""

from __future__ import annotations

import hashlib

import numpy as np


PURPOSES = (
    "terrain_choice",       # 图块地形类别抽样
    "elevation_field",      # 逐区域连续高程场
    "density_field",        # 逐区域连续密度场
    "resource_placement",   # 资源点位置抽样
)


def derive_subseed(master_seed: int, purpose: str, region_id: str | None = None,
                   index: int | None = None) -> int:
    """确定性派生 64 位子种子。"""
    material = f"{int(master_seed)}|{purpose}"
    if region_id is not None:
        material += f"|r={region_id}"
    if index is not None:
        material += f"|i={int(index)}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def make_rng(subseed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed=int(subseed)))


def build_seed_tree(master_seed: int, region_ids: list[str]) -> dict:
    """生成账本可追溯的子种子派生树（区域按 id 排序）。"""
    tree: dict = {"master_seed": int(master_seed), "algorithms": {
        "rng": "numpy.PCG64",
        "derive": "sha256('seed|purpose[|r=region][|i=index]')[:8] big-endian",
    }, "streams": {}}
    streams = tree["streams"]
    streams["terrain_choice"] = {
        "subseed": derive_subseed(master_seed, "terrain_choice"),
        "path": f"{master_seed}|terrain_choice",
    }
    streams["resource_placement"] = {
        "subseed": derive_subseed(master_seed, "resource_placement"),
        "path": f"{master_seed}|resource_placement",
    }
    for purpose in ("elevation_field", "density_field"):
        per_region = {}
        for idx, rid in enumerate(sorted(region_ids)):
            per_region[rid] = {
                "subseed": derive_subseed(master_seed, purpose, rid, idx),
                "path": f"{master_seed}|{purpose}|r={rid}|i={idx}",
            }
        streams[purpose] = per_region
    return tree

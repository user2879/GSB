"""体素级全局点云合并。

合并按视图序号串行处理（并发仅存在于上游深度阶段），同一体素内的点按
量化坐标稳定排序后取中位代表点；体素键直接由 voxel_size 派生，
修改体素尺寸会真实改变体素划分与代表点选取。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import ReconstructionConfig
from .util import content_fingerprint, fingerprint_many, quantize_scalar


def _voxel_key(xyz_q: list[int], frac_bits: int, voxel_size: float) -> tuple[int, int, int]:
    scale = 1 << frac_bits
    voxel_q = voxel_size * scale
    return tuple(int(np.floor(value / voxel_q)) for value in xyz_q)


def merge_point_cloud(
    view_ids: list[int],
    match_artifacts: list[dict],
    cfg: ReconstructionConfig,
) -> dict[str, Any]:
    frac_bits = cfg.fixed_point_frac_bits
    accepted = [a for a in match_artifacts if a["content"].get("accepted")]
    accepted.sort(key=lambda item: tuple(int(part) for part in item["pair_id"].split("-")[1:]))

    # 点去重（同 point_key 只处理一次），记录引用视图 / 匹配对
    points_by_key: dict[str, dict[str, Any]] = {}
    for artifact in accepted:
        content = artifact["content"]
        for point in content["points"]:
            entry = points_by_key.setdefault(
                point["point_key"],
                {
                    "point_key": point["point_key"],
                    "xyz_q": list(point["xyz_q"]),
                    "pair_ids": [],
                    "view_ids": set(),
                },
            )
            entry["pair_ids"].append(content["pair_id"])
            entry["view_ids"].update(point["source_view_ids"])

    # 按视图序号串行喂入，体素内保留稳定的插入顺序信息
    voxels: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for view_id in sorted(view_ids):
        for point_key in sorted(points_by_key):
            entry = points_by_key[point_key]
            if view_id not in entry["view_ids"]:
                continue
            key = _voxel_key(entry["xyz_q"], frac_bits, cfg.voxel_size)
            voxels.setdefault(key, []).append(entry)

    representatives: list[dict[str, Any]] = []
    for voxel_key in sorted(voxels):
        members = voxels[voxel_key]
        # 同体素多点：按量化坐标字典序、再按 point_key 稳定排序
        members.sort(key=lambda item: (tuple(item["xyz_q"]), item["point_key"]))
        middle = (len(members) - 1) // 2
        representative = members[middle]
        ref_views = sorted({view for item in members for view in item["view_ids"]})
        ref_pairs = sorted({pair for item in members for pair in item["pair_ids"]})
        ref_points = sorted(item["point_key"] for item in members)
        representatives.append(
            {
                "voxel_key": list(voxel_key),
                "point_key": representative["point_key"],
                "xyz_q": list(representative["xyz_q"]),
                "member_count": len(members),
                "ref_view_ids": ref_views,
                "ref_pair_ids": ref_pairs,
                "ref_point_keys": ref_points,
            }
        )

    # 点序：体素键字典序（等价于空间稳定排序），与线程调度无关
    representatives.sort(key=lambda item: (tuple(item["voxel_key"]), item["point_key"]))
    content = {
        "voxel_size_q": quantize_scalar(cfg.voxel_size, frac_bits),
        "frac_bits": frac_bits,
        "point_count": len(representatives),
        "representatives": representatives,
    }
    return {
        "artifact_id": "cloud-global",
        "stage": "cloud",
        "content": content,
        "content_fingerprint": content_fingerprint(content),
    }


def cloud_artifact_key(depth_fps: list[str], param_fp: str) -> str:
    return fingerprint_many("cloud", *sorted(depth_fps), param_fp)

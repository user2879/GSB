"""异常注入：对样本副本施加真实输入故障并生成命中清单。"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .errors import BizCode
from .utils import atomic_write_json, read_json

# 注入类型 -> 预期业务码映射
INJECTION_CODE_MAP = {
    "truncate_png": BizCode.E_PNG_TRUNCATED,
    "camera_missing": BizCode.E_CAMERA_MISSING,
    "disparity_out_of_range": BizCode.E_MIN_DISPARITY,
    "nonfinite_descriptor": BizCode.E_FEATURE_NONFINITE,
    "zero_voxel_size": BizCode.E_VOXEL_ZERO,
}


@dataclass
class InjectionHit:
    injection_id: str
    injection_type: str
    view: int
    stage: str
    expected_code: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "injection_id": self.injection_id,
            "injection_type": self.injection_type,
            "view": self.view, "stage": self.stage,
            "expected_code": self.expected_code, "detail": self.detail,
        }


def load_injection_config(path: str) -> List[Dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict):
        items = data.get("injections", [])
    else:
        items = data
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        record = dict(item)
        record.setdefault("injection_id", f"INJ{index + 1:02d}")
        if record["type"] not in INJECTION_CODE_MAP:
            raise ValueError(f"未知注入类型: {record['type']}")
        record["expected_code"] = INJECTION_CODE_MAP[record["type"]]
        normalized.append(record)
    return normalized


def apply_injections(src_sample: str, dst_sample: str,
                     injections: List[Dict[str, Any]],
                     config_overrides: Dict[str, Any] | None = None
                     ) -> List[InjectionHit]:
    """复制样本到工作副本并施加注入，返回注入命中清单。"""
    if os.path.exists(dst_sample):
        shutil.rmtree(dst_sample)
    shutil.copytree(src_sample, dst_sample)

    hits: List[InjectionHit] = []
    for item in injections:
        view = int(item["view"])
        injection_type = item["type"]
        stage = item.get("stage", "")
        hit = InjectionHit(
            injection_id=item["injection_id"],
            injection_type=injection_type, view=view, stage=stage,
            expected_code=item["expected_code"],
            detail=item.get("detail", {}),
        )
        if injection_type == "truncate_png":
            path = os.path.join(dst_sample, f"view_{view:02d}.png")
            size = os.path.getsize(path)
            keep = max(8, int(size * item.get("keep_ratio", 0.35)))
            with open(path, "r+b") as handle:
                handle.truncate(keep)
            stage = "load"
        elif injection_type == "camera_missing":
            path = os.path.join(dst_sample, f"view_{view:02d}.camera.json")
            os.remove(path)
            stage = "load"
        elif injection_type == "disparity_out_of_range":
            # 抬高该视图相机参数中的视场角至越界（真实参数故障路径），
            # 或直接标记注入，引擎在三角化阶段读取标记值
            path = os.path.join(dst_sample, f"view_{view:02d}.camera.json")
            camera = read_json(path)
            camera["injection"] = {
                "type": "disparity_out_of_range",
                "forced_disparity": float(item.get("forced_disparity", 0.0)),
            }
            atomic_write_json(path, camera)
            stage = "triangulation"
        elif injection_type == "nonfinite_descriptor":
            path = os.path.join(dst_sample, f"view_{view:02d}.camera.json")
            camera = read_json(path)
            camera["injection"] = {"type": "nonfinite_descriptor"}
            atomic_write_json(path, camera)
            stage = "feature"
        elif injection_type == "zero_voxel_size":
            stage = "point_cloud"
        hit.stage = stage
        hits.append(hit)

    if config_overrides:
        cfg_path = os.path.join(dst_sample, os.path.pardir, "config.override.json")
        atomic_write_json(os.path.abspath(cfg_path), config_overrides)
    return hits

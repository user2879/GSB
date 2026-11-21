"""样本编辑命令：替换视图图像、修改相机参数、调整重建参数。

编辑作用于工作副本，用于验证增量重算的失效范围与可审计清单。
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from .utils import atomic_write_json, read_json


@dataclass
class EditOutcome:
    command: str
    target: str
    changed_fingerprints: List[str]
    expected_invalidation: List[str]


def apply_edit_sequence(src_sample: str, dst_sample: str,
                        commands: List[Dict[str, Any]],
                        config_path: str | None = None
                        ) -> List[EditOutcome]:
    if os.path.exists(dst_sample):
        shutil.rmtree(dst_sample)
    shutil.copytree(src_sample, dst_sample)
    outcomes: List[EditOutcome] = []
    for command in commands:
        kind = command["command"]
        if kind == "replace_image":
            outcomes.append(_replace_image(dst_sample, command))
        elif kind == "edit_camera":
            outcomes.append(_edit_camera(dst_sample, command))
        elif kind == "edit_config":
            outcomes.append(_edit_config(command, config_path))
        else:
            raise ValueError(f"未知编辑命令: {kind}")
    return outcomes


def _replace_image(sample_dir: str, command: Dict[str, Any]) -> EditOutcome:
    view = int(command["view"])
    path = os.path.join(sample_dir, f"view_{view:02d}.png")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8).copy()
    # 确定性扰动：在固定像素叠加矩形，改变角点响应分布
    x = int(command.get("x", 20))
    y = int(command.get("y", 20))
    size = int(command.get("size", 14))
    arr[y:y + size, x:x + size] = np.asarray(
        command.get("color", [255, 0, 0]), dtype=np.uint8
    )
    Image.fromarray(arr, mode="RGB").save(path, format="PNG")
    return EditOutcome(
        command="replace_image", target=f"view_{view:02d}.png",
        changed_fingerprints=[f"image:view{view}"],
        expected_invalidation=[
            f"feature:view{view}",
            f"match:view{view - 1}", f"match:view{view}",
            f"depth:view{view}", "point_cloud",
        ],
    )


def _edit_camera(sample_dir: str, command: Dict[str, Any]) -> EditOutcome:
    view = int(command["view"])
    path = os.path.join(sample_dir, f"view_{view:02d}.camera.json")
    camera = read_json(path)
    field_path = command["field"].split(".")
    node: Any = camera
    for part in field_path[:-1]:
        node = node[part]
    node[field_path[-1]] = command["value"]
    atomic_write_json(path, camera)
    return EditOutcome(
        command="edit_camera", target=f"view_{view:02d}.camera.json",
        changed_fingerprints=[f"camera:view{view}"],
        expected_invalidation=[
            f"depth:view{view}",
            f"match:view{view - 1}", f"match:view{view}",
            "point_cloud",
        ],
    )


def _edit_config(command: Dict[str, Any],
                 config_path: str | None) -> EditOutcome:
    if config_path is None:
        raise ValueError("edit_config 需要 config_path")
    config = read_json(config_path)
    config[command["field"]] = command["value"]
    atomic_write_json(config_path, config)
    return EditOutcome(
        command="edit_config", target=os.path.basename(config_path),
        changed_fingerprints=[f"config:{command['field']}"],
        expected_invalidation=["feature", "match", "depth", "point_cloud"],
    )

"""检查点演进、版本校验与原子持久化。"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from .errors import E_CHECKPOINT_VERSION
from .locking import atomic_write, checkpoint_lock
from .models import (
    CHECKPOINT_STRUCT_VERSION,
    NORMALIZATION_POLICY_VERSION,
    NormalizationPolicy,
)

CHECKPOINT_FILENAME = "checkpoint.json"
LOCK_FILENAME = "checkpoint.lock"


def build_checkpoint(
    completed_files: list[dict[str, Any]],
    aggregate: dict[str, Any],
    policies: list[NormalizationPolicy],
) -> dict[str, Any]:
    return {
        "struct_version": CHECKPOINT_STRUCT_VERSION,
        "policy_version": NORMALIZATION_POLICY_VERSION,
        "policies": [p.as_dict() for p in policies],
        "completed_files": completed_files,
        "aggregate": aggregate,
    }


def save_checkpoint(directory: str, checkpoint: dict[str, Any]) -> str:
    os.makedirs(directory, exist_ok=True)
    lock_path = os.path.join(directory, LOCK_FILENAME)
    target = os.path.join(directory, CHECKPOINT_FILENAME)
    with checkpoint_lock(lock_path):
        atomic_write(target, json.dumps(checkpoint, ensure_ascii=False, sort_keys=True, indent=2))
    return target


def load_checkpoint(
    directory: str, policies: list[NormalizationPolicy]
) -> Optional[dict[str, Any]]:
    """加载检查点；结构/策略版本变化时返回整体重建信号。"""
    target = os.path.join(directory, CHECKPOINT_FILENAME)
    if not os.path.exists(target):
        return None
    try:
        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        # 强制终止等场景可能留下截断/不可读检查点：
        # 视为无有效检查点，调用方整体重建，绝不基于残缺状态续接。
        return None
    if data.get("struct_version") != CHECKPOINT_STRUCT_VERSION:
        data["_rebuild_reason"] = E_CHECKPOINT_VERSION
        return data
    if data.get("policy_version") != NORMALIZATION_POLICY_VERSION:
        data["_rebuild_reason"] = E_CHECKPOINT_VERSION
        return data
    expected = sorted((p.as_dict() for p in policies), key=lambda d: d["name"])
    actual = sorted(data.get("policies", []), key=lambda d: d.get("name", ""))
    if actual != expected:
        data["_rebuild_reason"] = E_CHECKPOINT_VERSION
    return data


def should_rebuild(checkpoint: Optional[dict[str, Any]]) -> bool:
    return checkpoint is not None and checkpoint.get("_rebuild_reason") == E_CHECKPOINT_VERSION

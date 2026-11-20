"""结构化异常注入。

注入只改变输入文件或内存参数，绝不修改源码。三类注入分别对应：
* ``corrupt_config``  -> E_CONFIG：破坏配置字段（如库数改为 0）；
* ``ring_conflict``   -> E_RING：哈希环同权重节点冲突；
* ``version_mismatch`` -> E_VERSION：快照与磁盘配置版本不一致。
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

from .config import load_config_from_dict
from .errors import ConfigError, ShardBenchError, VersionError
from .snapshot import load_snapshot


def inject_corrupt_config(config_payload: dict[str, Any]) -> dict[str, Any]:
    corrupted = copy.deepcopy(config_payload)
    corrupted["tables"][0]["db_count"] = 0
    return corrupted


def inject_ring_conflict(config_payload: dict[str, Any]) -> dict[str, Any]:
    corrupted = copy.deepcopy(config_payload)
    for table in corrupted["tables"]:
        if table.get("algorithm") == "consistent_hash":
            first = table["params"]["nodes"][0]
            table["params"]["nodes"].append(
                {"id": first["id"], "weight": first["weight"]}
            )
            return corrupted
    raise ConfigError(
        "样本中不存在一致性哈希表，无法注入环冲突",
        paths=["injection.ring_conflict"],
    )


def write_variant(directory: str, name: str, payload: dict[str, Any]) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def trigger_config_injection(
    config_payload: dict[str, Any], units_payload: dict[str, Any]
) -> ShardBenchError:
    variant = inject_corrupt_config(config_payload)
    try:
        load_config_from_dict(variant, units_payload)
    except ShardBenchError as error:
        return error
    raise RuntimeError("配置损坏注入未触发 E_CONFIG")


def trigger_ring_injection(
    config_payload: dict[str, Any], units_payload: dict[str, Any]
) -> ShardBenchError:
    variant = inject_ring_conflict(config_payload)
    try:
        load_config_from_dict(variant, units_payload)
    except ShardBenchError as error:
        return error
    raise RuntimeError("环冲突注入未触发 E_RING")


def trigger_version_injection(snapshot_path: str) -> ShardBenchError:
    snapshot = load_snapshot(snapshot_path)
    tampered = copy.deepcopy(snapshot.versions)
    tampered["config_digest"] = "0" * 64
    snapshot.versions = tampered
    from .versioning import VersionState

    current = VersionState(
        config_version=tampered["config_version"],
        ring_version=tampered["ring_version"],
        dependency_version=tampered["dependency_version"],
        budget_version=tampered["budget_version"],
        config_digest="a" * 64,
        ring_digest=tampered["ring_digest"],
        dependency_digest=tampered["dependency_digest"],
        budget_digest=tampered["budget_digest"],
    )
    from .snapshot import compare_snapshot_versions

    mismatches = compare_snapshot_versions(snapshot, current)
    if not mismatches:
        raise RuntimeError("版本不一致注入未触发 E_VERSION")
    return VersionError(
        "快照版本与当前磁盘配置不一致，须整体替换缓存与依赖图状态",
        paths=["snapshot.versions"],
        details={"mismatches": mismatches},
    )

"""恢复后的一致性重放校验。"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import json

from .compiler import recompile_from_manifest
from .convergence import check_convergence
from .dispatcher import DispatchOutcome
from .errors import ReplayError
from .model import ConfigItem, DispatchStrategy, ManifestVersion, Receipt
from .snapshot import VersionSnapshot, content_digest

COMPARE_FIELDS = (
    "quantized",
    "error_bound",
    "propagation_path",
    "coef_quantized",
    "targets",
    "is_derived",
    "source_value",
)


def _first_mismatch(
    snapshot_compiled: Dict[str, Any], replay_compiled: Dict[str, Any]
) -> Tuple[str, str]:
    keys = sorted(set(snapshot_compiled) | set(replay_compiled))
    for key in keys:
        if key not in snapshot_compiled:
            return key, "<missing_in_snapshot>"
        if key not in replay_compiled:
            return key, "<missing_in_replay>"
        snap_json = snapshot_compiled[key].to_json()
        replay_json = replay_compiled[key].to_json()
        for field_name in COMPARE_FIELDS:
            if snap_json.get(field_name) != replay_json.get(field_name):
                return key, field_name
    return "", ""


def replay_snapshot(snapshot: VersionSnapshot) -> Dict[str, Any]:
    """按快照版本与格度重新推导失效集合、重算误差界并复核回执收敛。

    重放以快照内保存的清单 JSON 为输入、用全新编译器全量重编译，
    再与快照逐项比较；差异定位到首个不一致的 (配置项键, 字段) 并报 E_REPLAY。
    """
    # 从快照保存的清单 JSON 还原完整版本对象
    saved = json.loads(snapshot.manifest_json)
    manifest = ManifestVersion(
        version=snapshot.version,
        items=tuple(ConfigItem.from_json(it) for it in saved["items"]),
        strategy=DispatchStrategy.from_json(saved["strategy"]),
    )
    replayed = recompile_from_manifest(manifest, snapshot.grid)

    # 1) 内容摘要
    replay_digest = content_digest(
        snapshot.version,
        snapshot.grid,
        replayed.compiled,
        replayed.plan,
        replayed.graph,
    )
    if replay_digest != snapshot.content_digest:
        bad_key, bad_field = _first_mismatch(snapshot.compiled, replayed.compiled)
        raise ReplayError(
            "重放内容摘要与快照不一致",
            {
                "version": snapshot.version,
                "snapshot_digest": snapshot.content_digest,
                "replay_digest": replay_digest,
                "first_mismatch_key": bad_key or "<content>",
                "first_mismatch_field": bad_field or "digest",
            },
        )

    # 2) 逐字节产物 + 字段级定位
    for key in sorted(snapshot.compiled):
        snap_bytes = snapshot.compiled[key].canonical_bytes()
        replay_bytes = replayed.compiled[key].canonical_bytes()
        if snap_bytes != replay_bytes:
            snap_json = snapshot.compiled[key].to_json()
            replay_json = replayed.compiled[key].to_json()
            bad_field = next(
                (
                    field_name
                    for field_name in COMPARE_FIELDS
                    if snap_json.get(field_name) != replay_json.get(field_name)
                ),
                "canonical_bytes",
            )
            raise ReplayError(
                f"配置项 {key} 重放产物逐字节不一致",
                {
                    "version": snapshot.version,
                    "first_mismatch_key": key,
                    "first_mismatch_field": bad_field,
                },
            )

    # 3) 失效集合重推：无变更基准下应为空集合
    replayed_invalidated = replayed.graph.downstream_closure([])
    if replayed_invalidated:
        raise ReplayError(
            "无变更基准下重放出非空失效集合",
            {
                "first_mismatch_key": replayed_invalidated[0],
                "first_mismatch_field": "invalidated",
            },
        )

    # 4) 回执收敛复核（采信快照内 accepted 最终回执，重试回执保留但不采信）
    accepted: Dict[str, Receipt] = {}
    all_receipts: List[Receipt] = []
    for receipt in snapshot.receipts:
        all_receipts.append(receipt)
        if receipt.accepted:
            accepted[receipt.node_id] = receipt
    outcome = DispatchOutcome(version=snapshot.version, receipts=accepted)
    outcome.all_receipts = all_receipts
    report = check_convergence(
        snapshot.plan,
        replayed.compiled,
        replayed.graph,
        outcome,
        snapshot.grid.scale,
    )
    return {
        "version": snapshot.version,
        "digest_match": True,
        "byte_identical_keys": sorted(snapshot.compiled),
        "recompiled_keys": replayed.recompiled,
        "convergence": report,
    }

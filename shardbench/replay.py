"""独立重放校验。

从账本任一已提交轮次取出当时持久化的规范配置与路由单元，在全新进程式
上下文里重新编译路由计划，再与账本中逐项比对：路由标识、目标库表集合、
广播标记、结果摘要逐字段核对；首个不一致即抛出 E_REPLAY 并定位到
路由单元、库表与字段。
"""

from __future__ import annotations

import json
from typing import Any

from .config import load_config_from_dict
from .determinism import stable_targets
from .errors import ReplayError
from .engine import full_recompute
from .ledger import Ledger

COMPARE_FIELDS = (
    "route_key",
    "normalized_predicate",
    "is_broadcast",
    "broadcast_reason",
    "targets",
    "summary",
)


def replay_round(ledger_path: str, round_id: int) -> dict[str, Any]:
    ledger = Ledger(ledger_path)
    try:
        bundle = ledger.load_round(round_id)
    finally:
        ledger.close()
    round_row = bundle["round"]
    if round_row["status"] != "committed":
        raise ReplayError(
            f"轮次 {round_id} 未完整提交（状态 {round_row['status']}），不能作为重放基线",
            details={"round_id": round_id, "status": round_row["status"]},
        )
    config_payload = json.loads(round_row["config_json"])
    units_payload = json.loads(round_row["units_json"])
    config = load_config_from_dict(config_payload, units_payload)
    # 与引擎完全相同的编译 + 依赖合并路径，保证重放口径一致。
    recomputed = full_recompute(config)

    stored = {row["unit_id"]: _decode_unit_row(row) for row in bundle["units"]}
    _compare(stored, recomputed, config)
    return {
        "round_id": round_id,
        "status": "ok",
        "units_checked": len(stored),
        "payload_digest": round_row["payload_digest"],
    }


def _decode_unit_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "unit_id": row["unit_id"],
        "table": row["table_name"],
        "route_key": row["route_key"],
        "normalized_predicate": row["normalized_predicate"],
        "is_broadcast": bool(row["is_broadcast"]),
        "broadcast_reason": row["broadcast_reason"],
        "targets": json.loads(row["targets_json"]),
        "summary": row["summary"],
        "plan": json.loads(row["plan_json"]),
    }


def _compare(
    stored: dict[str, dict[str, Any]],
    recomputed: dict[str, dict[str, Any]],
    config: Any,
) -> None:
    for unit_id in config.topo_order:
        if unit_id not in stored:
            raise ReplayError(
                f"路由单元 {unit_id} 在账本中缺失",
                unit=unit_id,
                field="existence",
            )
        left = stored[unit_id]
        right = recomputed[unit_id]
        for field_name in COMPARE_FIELDS:
            expected = left.get(field_name)
            actual = right.get(field_name)
            if field_name == "targets":
                expected = stable_targets(expected)
                actual = stable_targets(actual)
            if expected != actual:
                target = _first_target_difference(expected, actual, field_name)
                raise ReplayError(
                    f"重放不一致：单元 {unit_id} 字段 {field_name} 出现首个差异",
                    unit=unit_id,
                    target=target,
                    field=field_name,
                    expected=expected,
                    actual=actual,
                )


def _first_target_difference(
    expected: Any, actual: Any, field_name: str
) -> str | None:
    if field_name != "targets":
        return None
    expected_set = set(expected or [])
    actual_set = set(actual or [])
    difference = sorted(expected_set.symmetric_difference(actual_set))
    return difference[0] if difference else None

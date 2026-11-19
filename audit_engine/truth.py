"""基于同步标记的真值对照。

真值唯一来源是清单内显式携带的 truth 字段（原始权限数值、原始路径
字节、同步标记、预期误差界区间）；生成器的任何中间计算结果均不参与。
"""

from __future__ import annotations

from typing import Any

from .errors import E_TRUTH_MISMATCH
from .models import Record, RecordVerdict, TruthMarker


def _raw_path_hex(record: Record) -> str:
    """以发送端相同的 UTF-8 编码重建原始路径字节 hex。"""
    return record.target_path.encode("utf-8").hex()


def verify_record(record: Record, verdict: RecordVerdict) -> dict[str, Any]:
    truth: TruthMarker | None = record.truth
    if truth is None:
        return {"status": "no_truth", "mismatches": []}

    mismatches: list[dict[str, Any]] = []

    observed_hex = _raw_path_hex(record)
    if observed_hex != truth.raw_path_bytes:
        mismatches.append(
            {
                "field": "raw_path_bytes",
                "observed": observed_hex,
                "expected": truth.raw_path_bytes,
            }
        )

    if record.perm_value != truth.raw_perm_value:
        mismatches.append(
            {
                "field": "raw_perm_value",
                "observed": record.perm_value,
                "expected": truth.raw_perm_value,
            }
        )

    epsilon = verdict.perm_epsilon
    if not (truth.expected_epsilon_lo <= epsilon <= truth.expected_epsilon_hi):
        mismatches.append(
            {
                "field": "epsilon",
                "observed": epsilon,
                "expected_interval": [
                    truth.expected_epsilon_lo,
                    truth.expected_epsilon_hi,
                ],
            }
        )

    if verdict.record_id != record.record_id or verdict.index != record.index:
        mismatches.append({"field": "identity", "observed": verdict.record_id})

    return {
        "status": "consistent" if not mismatches else "inconsistent",
        "sync_marker": truth.sync_marker,
        "mismatches": mismatches,
        "code": None if not mismatches else E_TRUTH_MISMATCH,
    }


def summarize(verifications: list[dict[str, Any]]) -> dict[str, Any]:
    consistent = sum(1 for v in verifications if v["status"] == "consistent")
    inconsistent = [v for v in verifications if v["status"] == "inconsistent"]
    missing = sum(1 for v in verifications if v["status"] == "no_truth")
    return {
        "consistent_count": consistent,
        "inconsistent_count": len(inconsistent),
        "no_truth_count": missing,
        "inconsistent_details": inconsistent,
    }

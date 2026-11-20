"""恢复后的一致性重放校验。

依据快照记录的 trace_version 与估计版本，对同一脚本重新执行裁决，
逐项比较窗口判定、翻转记录、结局与估计序列；首个不一致定位到
通道标识、组序号与字段名并以 E_REPLAY 报告。
"""

from __future__ import annotations

from dataclasses import asdict

from .errors import BusinessError, E_REPLAY
from .model import ReassemblyResult


def _compare_named(channel_id, label, expected, actual, key_fields):
    if len(expected) != len(actual):
        raise BusinessError(
            E_REPLAY,
            f"通道 {channel_id} {label} 条数不一致：期望 {len(expected)} 实际 {len(actual)}",
            path=f"{channel_id}.{label}",
        )
    for index, (exp, act) in enumerate(zip(expected, actual)):
        exp_d = asdict(exp) if hasattr(exp, "__dataclass_fields__") else exp
        act_d = asdict(act) if hasattr(act, "__dataclass_fields__") else act
        for field in exp_d:
            if exp_d[field] != act_d.get(field):
                loc = f"{channel_id}.{label}[{index}]"
                for key in key_fields:
                    if key in exp_d:
                        loc += f".{key}={exp_d[key]}"
                raise BusinessError(
                    E_REPLAY,
                    f"{loc} 字段 {field} 不一致：期望 {exp_d[field]!r} 实际 {act_d.get(field)!r}",
                    path=f"{loc}.{field}",
                )


def compare_results(expected: ReassemblyResult, actual: ReassemblyResult) -> None:
    """逐字段比较两条同通道裁决结果，首个差异即抛出 E_REPLAY。"""
    cid = expected.channel_id
    if cid != actual.channel_id:
        raise BusinessError(E_REPLAY, f"通道标识不一致 {cid} != {actual.channel_id}")
    _compare_named(cid, "estimates", expected.estimates, actual.estimates, ("version", "sample_count"))
    _compare_named(cid, "window_events", expected.window_events, actual.window_events, ("window_seq", "action"))
    _compare_named(cid, "flips", expected.flips, actual.flips, ("group_seq", "from_version"))
    _compare_named(cid, "outcomes", expected.outcomes, actual.outcomes, ("group_seq",))
    _compare_named(cid, "fragments", expected.fragments, actual.fragments, ("group_seq", "frag_offset"))


def replay_and_verify(snapshot_payload: dict, rerun_results: dict) -> None:
    """快照结果与重跑结果全量比对（快照项为 dict，重跑项为 ReassemblyResult）。"""
    expected_by_channel = {item["channel_id"]: item for item in snapshot_payload["results"]}
    if set(expected_by_channel) != set(rerun_results):
        raise BusinessError(
            E_REPLAY,
            f"通道集合不一致：快照 {sorted(expected_by_channel)} 重跑 {sorted(rerun_results)}",
        )
    for cid in sorted(expected_by_channel):
        exp = expected_by_channel[cid]
        actual = rerun_results[cid]
        for label, key_fields in (
            ("estimates", ("version", "sample_count")),
            ("window_events", ("window_seq", "action")),
            ("flips", ("group_seq", "from_version")),
            ("outcomes", ("group_seq",)),
        ):
            _compare_named(cid, label, exp[label], [asdict(x) for x in getattr(actual, label)], key_fields)

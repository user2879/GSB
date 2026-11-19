"""恢复后的一致性重放校验。

按快照轨迹逐轮重新推导受影响集合与递增后的注释版本，和账本逐项
比较。首个不一致即定位到触发条目键、受影响条目键与字段并抛 E_REPLAY。
v1 快照升级后没有命名空间/注释口径，属于可预期的不可复现，同样
如实报告而不是静默通过。
"""

from __future__ import annotations

from .errors import EReplay
from .model import EntryID, Graph, RoundRecord
from .propagation import reachable_closure
from .snapshot import DEFAULT_NAMESPACE


def replay_rounds(graph: Graph, rounds: list[RoundRecord],
                  snapshot_format_version: int) -> None:
    if snapshot_format_version == 1:
        first = rounds[0]
        raw_trigger = (first.explicit_triggers or first.expired_triggers
                       or first.refresh_triggers or ("<none>",))[0]
        if raw_trigger != "<none>":
            trigger_key = (
                f"{DEFAULT_NAMESPACE}/{EntryID.parse(raw_trigger).key}")
        else:
            trigger_key = raw_trigger
        raise EReplay(
            "v1 快照仅含条目键与过期时刻，升级后注释依赖集合口径缺失，"
            "传播轨迹无法按原口径复现",
            key=trigger_key,
            field_path="entries[].annotations",
            details={
                "round_no": first.round_no,
                "basis": "format_v1_lacks_annotation_sets",
                "snapshot_format_version": 1,
            },
        )

    versions = {
        entry_id: 0 for entry_id in graph.entry_ids_sorted()
    }
    alive = set(graph.entries)
    for record in rounds:
        seeds = _record_seeds(record)
        unknown = [seed.to_str() for seed in seeds if seed not in graph.entries]
        if unknown:
            raise EReplay(
                f"轮次 {record.round_no} 的触发条目在恢复后的图中不存在",
                key=unknown[0],
                field_path="rounds[].triggers",
                details={"round_no": record.round_no, "missing": unknown},
            )
        valid_seeds = [seed for seed in seeds if seed in alive]
        derived = reachable_closure(graph, set(valid_seeds), alive=alive)
        derived_str = [entry_id.to_str() for entry_id in derived]
        if derived_str != list(record.affected):
            first_index = _first_list_diff(derived_str, list(record.affected))
            actual = _at(derived_str, first_index)
            expected = _at(list(record.affected), first_index)
            raise EReplay(
                f"轮次 {record.round_no} 受影响集合与账本不一致",
                key=_trigger_key(record),
                field_path=f"rounds[{record.round_no}].affected[{first_index}]",
                details={
                    "round_no": record.round_no,
                    "mismatched_entry": expected or actual,
                    "replayed_value": actual,
                    "ledger_value": expected,
                    "basis": "affected_set_mismatch",
                },
            )

        for entry_id in derived:
            versions[entry_id] += 1
            alive.discard(entry_id)
        for entry_text, expected_version in sorted(
                record.annotation_versions.items()):
            entry_id = EntryID.parse(entry_text)
            actual_version = versions.get(entry_id)
            if actual_version != expected_version:
                raise EReplay(
                    f"轮次 {record.round_no} 条目 {entry_text} 注释版本不一致",
                    key=_trigger_key(record),
                    field_path=(
                        f"rounds[{record.round_no}]"
                        f".annotation_versions.{entry_text}"),
                    details={
                        "round_no": record.round_no,
                        "mismatched_entry": entry_text,
                        "replayed_value": actual_version,
                        "ledger_value": expected_version,
                        "basis": "annotation_version_mismatch",
                    },
                )


def _record_seeds(record: RoundRecord) -> list[EntryID]:
    seeds = {
        EntryID.parse(text)
        for text in (record.explicit_triggers + record.expired_triggers
                     + record.refresh_triggers)
    }
    return sorted(seeds)


def _first_list_diff(left: list[str], right: list[str]) -> int:
    index = 0
    while index < len(left) and index < len(right):
        if left[index] != right[index]:
            return index
        index += 1
    return index


def _at(values: list[str], index: int) -> str | None:
    return values[index] if index < len(values) else None


def _trigger_key(record: RoundRecord) -> str:
    return (record.explicit_triggers or record.expired_triggers
            or record.refresh_triggers or ("<none>",))[0]

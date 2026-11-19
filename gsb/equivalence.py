"""跨格式等价性核对：同一逻辑事件在三种承载下逐字段比对。"""

from __future__ import annotations

from itertools import combinations

from .model import Event
from .pipeline import FORMATS


def compare_events(a: Event, b: Event) -> str | None:
    """返回首个不相等字段名；完全相等返回 None。"""
    da, db = a.identity_dict(), b.identity_dict()
    for key in sorted(da):
        if da[key] != db[key]:
            return key
    return None


def equivalence_report(streams: dict[str, list[Event]]) -> dict:
    """对每对格式输出等价性结论与首个不相等字段。"""
    pairs = {}
    all_equal = True
    for fa, fb in combinations(FORMATS, 2):
        sa, sb = streams[fa], streams[fb]
        pair = {"equal": True, "events_compared": min(len(sa), len(sb)), "mismatches": []}
        if len(sa) != len(sb):
            pair["equal"] = False
            pair["mismatches"].append(
                {"event_index": None, "field": "event_count",
                 "left": len(sa), "right": len(sb)}
            )
        for idx, (ea, eb) in enumerate(zip(sa, sb)):
            field = compare_events(ea, eb)
            if field is not None:
                pair["equal"] = False
                pair["mismatches"].append(
                    {"event_index": idx, "field": field,
                     "left": ea.identity_dict()[field], "right": eb.identity_dict()[field]}
                )
                break  # 只报告首个不相等字段
        pairs[f"{fa}~{fb}"] = pair
        all_equal = all_equal and pair["equal"]
    return {"all_equal": all_equal, "pairs": pairs}

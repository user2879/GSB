"""确定性访问事件流的整数定点判定与误差界（E_ORDER）。"""

import json

import numpy as np

from .errors import WorkbenchError

# 定点缩放因子：1 次操作 = SCALE 个定点单位
SCALE = 1000
# 截断方向：向零截断（非负值等价于向下取整）
TRUNCATION = "toward_zero"

VERDICTS = ("allow", "throttle", "over_quota", "indeterminate_boundary", "unconstrained")


def load_events(path):
    events = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            events.append({
                "volume_id": ev["volume_id"],
                "ts_ms": int(ev["ts_ms"]),
                "op_count": int(ev["op_count"]),
                "op_bytes": int(ev["op_bytes"]),
                "seq": line_no,
            })
    return events


def evaluate_events(events, conclusions_by_volume):
    """按卷生效配额逐事件判定。

    令牌补充：tokens += rate * dt_ms * SCALE // 1000（向零截断）；
    容量比较：bytes_used > capacity_limit_bytes 判超配额；
    误差界：单值量子 = 1/SCALE 次操作；比较判定误差界 = 2*(配额量化误差*SCALE + 1)；
    |tokens - cost| <= 比较误差界 时判定为 indeterminate_boundary。
    时间戳倒退以 E_ORDER 拒绝该事件。
    """
    state = {}
    verdict_rows = []
    for ev in events:
        vid = ev["volume_id"]
        st = state.setdefault(vid, {"tokens": None, "last_ts": None, "bytes_used": 0})
        if st["last_ts"] is not None and ev["ts_ms"] < st["last_ts"]:
            raise WorkbenchError(
                "E_ORDER",
                f"卷 {vid} 事件时间戳倒退：{ev['ts_ms']} < {st['last_ts']}",
                {"volume_id": vid, "ts_ms": ev["ts_ms"],
                 "last_ts_ms": st["last_ts"], "seq": ev["seq"]})
        concl = conclusions_by_volume.get(vid, {})
        rate_c = concl.get("ops_per_second")
        burst_c = concl.get("burst_bucket_capacity")
        cap_c = concl.get("capacity_limit_bytes")
        if rate_c is None or burst_c is None:
            verdict = "unconstrained"
            cmp_bound = 0
        else:
            rate = rate_c["value"]
            burst = burst_c["value"]
            if st["tokens"] is None:
                st["tokens"] = burst * SCALE
            dt = ev["ts_ms"] - (st["last_ts"] if st["last_ts"] is not None else ev["ts_ms"])
            refill = (rate * dt * SCALE) // 1000  # 向零截断
            st["tokens"] = min(burst * SCALE, st["tokens"] + refill)
            # 配额来源量化误差合成到判定误差上界（定点单位）
            err = (rate_c["error_bound"] + burst_c["error_bound"]) * SCALE + 1
            cmp_bound = 2 * err
            cost = ev["op_count"] * SCALE
            st["bytes_used"] += ev["op_bytes"]
            if cap_c is not None and st["bytes_used"] > cap_c["value"]:
                verdict = "over_quota"
            elif abs(st["tokens"] - cost) <= cmp_bound:
                verdict = "indeterminate_boundary"
            elif st["tokens"] >= cost:
                verdict = "allow"
                st["tokens"] -= cost
            else:
                verdict = "throttle"
        st["last_ts"] = ev["ts_ms"]
        verdict_rows.append({
            "seq": ev["seq"], "volume_id": vid, "ts_ms": ev["ts_ms"],
            "verdict": verdict, "comparison_error_bound": cmp_bound,
        })
    return verdict_rows


def summarize_verdicts(verdict_rows, conclusions_by_volume):
    """按卷汇总判定结果（numpy 统计），并声明误差界与截断规则。"""
    summaries = []
    by_volume = {}
    for row in verdict_rows:
        by_volume.setdefault(row["volume_id"], []).append(row)
    for vid in sorted(by_volume):
        rows = by_volume[vid]
        codes = np.array([VERDICTS.index(r["verdict"]) for r in rows], dtype=np.int64)
        counts = np.bincount(codes, minlength=len(VERDICTS))
        concl = conclusions_by_volume.get(vid, {})
        quota_err = sum(c["error_bound"] for c in concl.values())
        summaries.append({
            "volume_id": vid,
            "metric": "-",
            "event_count": int(codes.size),
            "verdict_counts": {name: int(counts[i]) for i, name in enumerate(VERDICTS)},
            "single_value_quantum": f"1/{SCALE} op",
            "comparison_error_bound": 2 * (quota_err * SCALE + 1),
            "quota_error_bound_composed": quota_err,
            "truncation": TRUNCATION,
        })
    return summaries

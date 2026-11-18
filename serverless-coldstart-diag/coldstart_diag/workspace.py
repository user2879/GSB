"""快照目录管理与稳定视图模型（GUI 与演示共用，纯函数可无头测试）。"""

import json
from pathlib import Path

from .ledger import canonical_digest


def write_snapshot(snapshot_dir, run_id, manifest, payloads, agg, recovery=None):
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "run_id": run_id,
        "declaration_version": manifest["declaration_version"],
        "scale": manifest["scale"],
        "manifest_digest": canonical_digest(manifest),
        "recovery": recovery,
        "aggregate": agg,
        "minutes": payloads,
    }
    path = snapshot_dir / f"{run_id}.snapshot.json"
    path.write_text(json.dumps(doc, sort_keys=True, indent=2), encoding="utf-8")
    return path


def list_snapshots(snapshot_dir):
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        return []
    items = []
    for path in sorted(snapshot_dir.glob("*.snapshot.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        items.append(
            {
                "path": str(path),
                "run_id": doc["run_id"],
                "declaration_version": doc["declaration_version"],
                "scale": doc["scale"],
                "minutes": len(doc["minutes"]),
            }
        )
    return sorted(items, key=lambda x: (x["declaration_version"], x["run_id"]))


def load_snapshot(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# GUI 表格列（稳定渲染顺序）
TABLE_COLUMNS = (
    ("minute", "分钟"),
    ("calls", "调用数"),
    ("burst_calls", "突发"),
    ("required_instances", "所需实例"),
    ("available_instances", "可用实例"),
    ("reserved_used", "预留占用"),
    ("hot_calls", "热调用"),
    ("cold_calls", "冷调用"),
    ("cold_starts", "冷启动"),
    ("queued_calls", "排队"),
    ("cold_ratio_q", "冷启动比例(定点)"),
    ("end_to_end_bound_ulp", "误差界(ulp)"),
    ("avg_wait_ms", "平均等待(ms)"),
)


def build_table_rows(payloads):
    rows = []
    for p in payloads:
        rows.append(
            {
                "minute": p["minute"],
                "calls": p["calls"],
                "burst_calls": p["burst_calls"],
                "required_instances": p["required_instances"],
                "available_instances": p["available_instances"],
                "reserved_used": f"{p['reserved_used']}/{p['reserved_target']}",
                "hot_calls": p["hot_calls"],
                "cold_calls": p["cold_calls"],
                "cold_starts": p["cold_starts"],
                "queued_calls": p["queued_calls"],
                "cold_ratio_q": p["cold_ratio_q"],
                "end_to_end_bound_ulp": p["error_block"]["end_to_end_bound_ulp"],
                "avg_wait_ms": p["avg_wait_ms"],
            }
        )
    return rows


def heat_level(cold_ratio_q, scale):
    """冷启动比例热力分档（确定性阈值）。"""
    ratio = cold_ratio_q / scale
    if ratio <= 0.05:
        return 0, "低"
    if ratio <= 0.20:
        return 1, "中"
    if ratio <= 0.50:
        return 2, "高"
    return 3, "严重"


def wait_curve_series(payloads):
    return [{"minute": p["minute"], "avg_wait_ms": p["avg_wait_ms"],
             "queued_calls": p["queued_calls"]} for p in payloads]

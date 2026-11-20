"""会话快照：版本识别、整体替换恢复与重放校验。"""
from __future__ import annotations

import json
import os
import shutil

from . import SNAPSHOT_FORMAT_VERSION
from .errors import ReplayError
from .util import canonical_json, digest_of


def _snapshot_payload(tenants, pools, selection, result, processed_cursor):
    tenant_digest = digest_of(
        [
            {"tid": t.tid, "quota": t.quota, "concurrency_cap": t.concurrency_cap, "burst": t.burst}
            for t in sorted(tenants, key=lambda x: x.tid)
        ]
    )
    pool_digest = digest_of(
        [
            {"pid": p.pid, "capacity": p.capacity, "priority_order": list(p.priority_order)}
            for p in sorted(pools, key=lambda x: x.pid)
        ]
    )
    return {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "tenant_set_version": tenant_digest,
        "pool_set_version": pool_digest,
        "default_path": selection["selected"],
        "path_selection": selection,
        "tenant_quota_final": dict(sorted(result.final_tenant_quota.items())),
        "tenant_burst_final": dict(sorted(result.final_tenant_burst.items())),
        "tenant_inflight_final": dict(sorted(result.final_inflight.items())),
        "pool_capacity_final": dict(sorted(result.final_pool_capacity.items())),
        "processed_cursor": processed_cursor,
        "outcomes": [list(o.canonical_tuple()) for o in result.outcomes],
        "queue_trace": [
            [e.rid, e.tid, e.pid, e.kind, e.at_ms, e.wait_ms, e.basis, e.detail]
            for e in result.queue_events
        ],
    }


def write_snapshot(path: str, tenants, pools, selection, result, processed_cursor):
    payload = _snapshot_payload(tenants, pools, selection, result, processed_cursor)
    payload["content_digest"] = digest_of({k: v for k, v in payload.items() if k != "content_digest"})
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(payload))
    os.replace(tmp_path, path)
    return payload["content_digest"]


def read_snapshot(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    stored = payload.get("content_digest")
    recomputed = digest_of({k: v for k, v in payload.items() if k != "content_digest"})
    if stored != recomputed:
        raise ReplayError(
            "快照内容摘要不符，需要整体替换恢复",
            field="content_digest",
            detail={"stored": stored, "recomputed": recomputed},
        )
    if payload.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        raise ReplayError(
            "快照格式版本不符，需要整体替换恢复",
            field="format_version",
            detail={"stored": payload.get("format_version"), "expected": SNAPSHOT_FORMAT_VERSION},
        )
    return payload


def current_set_versions(tenants, pools):
    tenant_digest = digest_of(
        [
            {"tid": t.tid, "quota": t.quota, "concurrency_cap": t.concurrency_cap, "burst": t.burst}
            for t in sorted(tenants, key=lambda x: x.tid)
        ]
    )
    pool_digest = digest_of(
        [
            {"pid": p.pid, "capacity": p.capacity, "priority_order": list(p.priority_order)}
            for p in sorted(pools, key=lambda x: x.pid)
        ]
    )
    return tenant_digest, pool_digest


def restore_snapshot(snapshot_path: str, current_tenants, current_pools):
    """打开快照并识别版本；不一致时整体替换，旧快照保留为只读备份。"""
    payload = read_snapshot(snapshot_path)
    tenant_digest, pool_digest = current_set_versions(current_tenants, current_pools)
    mismatch = (
        payload["tenant_set_version"] != tenant_digest
        or payload["pool_set_version"] != pool_digest
    )
    backup_path = None
    if mismatch:
        backup_path = snapshot_path + ".readonly.bak"
        shutil.copy2(snapshot_path, backup_path)
        os.chmod(backup_path, 0o444)
    return payload, mismatch, backup_path


def verify_replay(payload: str, replay_result, target_cursor):
    """按快照记录的路径重算后逐项比较，首个不一致以 E_REPLAY 报明。"""
    if len(payload["outcomes"]) != len(replay_result.outcomes):
        raise ReplayError(
            "重放结局数量与快照不一致",
            ref_id=None,
            field="outcomes_length",
            detail={"snapshot": len(payload["outcomes"]), "replay": len(replay_result.outcomes)},
        )
    fields = (
        "rid", "tid", "pid", "arrival", "verdict", "reason", "basis",
        "decided_at", "granted_at", "completed_at", "wait_ms",
        "tenant_quota_remaining", "tenant_burst_remaining", "inflight",
        "pool_capacity_remaining", "demand", "enqueued",
    )
    for snap_row, outcome in zip(payload["outcomes"], replay_result.outcomes):
        replay_row = list(outcome.canonical_tuple())
        if snap_row != replay_row:
            for field_name, x, y in zip(fields, snap_row, replay_row):
                if x != y:
                    raise ReplayError(
                        "重放结局不可复现",
                        ref_id=outcome.rid,
                        field=field_name,
                        detail={"snapshot": x, "replay": y},
                    )
    state_checks = [
        ("tenant_quota_final", replay_result.final_tenant_quota),
        ("tenant_burst_final", replay_result.final_tenant_burst),
        ("tenant_inflight_final", replay_result.final_inflight),
        ("pool_capacity_final", replay_result.final_pool_capacity),
    ]
    for field_name, replay_state in state_checks:
        snap_state = payload[field_name]
        for key in sorted(set(snap_state) | set(replay_state)):
            if snap_state.get(key) != replay_state.get(key):
                raise ReplayError(
                    "重放终态不可复现",
                    ref_id=key,
                    field=field_name,
                    detail={"snapshot": snap_state.get(key), "replay": replay_state.get(key)},
                )
    snap_trace = [tuple(row) for row in payload["queue_trace"]]
    replay_trace = [
        (e.rid, e.tid, e.pid, e.kind, e.at_ms, e.wait_ms, e.basis, e.detail)
        for e in replay_result.queue_events
    ]
    if snap_trace != replay_trace:
        for index, (a, b) in enumerate(zip(snap_trace, replay_trace)):
            if a != b:
                raise ReplayError(
                    "重放排队轨迹不可复现",
                    ref_id=a[0],
                    field=f"queue_trace[{index}]",
                    detail={"snapshot": list(a), "replay": list(b)},
                )
        raise ReplayError("重放排队轨迹长度不一致", field="queue_trace")
    if target_cursor != payload["processed_cursor"]:
        raise ReplayError(
            "重放处理游标与快照不一致",
            ref_id=target_cursor,
            field="processed_cursor",
            detail={"snapshot": payload["processed_cursor"], "replay": target_cursor},
        )
    return True

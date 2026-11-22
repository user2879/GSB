"""稳定顺序的诊断报告（numpy 统计命中率）。"""

import numpy as np


def build_report(ledger, title="CDN 边缘缓存一致性诊断报告"):
    lines = []
    lines.append(f"===== {title} =====")
    lines.append(f"format_version: {ledger.get_meta('format_version')}")
    lines.append(f"policy_version: {ledger.get_meta('policy_version')}")
    lines.append(f"content_digest: {ledger.get_meta('content_digest')}")

    requests = [e["payload"] for e in ledger.events("request")]
    lines.append("")
    lines.append("---- 请求结果（按账本顺序） ----")
    lines.append("batch  node     path                 result   source         etag                 ttl_left_ms")
    for r in requests:
        lines.append(
            f"{r['batch']:<6} {r['node']:<8} {r['path']:<20} {r['result']:<8} "
            f"{str(r['source']):<14} {str(r['etag']):<20} {r['ttl_remaining_ms']}"
        )

    lines.append("")
    lines.append("---- 逐出事件 ----")
    any_ev = False
    for r in requests:
        for ev in r["evictions"]:
            lines.append(
                f"batch={r['batch']} node={r['node']} key={ev['cache_key']} reason={ev['reason']}"
            )
            any_ev = True
    if not any_ev:
        lines.append("(无)")

    attempts = [e["payload"] for e in ledger.events("pull_attempt")]
    lines.append("")
    lines.append("---- 拉取调度尝试 ----")
    for a in attempts:
        lines.append(
            f"task={a['task_seq']} node={a['node']} path={a['path']} "
            f"lease_version={a['lease_version']} attempt={a['attempt_no']} "
            f"outcome={a['outcome']} wait_ms={a['wait_end_mono_ms'] - a['wait_start_mono_ms']}"
        )
    if not attempts:
        lines.append("(无)")

    props = [e["payload"] for e in ledger.events("propagation")]
    lines.append("")
    lines.append("---- 派生失效传播轨迹 ----")
    for p in props:
        lines.append(
            f"trigger={p['trigger_path']} affected={p['affected_path']} node={p['node']} "
            f"present={p['present']} entry_version={p['old_entry_version']}->{p['new_entry_version']} "
            f"etag={p['new_etag']}"
        )
    if not props:
        lines.append("(无)")

    conflicts = [e["payload"] for e in ledger.events("lease_conflict")]
    lines.append("")
    lines.append("---- 租约冲突裁决 ----")
    for c in conflicts:
        lines.append(
            f"rule={c['rule']} node={c['node_id']} path={c['path']} "
            f"revoked={c['revoked_lease_id']} {c['old_content_version']}->{c['new_content_version']}"
        )
    if not conflicts:
        lines.append("(无)")

    lines.append("")
    lines.append("---- 节点统计（numpy） ----")
    nodes = sorted({r["node"] for r in requests})
    for node in nodes:
        rs = [r for r in requests if r["node"] == node]
        hits = np.array([1 if r["result"] == "hit" else 0 for r in rs], dtype=np.float64)
        rate = float(hits.mean()) if hits.size else 0.0
        ev_count = sum(len(r["evictions"]) for r in rs)
        lines.append(
            f"{node}: requests={len(rs)} hits={int(hits.sum())} "
            f"hit_rate={rate:.3f} evictions={ev_count}"
        )
    return "\n".join(lines)

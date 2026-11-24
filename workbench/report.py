from __future__ import annotations

from typing import Any

from .constants import WATERMARK_RULE_METADATA


def build_report(workbench: Any) -> dict[str, Any]:
    state = workbench.state
    validation = workbench.validation
    node_rows = []
    for node_id in sorted(state["node_versions"]):
        versions = state["node_versions"][node_id]
        watermark = state["watermarks"].get(node_id, {})
        summary = state["node_summaries"].get(node_id, {})
        occupancy = int(summary.get("occupancy", 0))
        slots = int(versions["capacity_slots"])
        permille_quotient, permille_remainder = divmod(occupancy * 1000, slots) if slots else (0, 0)
        occupancy_permille = permille_quotient + (1 if 2 * permille_remainder >= slots else 0)
        node_rows.append({
            "node_id": node_id,
            "loopback_host": "127.0.0.1",
            "port": validation["nodes"][node_id]["port"],
            "capacity_slots": slots,
            "occupancy": occupancy,
            "occupancy_permille_exact": f"{occupancy * 1000}/{slots}",
            "occupancy_permille_rounded": occupancy_permille,
            "soft_limit": watermark.get("soft_limit"),
            "hard_limit": watermark.get("hard_limit"),
            "soft_permille": watermark.get("soft_permille"),
            "hard_permille": watermark.get("hard_permille"),
            "watermark_state": watermark.get("watermark_state"),
            "active_leases": len(state["residents"][node_id]),
            "lease_limit": versions["lease_limit"],
            "quota_version": versions["quota_version"],
            "capacity_version": versions["capacity_version"],
            "resident_keys": summary.get("keys", sorted(state["residents"][node_id])),
            "resident_fingerprint": summary.get("resident_fingerprint"),
        })

    authority_rows = [state["authority"][key] for key in sorted(state["authority"])]
    decision_rows = [
        state["pending_decisions"][app_id]
        for app_id in sorted(
            state["pending_decisions"],
            key=lambda app_id: (
                state["pending_decisions"][app_id]["artifact_key"],
                state["applications"][app_id]["arrival_seq"],
                app_id,
            ),
        )
    ]
    eviction_order = [
        {
            "seq": item["seq"],
            "node_id": item["node_id"],
            "key": item["key"],
            "reason": item["reason"],
            "plan_id": item.get("plan_id"),
            "released_slots": item.get("released_slots"),
        }
        for item in state["evictions"]
    ]

    return {
        "schema_version": "workbench.report.v1",
        "config_hash": state["config_hash"],
        "event_cursor": state["cursor"],
        "processed_event_count": state["processed_event_count"],
        "watermark_rules": WATERMARK_RULE_METADATA,
        "nodes": node_rows,
        "authority_replicas": authority_rows,
        "eviction_order": eviction_order,
        "eviction_details": state["evictions"],
        "pending_decisions": decision_rows,
        "reconciliations": state["reconciliations"],
        "budget": state["budget"],
        "cache_ledger": state["cache_ledger"],
        "quota_changes": state["quota_changes"],
        "lease_renewals": state["renewals"],
        "read_results": state["read_results"],
        "manual_releases": state["manual_releases"],
        "overwrite_references": state["overwrites"],
        "protocol_violations": state["protocol_violations"],
        "failure_list": state["failure_list"],
        "code_counts": state["code_counts"],
        "input_summary": {
            "declared_node_count": len(validation.get("config", {}).get("nodes", []) or []),
            "loaded_node_count": len(validation["nodes"]),
            "manifest_stage_count": len(validation.get("manifest", {}).get("stages", [])),
            "valid_event_count": len(validation.get("events", [])),
            "input_failure_count": len(validation.get("failures", [])),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 推荐特征缓存租约与水印诊断报告",
        "",
        f"- 配置哈希：`{report['config_hash']}`",
        f"- 事件游标：`{report['event_cursor']}`",
        f"- 已处理事件：`{report['processed_event_count']}`",
        "",
        "## 节点占用与水位",
        "",
        "| 节点 | 占用/槽位 | 软上限 | 硬上限 | 水位 | 租约 | 配额版本 | 容量版本 |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for node in report["nodes"]:
        lines.append(
            f"| {node['node_id']} | {node['occupancy']}/{node['capacity_slots']} | "
            f"{node['soft_limit']} | {node['hard_limit']} | {node['watermark_state']} | "
            f"{node['active_leases']}/{node['lease_limit']} | {node['quota_version']} | "
            f"{node['capacity_version']} |"
        )

    lines.extend(["", "## 权威副本", ""])
    if not report["authority_replicas"]:
        lines.append("无当前副本。")
    for item in report["authority_replicas"]:
        lines.append(
            f"- `{item['artifact_key']}` → `{item['winner_node_id']}` "
            f"(cv={item['capacity_version']}, elapsed_ms={item['elapsed_ms']}, digest={item['content_digest']})"
        )

    lines.extend(["", "## 驱逐明细", ""])
    if not report["eviction_details"]:
        lines.append("无驱逐。")
    for item in report["eviction_details"]:
        lines.append(
            f"- seq={item['seq']} node=`{item['node_id']}` key=`{item['key']}` "
            f"reason={item['reason']} released_slots={item.get('released_slots')}"
        )

    lines.extend(["", "## 预占与拒绝", ""])
    if not report["pending_decisions"]:
        lines.append("无待处理申请裁决。")
    for item in report["pending_decisions"]:
        lines.append(
            f"- `{item['artifact_key']}` on `{item['node_id']}`: **{item['decision']}** "
            f"reason={item['reason']} evictions={','.join(item['planned_eviction_keys']) or '-'}"
        )

    lines.extend(["", "## 业务码计数", ""])
    for code, count in sorted(report["code_counts"].items()):
        if count:
            lines.append(f"- `{code}`: {count}")
    if not any(report["code_counts"].values()):
        lines.append("- 无异常码")
    lines.append("")
    return "\n".join(lines)


def phase_products(workbench: Any) -> dict[str, Any]:
    report = build_report(workbench)
    state = workbench.state
    return {
        "01_load_validation": {
            "schema_version": "workbench.phase.validation.v1",
            "config_hash": workbench.validation["config_hash"],
            "loaded_nodes": sorted(workbench.validation["nodes"]),
            "manifest": workbench.validation.get("manifest"),
            "valid_event_count": len(workbench.validation.get("events", [])),
            "failures": workbench.validation.get("failures", []),
        },
        "02_capacity_watermarks": {
            "schema_version": "workbench.phase.watermark.v1",
            "rules": WATERMARK_RULE_METADATA,
            "watermarks": state["watermarks"],
            "node_summaries": state["node_summaries"],
        },
        "03_leases_evictions": {
            "schema_version": "workbench.phase.lease_eviction.v1",
            "residents": state["residents"],
            "renewals": state["renewals"],
            "evictions": state["evictions"],
            "plans": list(state["plans"].values()),
            "reconciliations": state["reconciliations"],
        },
        "04_ledgers": {
            "schema_version": "workbench.phase.ledgers.v1",
            "authority": state["authority"],
            "authority_history": state["authority_history"],
            "overwrites": state["overwrites"],
            "pending_decisions": list(state["pending_decisions"].values()),
            "decision_history": state["decision_history"],
            "reconciliations": state["reconciliations"],
            "budget": state["budget"],
            "cache_ledger": state["cache_ledger"],
        },
        "resident_snapshot": {
            "schema_version": "workbench.resident_snapshot.v1",
            "cursor": state["cursor"],
            "residents": state["residents"],
            "node_summaries": state["node_summaries"],
        },
        "report": report,
    }

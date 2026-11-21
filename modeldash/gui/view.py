"""Stable text rendering used by both headless diagnostics and tests."""

from __future__ import annotations

from typing import Any


def format_text_view(view: dict[str, Any]) -> str:
    session = view.get("session")
    if not session:
        return "No current session."
    lines = [
        f"session={session['session_id']} spec_version={session['spec_version']} current_version={view['current_version']}",
        f"spec_digest={session['spec_digest']} content_digest={session['content_digest']}",
        "",
        "PRIORITY QUEUE (priority desc, task id):",
    ]
    for task in view.get("priority_queue", []):
        shards = ",".join(str(item["shard_index"]) for item in task.get("completed_shards", []))
        lines.append(
            f"  {task['task_id']} p={task['priority']} node={task['node_id']} state={task['state']} "
            f"completed=[{shards}] wait_ms={task['wait_ms']} quota_ms={task['used_quota_ms']} error={task['error_code']}"
        )
    lines.extend(["", "PREEMPTIONS:", ])
    for item in view.get("preemptions", []):
        lines.append(
            f"  {item['preemption_id']} victim={item['task_id']} by={item['by_task_id']} "
            f"requested={item['requested_shard']} suspended={item['suspended_shard']} status={item['status']}"
        )
    lines.extend(["", "PREEMPTION DECISIONS:"])
    for item in view.get("decisions", []):
        lines.append(
            f"  {item['decision_id']} high={item['high_task_id']} victim={item['victim_task_id']} "
            f"verdict={item['verdict']} preemptable={item['preemptable']} reason={item['reason']}"
        )
    lines.extend(["", "NODE UNTRUST (score desc, node id):"])
    for item in view.get("node_untrust", []):
        lines.append(
            f"  rank={item['rank']} node={item['node_id']} score={item['score']} "
            f"untrusted={item['untrusted_receipts']}/{item['receipt_count']} max_decay={item['max_decay']}"
        )
    lines.extend(["", "RECEIPTS (node, task, shard, receipt key):"])
    for item in view.get("receipts", []):
        lines.append(
            f"  {item['node_id']} {item['task_id']} v={item['config_version']} s={item['shard_index']} "
            f"{item['status']} score={item['untrust_score']} decay={item['decay_count']} "
            f"out={item['output_digest'][:12]} key={item['receipt_key']}"
        )
    lines.extend(["", "SNAPSHOTS (snapshot id):"])
    for item in view.get("snapshots", []):
        lines.append(
            f"  {item['snapshot_id']} label={item['label']} format={item['format_version']} digest={item['content_digest'][:12]}"
        )
    lines.extend(["", "EVOLUTIONS:"])
    for item in view.get("evolutions", []):
        lines.append(
            f"  {item['old_session_id']} -> {item['new_session_id']} reason={item['reason']} "
            f"nodes={item['affected_nodes']} tasks={item['affected_tasks']}"
        )
    return "\n".join(lines) + "\n"

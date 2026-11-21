from __future__ import annotations

from typing import Any


def _line(char: str = "-", width: int = 72) -> str:
    return char * width


def render_overview(report: dict[str, Any]) -> str:
    lines = [
        "MVCC Visibility Diagnostics Workbench",
        _line("="),
        f"status:             {report.get('status')}",
        f"code:               {report.get('code')}",
        f"generation:         {report.get('generation')}",
        f"ledger version:     {report.get('ledger_version')}",
        f"conclusion digest:  {report.get('conclusion_digest')}",
        f"snapshot:           {report.get('snapshot')}",
    ]
    diagnostic = report.get("diagnostic")
    if diagnostic:
        lines.extend(
            [
                "",
                f"diagnostic code:    {diagnostic.get('code')}",
                f"first mismatch:     {diagnostic.get('location')}",
                f"message:            {diagnostic.get('message')}",
            ]
        )
    recovery = report.get("recovery") or {}
    lines.extend(
        [
            "",
            f"report available:   {recovery.get('report_available')}",
            f"unknown tasks:      {recovery.get('unknown_count', 0)}",
            f"next action:        {recovery.get('next_action')}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_chains(report: dict[str, Any], dataset: Any) -> str:
    lines = ["Version chains (sorted by row_key, then version_no)", _line("=")]
    for row_key in sorted(dataset.chains):
        lines.append(f"\n[{row_key}]")
        for record in dataset.chains[row_key]:
            txn = dataset.txns_by_id[record.txn_id]
            outcome = txn.outcome
            commit = "none" if txn.commit_seq is None else str(txn.commit_seq)
            lines.append(
                f"  v{record.version_no:02d} <- prev={record.prev_version} "
                f"txn={record.txn_id} outcome={outcome} commit={commit} "
                f"value={record.value_summary} digest={record.value_digest}"
            )
    lines.append("\nRead-point snapshot lines")
    lines.append(_line())
    for read_point in dataset.read_points:
        lines.append(
            f"  {read_point.read_point_id}: snapshot_seq={read_point.snapshot_seq} "
            f"reader_txn={read_point.reader_txn_id}"
        )
    return "\n".join(lines) + "\n"


def render_decisions(report: dict[str, Any]) -> str:
    lines = ["Visibility decisions (stable read_point/row/query order)", _line("=")]
    for decision in report.get("decisions", []):
        lines.append(
            f"\n[{decision['query_id']}] rp={decision['read_point_id']} row={decision['row_key']} "
            f"snapshot={decision['snapshot_seq']} reader={decision['reader_txn_id']}"
        )
        lines.append(
            f"  visible_version={decision['visible_version_no']} "
            f"visible_txn={decision['visible_txn_id']} outcome={decision['outcome']} "
            f"predicate={decision['predicate_matched']} cost={decision['cost_units']}"
        )
        lines.append(f"  reason_digest={decision['reason_digest']}")
        for step in decision.get("reason_chain", []):
            lines.append(
                f"    {step.get('step')}: {step.get('code')} "
                f"v={step.get('version_no', '-')} txn={step.get('txn_id', '-')} "
                f"{step.get('message', '')}"
            )
        policy = decision.get("visibility_policy") or {}
        lines.append(f"  policy: {policy.get('declaration', '')}")
    return "\n".join(lines) + "\n"


def render_schedule(report: dict[str, Any]) -> str:
    lines = ["Scheduling timeline (monotonic milliseconds)", _line("=")]
    for trace in report.get("schedule", []):
        lines.append(
            f"  {trace['query_id']} rp={trace['read_point_id']} status={trace['status']} "
            f"outcome={trace.get('decision_outcome')} "
            f"queue=[{trace['queue_enter_mono_ms']},{trace['queue_exit_mono_ms']}] "
            f"start={trace['started_mono_ms']} end={trace['ended_mono_ms']} "
            f"wait={trace['wait_ms']} worker_delay={trace.get('worker_delay_ms', 0)} "
            f"duration={trace['duration_ms']}"
        )
    return "\n".join(lines) + "\n"


def render_budget(report: dict[str, Any]) -> str:
    plan = report.get("budget_plan") or {}
    lines = [
        "Budget and ledger",
        _line("="),
        f"global_cost_budget={plan.get('global_cost_budget')} "
        f"round_cost_budget={plan.get('round_cost_budget')} "
        f"allocated={plan.get('total_allocated_cost')}",
    ]
    for budget_round in plan.get("rounds", []):
        lines.append(
            f"\nRound {budget_round['round_no']} cap={budget_round['round_cost_budget']} "
            f"used={budget_round['allocated_cost']}"
        )
        for allocation in budget_round.get("allocations", []):
            lines.append(
                f"  {allocation['query_id']} rp={allocation['read_point_id']} "
                f"priority={allocation['priority']} cost={allocation['estimated_cost']}"
            )
    if plan.get("deferred"):
        lines.append("\nDeferred queries")
        for item in plan["deferred"]:
            lines.append(
                f"  {item['query_id']} reason={item['reason']} cost={item['estimated_cost']} "
                f"detail={item['detail']}"
            )
    lines.append("\nCommitted rounds")
    for row in report.get("rounds", []):
        lines.append(
            f"  gen={row['generation']} round={row['round_no']} cursor={row['ledger_cursor']} "
            f"digest={row['content_digest']}"
        )
    return "\n".join(lines) + "\n"


def render_affected(report: dict[str, Any]) -> str:
    affected = report.get("affected")
    lines = ["Affected set and replay checks", _line("=")]
    if not affected:
        lines.append("  no replacement event in this view")
        return "\n".join(lines) + "\n"
    lines.append(f"affected read points: {affected.get('affected_read_point_ids')}")
    lines.append(f"changed transaction ids: {affected.get('changed_txn_ids')}")
    lines.append(f"changed row keys: {affected.get('changed_row_keys')}")
    for item in affected.get("changed_queries", []):
        lines.append(
            f"  {item['query_id']} rp={item['read_point_id']} row={item['row_key']} "
            f"visible {item['before_visible_version_no']} -> {item['after_visible_version_no']} "
            f"reason {item['before_reason_digest']} -> {item['after_reason_digest']}"
        )
    for item in affected.get("unchanged_checks", []):
        lines.append(
            f"  unchanged {item['query_id']} byte_equal={item['byte_equal']} "
            f"reason={item['before_reason_digest']}"
        )
    return "\n".join(lines) + "\n"


def render_full_report(report: dict[str, Any], dataset: Any) -> str:
    return "\n".join(
        [
            render_overview(report),
            render_chains(report, dataset),
            render_decisions(report),
            render_schedule(report),
            render_budget(report),
            render_affected(report),
        ]
    )

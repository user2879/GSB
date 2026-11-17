"""Pure-data view model for the Tkinter diagnostic UI.

Kept free of any Tk dependency so rendering order and derived display values
can be unit-tested on headless machines without ``tkinter`` installed.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List

STATUS_ORDER = {"committed": 0, "running": 1, "unknown": 2, "pending": 3}


@dataclass(frozen=True)
class TimelineRow:
    index: int
    start_offset_ms: int
    target_id: str
    injection_type: str
    status: str
    observed_state: str
    duration_ms: int

    def render(self) -> str:
        state = self.observed_state or "-"
        return (
            f"#{self.index:02d} t={self.start_offset_ms:>5}ms "
            f"{self.injection_type:<13} -> {self.target_id:<10} "
            f"[{self.status:<9}/{state:<10}] {self.duration_ms}ms"
        )


@dataclass(frozen=True)
class ObservationPoint:
    injection_idx: int
    phase: str
    success_rate_bps: Any
    rate_error_bps: int
    p50_ms: Any
    p95_ms: Any
    max_ms: Any
    insufficient: bool

    def render(self) -> str:
        if self.insufficient:
            return (
                f"#{self.injection_idx:02d} {self.phase:<8} INSUFFICIENT "
                f"(metrics not fabricated)"
            )
        return (
            f"#{self.injection_idx:02d} {self.phase:<8} "
            f"success={self.success_rate_bps}bps +/-{self.rate_error_bps} "
            f"p50={self.p50_ms} p95={self.p95_ms} max={self.max_ms}ms "
            f"(lat err +/-{1}ms)"
        )


def build_view_model(conn: sqlite3.Connection) -> Dict[str, Any]:
    experiment = conn.execute("SELECT * FROM experiment WHERE id = 1").fetchone()
    injection_rows = conn.execute("SELECT * FROM injections ORDER BY idx").fetchall()
    timeline = [
        TimelineRow(
            index=row["idx"],
            start_offset_ms=row["start_offset_ms"],
            target_id=row["target_id"],
            injection_type=row["injection_type"],
            status=row["status"],
            observed_state=row["observed_state"] or "-",
            duration_ms=row["duration_ms"],
        )
        for row in injection_rows
    ]
    timeline.sort(key=lambda item: (item.index, item.target_id, item.injection_type))

    observation_rows = conn.execute(
        "SELECT * FROM observations ORDER BY injection_idx, id"
    ).fetchall()
    points: List[ObservationPoint] = []
    for row in observation_rows:
        summary = json.loads(row["summary_json"])
        points.append(
            ObservationPoint(
                injection_idx=row["injection_idx"],
                phase=row["phase"],
                success_rate_bps=summary["success_rate_bps"],
                rate_error_bps=summary["rate_quant_error_bps"],
                p50_ms=summary["p50_latency_ms"],
                p95_ms=summary["p95_latency_ms"],
                max_ms=summary["max_latency_ms"],
                insufficient=bool(summary["insufficient"]),
            )
        )

    replacements = [
        dict(row) for row in conn.execute("SELECT * FROM replacements ORDER BY id").fetchall()
    ]
    unknown_indexes = [
        row["idx"] for row in injection_rows if row["status"] == "unknown"
    ]
    progress = conn.execute("SELECT * FROM progress WHERE id = 1").fetchone()
    return {
        "ledger_format_version": experiment["ledger_format_version"] if experiment else None,
        "spec_version": experiment["spec_version"] if experiment else None,
        "seed": experiment["seed"] if experiment else None,
        "experiment_id": experiment["experiment_id"] if experiment else None,
        "timeline_digest": (experiment["timeline_digest"][:16] + "...") if experiment else "-",
        "timeline_rows": timeline,
        "observation_points": points,
        "replacements": replacements,
        "unknown_indexes": unknown_indexes,
        "progress": dict(progress) if progress else {},
    }


def render_text_report(view: Dict[str, Any]) -> str:
    lines = [
        f"Experiment : {view['experiment_id']} (spec v{view['spec_version']}, "
        f"ledger v{view['ledger_format_version']}, seed={view['seed']})",
        f"Timeline   : {view['timeline_digest']}",
        f"Progress   : last_committed={view['progress'].get('last_committed_idx')} "
        f"total={view['progress'].get('total_injections')}",
        f"Unknown    : {view['unknown_indexes'] or 'none'}",
        "",
        "Timeline (stable order):",
    ]
    lines.extend(f"  {row.render()}" for row in view["timeline_rows"])
    lines.append("")
    lines.append("Observations (fixed-point + error bounds):")
    lines.extend(f"  {point.render()}" for point in view["observation_points"])
    lines.append("")
    lines.append("Whole-snapshot replacements:")
    if view["replacements"]:
        for event in view["replacements"]:
            lines.append(
                f"  v{event['old_format_version']} -> v{event['new_format_version']} "
                f"skipped={event['skipped_injections']} backup={event['backup_path']}"
            )
    else:
        lines.append("  none")
    return "\n".join(lines)

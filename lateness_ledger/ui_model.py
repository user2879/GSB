"""Read-only, deterministically ordered projections for the Tkinter workbench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ledger import LedgerReader


def load_dashboard(ledger_path: str | Path) -> dict[str, Any]:
    """Load display data without mutating the ledger."""

    reader = LedgerReader(ledger_path)
    meta = reader.meta()
    windows = reader.windows()
    points = reader.watermark_points()
    displacements = reader.displacements()
    snapshots = reader.snapshots()

    partitions: dict[str, dict[str, Any]] = {}
    for point in points:
        label = str(point["partition_label"])
        entry = partitions.setdefault(
            label,
            {
                "partition_label": label,
                "partition": point["partition"],
                "points": [],
            },
        )
        entry["points"].append(
            {
                "event_seq": point["event_seq"],
                "event_id": point["event_id"],
                "phase": point["phase"],
                "partition_watermark": point["partition_watermark"],
                "global_watermark": point["global_watermark"],
                "strategy": point["strategy"],
            }
        )
    partition_rows = [partitions[label] for label in sorted(partitions)]

    window_rows: list[dict[str, Any]] = []
    late_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for window in sorted(
        windows,
        key=lambda item: (
            0 if item["status"] == "finalized" else 1,
            int(item["finalization_order"]) if item["finalization_order"] is not None else 10**12,
            item["window_id"],
        ),
    ):
        content = window.get("content") or {}
        stats = content.get("stats") or {}
        window_rows.append(
            {
                "window_id": window["window_id"],
                "rule_id": window["rule_id"],
                "partition": window["partition"],
                "start_ms": window["start_ms"],
                "end_ms": window["end_ms"],
                "finalization_order": window["finalization_order"],
                "trigger_seq": window["trigger_seq"],
                "strategy": window["strategy"],
                "version": window["version"],
                "status": window["status"],
                "count": stats.get("count"),
                "sum_fixed": stats.get("sum_fixed"),
                "digest": window["digest"],
            }
        )
        decisions = content.get("late_decisions") or {}
        for decision in list(decisions.get("accepted", [])) + list(
            decisions.get("discarded", [])
        ):
            late_rows.append(
                {
                    **decision,
                    "window_order": window["finalization_order"],
                    "window_status": window["status"],
                }
            )
        errors = content.get("errors") or {}
        if errors:
            error_rows.append(
                {
                    "window_id": window["window_id"],
                    "partition": window["partition"],
                    "start_ms": window["start_ms"],
                    "end_ms": window["end_ms"],
                    **errors,
                }
            )
    late_rows.sort(
        key=lambda item: (
            int(item["window_order"]) if item["window_order"] is not None else 10**12,
            int(item["event_seq"]),
        )
    )
    error_rows.sort(key=lambda item: (item["start_ms"], item["window_id"]))

    return {
        "ledger": str(Path(ledger_path).resolve()),
        "meta": meta,
        "partition_rows": partition_rows,
        "window_rows": window_rows,
        "late_rows": late_rows,
        "displacements": displacements,
        "error_rows": error_rows,
        "snapshots": snapshots,
    }


def pretty_partition(partition: Any) -> str:
    return json.dumps(partition, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

"""Stable, Tk-independent view model for the optional desktop inspector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .changes import load_changes
from .engine import run_full
from .incremental import run_incremental
from .ledger import Ledger, ledger_summary
from .logging_utils import read_log_codes
from .spec import load_spec


def collect_view_model(
    *,
    spec_path: str | Path,
    changes_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    baseline = run_full(spec)
    state = baseline
    invalidation: dict[str, Any] = {
        "bucket_columns": [],
        "packs": [],
        "trace": [],
        "mode": "baseline",
    }
    equivalence = None
    if changes_path:
        changes = load_changes(changes_path, spec)
        outcome = run_incremental(spec, changes, baseline=baseline)
        state = outcome.state
        invalidation = {
            "mode": "incremental",
            **outcome.report["invalidation"],
        }
        equivalence = outcome.report["equivalence"]

    schedule = {
        entry["pack_id"]: entry for entry in state.result["schedule"]["entries"]
    }
    packs = []
    for pack in sorted(state.packs, key=lambda item: item.pack_id):
        timing = schedule.get(pack.pack_id, {})
        packs.append(
            {
                "pack_id": pack.pack_id,
                "source_id": pack.source_id,
                "row_start": pack.row_start,
                "row_end": pack.row_end,
                "rows": pack.row_count,
                "width_used": pack.width_used,
                "cost_micros": pack.cost_micros,
                "queue_start_ms": timing.get("queue_start_ms", 0.0),
                "start_ms": timing.get("start_ms", 0.0),
                "end_ms": timing.get("end_ms", 0.0),
                "wait_ms": timing.get("wait_ms", 0.0),
                "digest": pack.digest,
            }
        )
    ledger: dict[str, Any] = {"format_version": None, "batches": [], "content_digest": None}
    if ledger_path and Path(ledger_path).exists():
        with Ledger(ledger_path) as ledger_handle:
            ledger = ledger_summary(ledger_handle)
    codes = read_log_codes(log_path)
    dependencies = state.result["graph"]["edges"]
    return {
        "summary": {
            "spec_digest": spec.digest,
            "content_digest": state.result["content_digest"],
            "source_count": len(spec.sources),
            "pack_count": len(state.packs),
            "row_count": state.result["merged"]["global"]["row_count"],
            "width_used": state.result["merged"]["global"]["width_used"],
            "max_active_width": state.result["schedule"]["max_active_width"],
            "global_width_budget": state.result["schedule"]["global_width_budget"],
            "cost_total_micros": state.result["cost_ledger"]["total_cost_micros"],
            "cost_within_budget": state.result["cost_ledger"]["within_budget"],
            "equivalence": equivalence,
        },
        "dependencies": sorted(
            dependencies,
            key=lambda item: (item["from"], item["to"]),
        ),
        "invalidation": invalidation,
        "packs": packs,
        "ledger": ledger,
        "codes": sorted(codes, key=lambda item: (item["seq"], item["code"])),
    }

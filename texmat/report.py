"""Diagnostic JSON report assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import EngineRunResult, TextureEngine
from .errors import Failure

MAX_PIXEL_LIST = 2000


def build_report(engine: TextureEngine, run: EngineRunResult,
                 decode_failures: list[Failure], diagnostics: dict[str, Any] | None,
                 *, report_path: str | None = None,
                 invalidation_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    boundary_pixels = diagnostics.get("boundary_sensitive_pixels", []) if diagnostics else []
    truncated = len(boundary_pixels) > MAX_PIXEL_LIST
    boundary_pixels = boundary_pixels[:MAX_PIXEL_LIST]

    boundary_ties: list[dict[str, Any]] = []
    manifest = engine.store.load_manifest()
    if manifest is not None:
        for node_id, state in sorted(manifest.artifacts.items()):
            if not node_id.startswith("channel:"):
                continue
            data = engine.store.load_npz(state.payload_path)
            tie = data.get("tie")
            boundary = data.get("boundary")
            if tie is not None and tie.size:
                boundary_ties.append({"node": node_id, "tie_pixel_count": int(tie.shape[0]),
                                      "sample_pixels": tie[:50].tolist()})
            if boundary is not None and boundary.size:
                boundary_ties.append({"node": node_id, "rounding_boundary_count": int(boundary.shape[0]),
                                      "sample_pixels": boundary[:50].tolist()})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if run.complete else ("signal_stopped" if run.stopped_by_signal else "pending"),
        "config_hash": engine.config_hash_value,
        "fractional_bits": engine.config.fractional_bits,
        "reference_bits": engine.config.fractional_bits + 1,
        "blend_mode": engine.config.blend_mode,
        "rounding": engine.config.rounding,
        "quantize": engine.config.quantize,
        "resolution": engine.config.active_tier,
        "workers": engine.config.workers,
        "point_budget": engine.config.point_budget,
        "batches": run.batches,
        "manifest_version": run.manifest_version,
        "recovery_point": run.recovery,
        "config_mismatch": run.mismatch,
        "error_attribution": diagnostics,
        "boundary_sensitive_pixels": boundary_pixels,
        "boundary_sensitive_truncated": truncated,
        "boundary_and_tie_annotations": boundary_ties,
        "cache": {
            "hits": len(run.reused),
            "reused_nodes": run.reused,
            "recomputed_nodes": run.recomputed,
            "invalidation_plan": invalidation_plan,
        },
        "budget": {
            "remaining": run.budget_remaining,
            "events": run.budget_events,
            "pending_nodes": run.pending,
        },
        "failures": [failure.to_dict() for failure in decode_failures] + run.failures,
        "cursor": run.cursor,
    }
    if report_path is not None:
        import json
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, sort_keys=True, indent=2, ensure_ascii=False)
    return report

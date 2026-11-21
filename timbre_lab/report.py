from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_report(
    *,
    semantic: dict[str, Any],
    config_hash: str,
    command_events: list[dict[str, Any]],
    snapshot: dict[str, Any] | None,
    resume_info: dict[str, Any],
    stop_requested: bool,
    elapsed_ms: float,
) -> dict[str, Any]:
    failure_codes = sorted({str(item.get("code")) for item in semantic.get("failures", []) if item.get("code")})
    full_comparison = semantic.get("full_rebuild_comparison") or semantic.get("initialization", {}).get("full_rebuild_comparison")
    command_summaries = []
    for event in command_events:
        command_summaries.append(
            {
                "command_id": event.get("command_id"),
                "command_index": event.get("command_index"),
                "status": event.get("status"),
                "code": event.get("code"),
                "command": event.get("command"),
                "impact": event.get("impact"),
                "preview": event.get("preview"),
                "elapsed_ms": event.get("elapsed_ms"),
                "incremental_vs_full_equal": event.get("incremental_vs_full_equal"),
                "budget_response_code": event.get("budget_response_code"),
                "cache_audit": event.get("semantic", {}).get("cache_audit"),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "stopped_by_signal" if stop_requested else "complete",
        "run_elapsed_ms": elapsed_ms,
        "config_hash": config_hash,
        "resume": resume_info,
        "counts": {
            "segments": len(semantic.get("segments", [])),
            "speakers": len(semantic.get("speaker_vectors", [])),
            "vector_dimensions": len(semantic.get("quantization_plan", [])),
            "assignments": len(semantic.get("assignments", [])),
            "failed_items": len(semantic.get("failures", [])),
        },
        "incremental_full_equal": bool(full_comparison.get("equal")) if isinstance(full_comparison, dict) else None,
        "incremental_full_comparison": full_comparison,
        "segments": semantic.get("segments", []),
        "segment_statistics": semantic.get("segment_statistics", []),
        "segment_assignments": semantic.get("assignments", []),
        "assignment_conflicts": semantic.get("assignment_conflicts", []),
        "boundary_samples": semantic.get("boundary_samples", []),
        "bridge_conflicts": semantic.get("bridge_conflicts", []),
        "speaker_vectors": semantic.get("speaker_vectors", []),
        "speaker_groups": semantic.get("speaker_groups", []),
        "style_summary": semantic.get("style_summary", {}),
        "quantization_plan": semantic.get("quantization_plan", []),
        "quantization_evidence": [
            {
                "segment_id": segment["segment_id"],
                "dimensions": [
                    {
                        "name": dimension["name"],
                        "bit_width": dimension["bit_width"],
                        "step": dimension["step"],
                        "integer": dimension["integer"],
                        "normalized": dimension["normalized"],
                        "residual": dimension["residual"],
                    }
                    for dimension in segment["quantization"]
                ],
            }
            for segment in semantic.get("segments", [])
        ],
        "cache_reuse_and_invalidation": semantic.get("cache_audit", {}),
        "cache_references": semantic.get("cache_refs", {}),
        "command_events": command_summaries,
        "snapshot": snapshot,
        "failures": semantic.get("failures", []),
        "failure_codes": failure_codes,
    }


def write_report(output_dir: str | Path, report: dict[str, Any]) -> Path:
    path = Path(output_dir) / "report.json"
    _atomic_json(path, report)
    return path


def write_fatal_report(output_dir: str | Path, *, code: str, message: str, details: dict[str, Any] | None = None) -> Path:
    return write_report(
        output_dir,
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "failure_codes": [code],
            "failures": [{"code": code, "message": message, "details": details or {}}],
        },
    )

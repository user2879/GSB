"""Human-readable audit exports derived from immutable snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .detection import detection_payload
from .evaluation import evaluation_payload
from .io import atomic_write_json
from .model import Snapshot
from .repair import repair_payload


def export_snapshot(snapshot: Snapshot, output_dir: Path, extra: dict[str, Any] | None = None) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_path = output_dir / "detection.json"
    repair_path = output_dir / "repair.json"
    evaluation_path = output_dir / "evaluation.json"
    summary_path = output_dir / "summary.json"
    atomic_write_json(detection_path, detection_payload(snapshot.detection, snapshot.evidence, snapshot.record))
    atomic_write_json(repair_path, repair_payload(snapshot.to_meta_dict(), snapshot.record, snapshot.repair))
    atomic_write_json(evaluation_path, evaluation_payload(snapshot.evaluation))
    summary = {
        "schema": "mocap-run-summary/v1",
        "version": snapshot.version.to_dict(),
        "frame_count": snapshot.frame_count,
        "fps": snapshot.record.fps,
        "thresholds": snapshot.thresholds.to_dict(),
        "keyframes": sorted(snapshot.keyframes),
        "suppressed_auto": sorted(snapshot.suppressed_auto),
        "detection": snapshot.detection.comparison,
        "evaluation": snapshot.evaluation.summary,
        "closure": snapshot.closure.to_dict() if snapshot.closure else None,
        "consistency": snapshot.consistency.to_dict() if snapshot.consistency else None,
        "artifact_hashes": {
            "record": snapshot.record.content_sha256,
            "truth": snapshot.truth.content_sha256,
            "detection": snapshot.detection.detection_hash,
            "repair": snapshot.repair.repair_hash,
            "evaluation": snapshot.evaluation.evaluation_hash,
            "content": snapshot.content_hash,
        },
        "extra": extra or {},
    }
    atomic_write_json(summary_path, summary)
    closure_path = None
    consistency_path = None
    if snapshot.closure:
        closure_path = output_dir / "closure.json"
        atomic_write_json(closure_path, snapshot.closure.to_dict())
    if snapshot.consistency:
        consistency_path = output_dir / "consistency.json"
        atomic_write_json(consistency_path, snapshot.consistency.to_dict())
    return {
        "detection": str(detection_path),
        "repair": str(repair_path),
        "evaluation": str(evaluation_path),
        "summary": str(summary_path),
        "closure": str(closure_path) if closure_path else "",
        "consistency": str(consistency_path) if consistency_path else "",
    }

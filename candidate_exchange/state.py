from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import CanonicalCandidate, FailureRecord, Track, UnifiedFrame
from .util import read_json, stable_hash


def candidate_to_state(candidate: CanonicalCandidate) -> dict[str, Any]:
    return {
        "series_name": candidate.series_name,
        "frame_index": candidate.frame_index,
        "class_code": candidate.class_code,
        "class_name": candidate.class_name,
        "x1_q": candidate.x1_q,
        "y1_q": candidate.y1_q,
        "x2_q": candidate.x2_q,
        "y2_q": candidate.y2_q,
        "score_micro": candidate.score_micro,
        "precision_label": candidate.precision_label,
        "source_version": candidate.source_version,
        "source_ordinal": candidate.source_ordinal,
        "missing_fields": list(candidate.missing_fields),
        "extra": candidate.extra,
    }


def candidate_from_state(value: dict[str, Any]) -> CanonicalCandidate:
    return CanonicalCandidate(
        series_name=value["series_name"],
        frame_index=int(value["frame_index"]),
        class_code=int(value["class_code"]),
        class_name=value["class_name"],
        x1_q=int(value["x1_q"]),
        y1_q=int(value["y1_q"]),
        x2_q=int(value["x2_q"]),
        y2_q=int(value["y2_q"]),
        score_micro=int(value["score_micro"]),
        precision_label=value["precision_label"],
        source_version=value["source_version"],
        source_ordinal=int(value["source_ordinal"]),
        missing_fields=tuple(value.get("missing_fields", [])),
        extra=dict(value.get("extra", {})),
    )


def frame_to_state(frame: UnifiedFrame) -> dict[str, Any]:
    return {
        "series_name": frame.series_name,
        "frame_index": frame.frame_index,
        "image_path": frame.image_path,
        "image_sha256": frame.image_sha256,
        "image_size": list(frame.image_size),
        "candidate_key": frame.candidate_key,
        "content_fingerprint": frame.content_fingerprint,
        "status": frame.status,
        "candidates": [candidate_to_state(item) for item in frame.candidates],
    }


def frame_from_state(value: dict[str, Any]) -> UnifiedFrame:
    return UnifiedFrame(
        series_name=value["series_name"],
        frame_index=int(value["frame_index"]),
        image_path=value.get("image_path", ""),
        image_sha256=value.get("image_sha256", ""),
        image_size=tuple(value.get("image_size", [0, 0])),
        candidate_key=value.get("candidate_key", ""),
        content_fingerprint=value.get("content_fingerprint", ""),
        status=value.get("status", "ok"),
        candidates=[candidate_from_state(item) for item in value.get("candidates", [])],
    )


def failure_to_state(failure: FailureRecord) -> dict[str, Any]:
    return failure.to_dict()


def failure_from_state(value: dict[str, Any]) -> FailureRecord:
    return FailureRecord(
        series_name=value["series_name"],
        frame_index=value.get("frame_index"),
        stage=value["stage"],
        error_code=value["error_code"],
        exception_type=value.get("exception_type", "Exception"),
        message=value.get("message", ""),
        injection_id=value.get("injection_id"),
        details=dict(value.get("details", {})),
    )


def track_to_state(track: Track) -> dict[str, Any]:
    return track.to_dict()


def load_sidecar_fingerprints(data_dir: str | Path) -> tuple[dict[tuple[str, int], str], str]:
    root = Path(data_dir)
    candidates = [
        root / "sidecar_metadata.json",
        root / "sidecar" / "candidates.v2_to_v1.sidecar.json",
    ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        return {}, stable_hash({"sidecar": "absent"})
    try:
        payload = read_json(path)
    except Exception:
        return {}, stable_hash({"sidecar": "invalid", "path": str(path)})
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    per_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            series_name = entry.get("series_name")
            frame_index = entry.get("frame_index")
            if isinstance(series_name, str) and isinstance(frame_index, int):
                per_frame.setdefault((series_name, frame_index), []).append(entry)
    fingerprints = {
        key: stable_hash(items) for key, items in sorted(per_frame.items())
    }
    return fingerprints, stable_hash(
        {
            "path": path.name,
            "declared_hash": payload.get("sidecar_metadata_hash")
            if isinstance(payload, dict)
            else None,
            "frame_fingerprints": fingerprints,
        }
    )

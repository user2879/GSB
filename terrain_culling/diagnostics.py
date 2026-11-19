"""Frame diagnostics, canonical hashing and first-divergence localization."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .errors import TrackDivergenceError


def _sanitize(obj):
    """Replace non-finite floats with None so output stays strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return _sanitize(obj.item())
    return obj


def canonical_bytes(obj) -> bytes:
    return json.dumps(_sanitize(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def content_hash(obj) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def frame_digest(frame_report: dict) -> dict:
    """Compact per-block decision signature; not based on counts alone."""
    signatures = {}
    for decision in frame_report["decisions"]:
        signatures[decision["block_id"]] = {
            "state": decision["state"],
            "total_error": round(decision["total_error"], 12),
            "screen_error_px": round(decision["screen_error_px"], 12),
            "hiz_level": decision["hiz_level"],
            "reasons": list(decision["reasons"]),
        }
    return signatures


def find_first_divergence(reports_a: list[dict], reports_b: list[dict],
                          label_a: str = "track_a",
                          label_b: str = "track_b") -> dict:
    """Locate the first frame and block whose culling decision differs.

    Equality is the full per-block decision signature, so two runs with the same
    visible/culled counts but different block assignments still diverge.
    """
    limit = min(len(reports_a), len(reports_b))
    for frame_index in range(limit):
        sig_a = frame_digest(reports_a[frame_index])
        sig_b = frame_digest(reports_b[frame_index])
        block_ids = sorted(set(sig_a) | set(sig_b))
        for block_id in block_ids:
            if sig_a.get(block_id) != sig_b.get(block_id):
                decision_a = next((d for d in reports_a[frame_index]["decisions"]
                                   if d["block_id"] == block_id), None)
                decision_b = next((d for d in reports_b[frame_index]["decisions"]
                                   if d["block_id"] == block_id), None)
                result = {
                    "diverged": True,
                    "frame": frame_index,
                    "block_id": block_id,
                    label_a: decision_a,
                    label_b: decision_b,
                    "camera_a": reports_a[frame_index]["camera"],
                    "camera_b": reports_b[frame_index]["camera"],
                }
                raise TrackDivergenceError(
                    f"first divergence at frame {frame_index} block {block_id}",
                    context=result)
    if len(reports_a) != len(reports_b):
        raise TrackDivergenceError(
            "tracks differ in frame count",
            context={"frames_a": len(reports_a), "frames_b": len(reports_b)})
    return {"diverged": False, "frames_compared": limit}


def summarize_frame(frame_report: dict) -> dict:
    """Human-readable single-line summary plus key diagnostics."""
    counts = frame_report["counts"]
    budget = frame_report["budget"]
    errors = frame_report["error_summary"]
    streaming = frame_report["streaming"]
    return {
        "frame": frame_report["frame"],
        "visible": counts["visible"],
        "occluded": counts["occluded"],
        "uncertain": counts["uncertain"],
        "frustum_out": counts["frustum_out"],
        "max_relative_error": errors["max_relative_error"],
        "budget_remaining": errors["budget_remaining"],
        "over_budget": errors["over_budget_count"],
        "memory_util": round(budget["memory_utilization"], 6),
        "vram_util": round(budget["vram_utilization"], 6),
        "loads": streaming["loads"],
        "retains": streaming["retains"],
        "evictions": streaming["evictions"],
        "degradation_step": frame_report["degradation"]["step"],
        "hiz_level": frame_report["hiz"]["selected_level"],
    }

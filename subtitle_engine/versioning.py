"""Canonical version and block digests."""

from __future__ import annotations

import hashlib
from typing import Any

from ._numpy import np
from .incremental import IncrementalPlan
from .spec import Edit, Spec
from .util import canonical_bytes, sha256_bytes


ENGINE_FORMAT_VERSION = 1


def _event_payload(event: Any) -> dict[str, Any]:
    return {
        "id": event.id,
        "start_ms": event.start_ms,
        "end_ms": event.end_ms,
        "text": event.text,
        "speaker": event.speaker,
        "language": event.language,
        "max_chars_per_sec": event.max_chars_per_sec,
        "critical_sound": event.critical_sound,
        "anchor": {
            "x": event.anchor.x,
            "y": event.anchor.y,
            "horizontal": event.anchor.horizontal,
            "vertical": event.anchor.vertical,
        },
    }


def _style_payload(style: Any) -> dict[str, Any]:
    return {
        "font_size": style.font_size,
        "line_height": style.line_height,
        "max_lines": style.max_lines,
        "padding": style.padding,
        "contrast_enhance": style.contrast_enhance,
        "mask_opacity": style.mask_opacity,
    }


def _edit_payload(edit: Edit) -> dict[str, Any]:
    if edit.op == "set_text":
        return {"op": edit.op, "event_id": edit.event_id, "text": edit.value}
    if edit.op == "set_time":
        return {
            "op": edit.op,
            "event_id": edit.event_id,
            "start_ms": edit.start_ms,
            "end_ms": edit.end_ms,
        }
    return {"op": edit.op, "field": edit.field, "value": edit.value}


def compute_version(spec: Spec, plan: IncrementalPlan) -> str:
    """Hash every output-affecting input, including the ordered edit script.

    Directories, concurrency, and memory budget are deliberately excluded so
    those execution choices cannot create pixel-incompatible versions.
    """

    payload = {
        "engine_format": ENGINE_FORMAT_VERSION,
        "schema": "subtitle-engine/v1",
        "frame": {
            "width": spec.frame.width,
            "height": spec.frame.height,
            "frame_ms": spec.frame.frame_ms,
            "background": spec.frame.background,
        },
        "safe_area": spec.safe_area.as_list(),
        "style": _style_payload(spec.style),
        "events": [_event_payload(event) for event in spec.events],
        "edits": [_edit_payload(edit) for edit in spec.edits],
        "duration_ms": spec.duration_ms,
        "block_ms": spec.runtime.block_ms,
        "final_state": {
            "style": _style_payload(plan.final_style),
            "events": [_event_payload(event) for event in plan.final_events],
            "duration_ms": plan.final_duration_ms,
        },
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def block_digest(version: str, index: int, frames: "np.ndarray", metadata: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(version.encode("ascii"))
    hasher.update(b"\0")
    hasher.update(str(index).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(str(int(frames.shape[0])).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(frames.tobytes(order="C"))
    hasher.update(b"\0")
    hasher.update(canonical_bytes(metadata))
    return hasher.hexdigest()


def frames_digest(frames: "np.ndarray") -> str:
    return sha256_bytes(frames.tobytes(order="C"))

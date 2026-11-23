"""v2 JSON exchange format with named timbres, float frames and passthrough."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig, canonical_json
from .errors import E_UNKNOWN_TIMBRE, E_V2_SCHEMA, FailureLog, PipelineError
from .model import Segment

SCHEMA_VERSION = "vc-exchange-v2"
_NUMERIC = ("segment", "start_frame", "end_frame", "intensity")


def parse_v2(path: str | Path, config: PipelineConfig,
             failures: FailureLog) -> list[Segment]:
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict) or not isinstance(doc.get("segments"), list):
        raise PipelineError(E_V2_SCHEMA, "v2 document needs a 'segments' array")
    segments: list[Segment] = []
    for position, entry in enumerate(doc["segments"]):
        if not isinstance(entry, dict):
            failures.add(E_V2_SCHEMA, "load", line=position,
                         field="segment", message="entry is not an object")
            continue
        bad = next((f for f in _NUMERIC if not isinstance(
            entry.get(f), (int, float)) or isinstance(entry.get(f), bool)), "")
        if bad:
            failures.add(E_V2_SCHEMA, "load", line=position, field=bad,
                         message="missing or non-numeric field")
            continue
        timbre = entry.get("timbre")
        if not isinstance(timbre, str):
            failures.add(E_V2_SCHEMA, "load", segment=int(entry["segment"]),
                         line=position, field="timbre",
                         message="missing or non-string timbre")
            continue
        if timbre not in config.timbre_table:
            failures.add(E_UNKNOWN_TIMBRE, "load",
                         segment=int(entry["segment"]), line=position,
                         field="timbre",
                         message=f"timbre {timbre!r} not in code table")
            continue
        if not 0.0 <= float(entry["intensity"]) <= 1.0:
            failures.add(E_V2_SCHEMA, "load", segment=int(entry["segment"]),
                         line=position, field="intensity",
                         message="intensity outside [0,1]")
            continue
        passthrough = entry.get("passthrough", {})
        if not isinstance(passthrough, dict):
            failures.add(E_V2_SCHEMA, "load", segment=int(entry["segment"]),
                         line=position, field="passthrough",
                         message="passthrough must be an object")
            continue
        segments.append(
            Segment(
                index=int(entry["segment"]),
                start_frame=float(entry["start_frame"]),
                end_frame=float(entry["end_frame"]),
                intensity=float(entry["intensity"]),
                timbre=timbre,
                source_version="v2",
                precision_tag=str(entry.get("precision_tag", "float-native")),
                start_residual=float(entry.get("start_residual", 0.0)),
                end_residual=float(entry.get("end_residual", 0.0)),
                missing=bool(entry.get("missing", False)),
                source_line=position,
                timbre_code=config.name_to_code(timbre),
                passthrough=dict(passthrough),
            )
        )
    segments.sort(key=lambda s: s.index)
    return segments


def _entry(seg: Segment) -> dict[str, Any]:
    return {
        "segment": seg.index,
        "start_frame": seg.start_frame,
        "end_frame": seg.end_frame,
        "intensity": seg.intensity,
        "timbre": seg.timbre,
        "precision_tag": seg.precision_tag,
        "start_residual": seg.start_residual,
        "end_residual": seg.end_residual,
        "missing": seg.missing,
        "passthrough": seg.passthrough,
    }


def render_v2(segments: list[Segment]) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "segments": [_entry(s) for s in sorted(segments, key=lambda s: s.index)],
    }


def write_v2(path: str | Path, segments: list[Segment]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        canonical_json(render_v2(segments)) + "\n", encoding="utf-8"
    )

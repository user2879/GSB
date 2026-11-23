"""Bidirectional v1 <-> v2 conversion, sidecars and precision-loss metrics."""
from __future__ import annotations

import json
from pathlib import Path

from .config import PipelineConfig, canonical_json
from .model import Segment
from .v1 import write_v1
from .v2 import write_v2

SIDECAR_SCHEMA = "vc-sidecar-v1"


def upgrade_v1_to_v2(segments: list[Segment],
                     config: PipelineConfig) -> list[Segment]:
    """Exact upgrade: permille / 1000 is representable, code -> named label."""
    upgraded = []
    for seg in segments:
        if seg.source_version != "v1":
            upgraded.append(seg.copy())
            continue
        code = seg.timbre_code if seg.timbre_code is not None else \
            config.name_to_code(seg.timbre)
        timbre = config.code_to_name(int(code)) if code is not None else seg.timbre
        upgraded.append(
            Segment(
                index=seg.index,
                start_frame=float(int(round(seg.start_frame))),
                end_frame=float(int(round(seg.end_frame))),
                intensity=seg.intensity_permille / 1000.0,
                timbre=timbre,
                source_version="v2",
                intensity_permille=seg.intensity_permille,
                precision_tag="exact-from-v1-permille",
                start_residual=0.0,
                end_residual=0.0,
                missing=seg.missing,
                source_line=seg.source_line,
                timbre_code=code,
                passthrough=dict(seg.passthrough),
            )
        )
    return upgraded


def downgrade_v2_to_v1(segments: list[Segment],
                       config: PipelineConfig) -> tuple[list[Segment], dict]:
    """Round float fields back to integers; preserve every loss in a sidecar."""
    downgraded: list[Segment] = []
    sidecar_entries: list[dict] = []
    for seg in segments:
        start_int = int(round(seg.start_frame))
        end_int = int(round(seg.end_frame))
        permille = int(round(seg.intensity * 1000.0))
        code = config.name_to_code(seg.timbre)
        lowered = Segment(
            index=seg.index,
            start_frame=float(start_int),
            end_frame=float(end_int),
            intensity=permille / 1000.0,
            timbre=seg.timbre,
            source_version="v1",
            intensity_permille=permille,
            precision_tag="rounded-to-v1",
            start_residual=seg.start_frame - start_int,
            end_residual=seg.end_frame - end_int,
            missing=seg.missing,
            source_line=seg.source_line,
            timbre_code=code,
            passthrough={},
        )
        downgraded.append(lowered)
        sidecar_entries.append({
            "segment": seg.index,
            "origin_version": "v2",
            "named_timbre": seg.timbre,
            "timbre_code": code,
            "original_start_frame": seg.start_frame,
            "original_end_frame": seg.end_frame,
            "rounded_start_frame": start_int,
            "rounded_end_frame": end_int,
            "start_residual": lowered.start_residual,
            "end_residual": lowered.end_residual,
            "original_intensity": seg.intensity,
            "rounded_permille": permille,
            "intensity_residual": seg.intensity - permille / 1000.0,
            "precision_tag": seg.precision_tag,
            "discarded_passthrough": seg.passthrough,
        })
    sidecar = {"schema": SIDECAR_SCHEMA, "segments": sidecar_entries}
    return downgraded, sidecar


def restore_v1_with_sidecar(segments: list[Segment],
                            sidecar: dict) -> list[Segment]:
    """Rebuild the pre-downgrade v2 representation from v1 + sidecar."""
    by_index = {entry["segment"]: entry for entry in sidecar["segments"]}
    restored = []
    for seg in segments:
        entry = by_index.get(seg.index)
        if entry is None:
            restored.append(seg.copy())
            continue
        restored.append(Segment(
            index=seg.index,
            start_frame=float(entry["original_start_frame"]),
            end_frame=float(entry["original_end_frame"]),
            intensity=float(entry["original_intensity"]),
            timbre=str(entry["named_timbre"]),
            source_version="v2",
            intensity_permille=seg.intensity_permille,
            precision_tag=str(entry["precision_tag"]),
            start_residual=float(entry["start_residual"]),
            end_residual=float(entry["end_residual"]),
            missing=seg.missing,
            source_line=seg.source_line,
            timbre_code=seg.timbre_code,
            passthrough=dict(entry["discarded_passthrough"]),
        ))
    return restored


def write_sidecar(path: str | Path, sidecar: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(sidecar) + "\n", encoding="utf-8")


def read_sidecar(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def conversion_pair_v2_to_v1(segments: list[Segment], config: PipelineConfig,
                             v1_path: str | Path,
                             sidecar_path: str | Path) -> dict:
    downgraded, sidecar = downgrade_v2_to_v1(segments, config)
    write_v1(v1_path, downgraded, config)
    write_sidecar(sidecar_path, sidecar)
    return sidecar


def conversion_pair_v1_to_v2(segments: list[Segment], config: PipelineConfig,
                             v2_path: str | Path) -> list[Segment]:
    upgraded = upgrade_v1_to_v2(segments, config)
    write_v2(v2_path, upgraded)
    return upgraded


def precision_loss(original: list[Segment],
                   rounded: list[Segment]) -> dict:
    """Quantify rounding loss between a v2 segment list and its v1 downgrade."""
    by_index = {s.index: s for s in rounded}
    intensity_abs: list[float] = []
    start_abs: list[float] = []
    end_abs: list[float] = []
    per_segment = []
    for seg in original:
        low = by_index[seg.index]
        di = abs(seg.intensity - low.intensity)
        ds = abs(seg.start_frame - low.start_frame)
        de = abs(seg.end_frame - low.end_frame)
        intensity_abs.append(di)
        start_abs.append(ds)
        end_abs.append(de)
        per_segment.append({
            "segment": seg.index,
            "intensity_abs_error": di,
            "start_frame_abs_error": ds,
            "end_frame_abs_error": de,
        })
    def _stats(values: list[float]) -> dict:
        if not values:
            return {"max_abs": 0.0, "mean_abs": 0.0}
        return {"max_abs": max(values), "mean_abs": sum(values) / len(values)}
    return {
        "segment_count": len(original),
        "intensity": _stats(intensity_abs),
        "start_frame": _stats(start_abs),
        "end_frame": _stats(end_abs),
        "per_segment": per_segment,
    }

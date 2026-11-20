"""Cross-version semantic class mapping for packet score columns."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _packet_codebook(
    class_codebook_version: str,
    config: dict[str, Any],
    codebooks: dict[str, Any],
) -> list[dict[str, Any]] | None:
    if class_codebook_version == config["class_codebook_version"]:
        entries = config["class_codebook"]
    else:
        catalog = codebooks.get("codebooks", {})
        item = catalog.get(class_codebook_version) if isinstance(catalog, dict) else None
        entries = item.get("entries") if isinstance(item, dict) else None
    if not isinstance(entries, list):
        return None
    valid_entries = [entry for entry in entries if isinstance(entry, dict)]
    if len(valid_entries) != len(entries):
        return None
    return sorted(valid_entries, key=lambda entry: entry.get("code", 10**9))


def _probability_space(row: list[float], score_type: str) -> list[float]:
    values = np.asarray(row, dtype=np.float64)
    if score_type == "logit":
        shifted = values - float(np.max(values))
        exponentials = np.exp(shifted)
        total = float(np.sum(exponentials))
        return [float(value) for value in (exponentials / total)] if total > 0 else [0.0] * len(row)
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= 0.0:
        return [0.0] * len(row)
    return [float(value) for value in values / total]


def map_packet_classes(
    parse_payload: dict[str, Any],
    config: dict[str, Any],
    codebooks: dict[str, Any],
) -> dict[str, Any]:
    """Map packet columns by codebook entry name, never by code position."""

    classes = list(config["classes"])
    class_index = {name: index for index, name in enumerate(classes)}
    entries = _packet_codebook(
        parse_payload["class_codebook_version"], config, codebooks
    )
    mappings: list[dict[str, Any]] = []
    unmapped_entries: list[dict[str, Any]] = []
    if entries is not None:
        for entry in entries:
            name = entry.get("name")
            global_index = class_index.get(name) if isinstance(name, str) else None
            mapping = {
                "code": entry.get("code"),
                "name": name,
                "global_index": global_index,
            }
            mappings.append(mapping)
            if global_index is None:
                unmapped_entries.append(mapping)

    frame_count = int(parse_payload["frame_count"])
    mapped_rows: list[list[float]] = []
    as_is_rows: list[list[float]] = []
    mapping_residuals: list[float] = []
    unmapped_frames: list[dict[str, Any]] = []
    unmapped_row_indices: list[int] = []
    no_codebook = entries is None
    score_type = parse_payload["raw_score_type"]

    for frame_index, row in enumerate(parse_payload["frames"]):
        as_is = [float(value) for value in row]
        as_is_rows.append(as_is)
        if no_codebook or len(entries) != len(classes) or unmapped_entries:
            mapped = [0.0] * len(classes)
            unmapped_row_indices.append(frame_index)
            unmapped_frames.append(
                {
                    "original_frame_index": frame_index,
                    "original_codes": (
                        [entry.get("code") for entry in entries]
                        if entries is not None
                        else list(range(len(row)))
                    ),
                    "original_scores": as_is,
                    "original_codebook_version": parse_payload["class_codebook_version"],
                    "reason": "codebook_version_missing"
                    if no_codebook
                    else "codebook_entries_unmapped",
                }
            )
        else:
            mapped = [0.0] * len(classes)
            for column_index, mapping in enumerate(mappings):
                mapped[int(mapping["global_index"])] = as_is[column_index]
        as_is_probability = _probability_space(as_is, score_type)
        mapped_probability = _probability_space(mapped, score_type)
        residual = sum(
            abs(left - right) for left, right in zip(as_is_probability, mapped_probability)
        ) / len(classes)
        mapped_rows.append(mapped)
        mapping_residuals.append(float(residual))

    return {
        "packet_id": parse_payload["packet_id"],
        "source_file": parse_payload["source_file"],
        "source_score_type": parse_payload["raw_score_type"],
        "content_sha256": parse_payload["content_sha256"],
        "audio_file": parse_payload["audio_file"],
        "start_ms": parse_payload["start_ms"],
        "end_ms": parse_payload["end_ms"],
        "frame_shift_ms": parse_payload["frame_shift_ms"],
        "frame_grid_version": parse_payload["frame_grid_version"],
        "class_codebook_version": parse_payload["class_codebook_version"],
        "detector_version": parse_payload.get("detector_version"),
        "frame_count": frame_count,
        "mapping_entries": mappings,
        "mapping_by_name": not no_codebook and not unmapped_entries,
        "codebook_found": not no_codebook,
        "unmapped_entries": unmapped_entries,
        "unmapped_frame_count": len(unmapped_frames),
        "unmapped_frames": unmapped_frames,
        "unmapped_row_indices": unmapped_row_indices,
        "mapped_rows": mapped_rows,
        "as_is_rows": as_is_rows,
        "mapping_residuals": mapping_residuals,
        "injection_ids": parse_payload.get("injection_ids", []),
    }

"""Shard package loading, header compatibility and global grid alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import CANONICAL_CLASSES
from .errors import failure_entry
from .models import CompatRecord, LoadedShard

LOAD_STAGE = "load_align"


def _validate_payload(payload: dict[str, Any], native_shift: int) -> str | None:
    shard_id = str(payload.get("shard_id", "unknown"))
    probs = payload.get("frame_probabilities")
    if not isinstance(probs, list) or not probs:
        return f"{shard_id}: empty frame_probabilities"
    width = len(probs[0])
    expected = (int(payload["end_ms"]) - int(payload["start_ms"])) // native_shift + 1
    if len(probs) != expected:
        return f"{shard_id}: got {len(probs)} frames, expected {expected}"
    for row in probs:
        if len(row) != width:
            return f"{shard_id}: ragged probability rows"
        for value in row:
            if not isinstance(value, (int, float)) or value < 0.0 or value > 1.0:
                return f"{shard_id}: probability {value!r} outside [0, 1]"
    if len(payload.get("class_codes", [])) != width:
        return f"{shard_id}: class_codes width does not match probability columns"
    return None


def _map_codebook(
    payload: dict[str, Any], codebook: dict[str, str]
) -> tuple[np.ndarray | None, dict[str, str], list[str], str | None]:
    matrix = np.asarray(payload["frame_probabilities"], dtype=np.float64)
    codes = [str(code) for code in payload["class_codes"]]
    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    columns: list[int] = []
    for column, code in enumerate(codes):
        name = codebook.get(code)
        if name in CANONICAL_CLASSES:
            mapping[code] = name
            columns.append(column)
        else:
            unmapped.append(code)
    if not columns:
        return None, mapping, unmapped, "no canonical class columns survived mapping"
    canonical = np.zeros((matrix.shape[0], len(CANONICAL_CLASSES)), dtype=np.float64)
    for column in columns:
        name = mapping[codes[column]]
        canonical[:, CANONICAL_CLASSES.index(name)] = matrix[:, column]
    return canonical, mapping, unmapped, None


def _resample_to_grid(
    native: np.ndarray,
    *,
    start_ms: int,
    end_ms: int,
    native_shift: int,
    target_shift: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Linear interpolation onto target grid; residuals recorded per class."""

    native_count = native.shape[0]
    native_centers = start_ms + np.arange(native_count) * native_shift
    target_count = (end_ms - start_ms) // target_shift + 1
    target_centers = start_ms + np.arange(target_count) * target_shift
    positions = (target_centers - start_ms) / native_shift
    lower = np.floor(positions).astype(int)
    upper = np.clip(lower + 1, 0, native_count - 1)
    lower = np.clip(lower, 0, native_count - 1)
    weight = (positions - lower)[:, None]
    resampled = native[lower] * (1 - weight) + native[upper] * weight

    residuals: dict[str, float] = {}
    # Compare native values interpolated at the old native centers against
    # the newly sampled values expressed at the same physical times.
    old_at_native = native
    new_positions = (native_centers - start_ms) / target_shift
    nlower = np.floor(new_positions).astype(int)
    nupper = np.clip(nlower + 1, 0, target_count - 1)
    nlower = np.clip(nlower, 0, target_count - 1)
    nweight = (new_positions - nlower)[:, None]
    new_at_native = resampled[nlower] * (1 - nweight) + resampled[nupper] * nweight
    per_class = np.mean(np.abs(new_at_native - old_at_native), axis=0)
    for idx, name in enumerate(CANONICAL_CLASSES):
        residuals[name] = round(float(per_class[idx]), 6)
    return resampled, residuals


def load_shards(
    shard_dir: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    shift = int(config["audio"]["frame_shift_ms"])
    total_frames = int(config["audio"]["total_duration_ms"]) // shift + 1
    current_grid = config["classes"]["current_grid_version"]
    current_codebook = config["classes"]["current_codebook_version"]
    codebooks = config["classes"]["codebooks"]
    grids = config["classes"]["grids"]
    target_shift = int(grids[current_grid]["frame_shift_ms"])
    active_codebook = codebooks[current_codebook]

    failures: list[dict[str, Any]] = []
    alignment_residuals: list[dict[str, Any]] = []
    compat_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    loaded: list[LoadedShard] = []

    specs = list(manifest["shards"]) + list(manifest.get("invalid_shards", []))
    for spec in specs:
        shard_id = str(spec["shard_id"])
        path = shard_dir / f"{shard_id}.json"
        if shard_id in seen_ids:
            failures.append(
                failure_entry(shard_id, LOAD_STAGE, "E_SHARD_ID_DUPLICATE", "duplicate shard_id")
            )
            continue
        if not path.exists():
            failures.append(
                failure_entry(shard_id, LOAD_STAGE, "E_TIMELINE_GAP_OR_OVERLAP", f"missing package {path}")
            )
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        header = payload["header"]
        grid_version = str(header["frame_grid_version"])
        codebook_version = str(header["codebook_version"])
        native_shift = int(header["frame_shift_ms"])
        start_ms, end_ms = int(payload["start_ms"]), int(payload["end_ms"])

        length_error = _validate_payload(payload, native_shift)
        if length_error is not None:
            if "expected" in length_error:
                code = "E_SHARD_LENGTH_MISMATCH"
            else:
                code = "E_PROBABILITY_RANGE"
            failures.append(failure_entry(shard_id, LOAD_STAGE, code, length_error))
            continue

        if start_ms % shift != 0 or end_ms % shift != 0:
            residual = {
                "shard_id": shard_id,
                "start_remainder_ms": start_ms % shift,
                "end_remainder_ms": end_ms % shift,
                "frame_shift_ms": shift,
            }
            alignment_residuals.append(residual)
            failures.append(
                failure_entry(
                    shard_id,
                    LOAD_STAGE,
                    "E_FRAME_GRID_MISALIGN",
                    f"boundaries {start_ms}..{end_ms} not divisible by frame shift {shift}",
                )
            )
            continue

        # Semantic mapping uses the shard's OWN declared codebook, then maps
        # class names onto the canonical grid. Code points are never connected
        # directly across codebook versions.
        shard_codebook = codebooks[codebook_version]
        canonical, mapping, unmapped, map_error = _map_codebook(payload, shard_codebook)
        if map_error is not None:
            failures.append(
                failure_entry(shard_id, LOAD_STAGE, "E_CLASS_MAP_UNMAPPED", map_error)
            )
            continue
        if unmapped:
            failures.append(
                failure_entry(
                    shard_id,
                    LOAD_STAGE,
                    "E_CLASS_MAP_UNMAPPED",
                    f"codes {sorted(unmapped)} have no canonical class name",
                )
            )

        resampled = False
        residuals: dict[str, float] = {}
        if grid_version != current_grid:
            canonical, residuals = _resample_to_grid(
                canonical,
                start_ms=start_ms,
                end_ms=end_ms,
                native_shift=native_shift,
                target_shift=target_shift,
            )
            resampled = True
            failures.append(
                failure_entry(
                    shard_id,
                    LOAD_STAGE,
                    "E_GRID_VERSION_RESAMPLE",
                    f"{grid_version} resampled onto {current_grid}",
                )
            )

        expected_target = (end_ms - start_ms) // target_shift + 1
        if canonical.shape[0] != expected_target:
            failures.append(
                failure_entry(
                    shard_id,
                    LOAD_STAGE,
                    "E_SHARD_LENGTH_MISMATCH",
                    f"post-compat frame count {canonical.shape[0]} != {expected_target}",
                )
            )
            continue

        unavailable: dict[str, list[int]] = {}
        if unmapped:
            raw = np.asarray(payload["frame_probabilities"], dtype=np.float64)
            raw_codes = [str(code) for code in payload["class_codes"]]
            frame_hits: list[int] = []
            for column, code in enumerate(raw_codes):
                if code in unmapped:
                    frame_hits.extend(np.flatnonzero(raw[:, column] > 0.05).tolist())
            for code in sorted(unmapped):
                unavailable[code] = sorted(set(frame_hits))

        first_frame = start_ms // target_shift
        last_frame = end_ms // target_shift
        compat = CompatRecord(
            shard_id=shard_id,
            codebook_version=codebook_version,
            grid_version=grid_version,
            semantic_mapping=mapping,
            unmapped_codes=unmapped,
            resampled=resampled,
            resample_residuals=residuals,
            native_frame_shift_ms=native_shift,
            target_frame_shift_ms=target_shift,
        )
        compat_records.append(compat.to_dict())
        loaded.append(
            LoadedShard(
                shard_id=shard_id,
                start_ms=start_ms,
                end_ms=end_ms,
                first_frame=first_frame,
                last_frame=last_frame,
                matrix=canonical,
                compat=compat,
                unavailable_class_frames=unavailable,
                duplicate_frame_kept=None,
            )
        )
        seen_ids.add(shard_id)

    loaded.sort(key=lambda item: (item.start_ms, item.shard_id))
    timeline_errors: list[dict[str, Any]] = []
    # Only adjacent successfully loaded shards can form a gap/overlap; failed
    # packages already have their own business-code entry.
    for prev, curr in zip(loaded, loaded[1:]):
        if prev.end_ms != curr.start_ms:
            timeline_errors.append(
                failure_entry(
                    curr.shard_id,
                    LOAD_STAGE,
                    "E_TIMELINE_GAP_OR_OVERLAP",
                    f"{prev.shard_id} ends {prev.end_ms} but {curr.shard_id} starts {curr.start_ms}",
                )
            )
    failures.extend(timeline_errors)

    global_grid = np.zeros((total_frames, len(CANONICAL_CLASSES)), dtype=np.float64)
    ownership: list[str | None] = [None] * total_frames
    duplicate_notes: list[dict[str, Any]] = []
    for shard in loaded:
        for offset, frame_index in enumerate(range(shard.first_frame, shard.last_frame + 1)):
            if ownership[frame_index] is not None:
                # Strict half-open continuity normally prevents this; keep a
                # deterministic left-wins rule instead of silent overwrite.
                duplicate_notes.append(
                    {
                        "frame": frame_index,
                        "kept_shard_id": ownership[frame_index],
                        "dropped_shard_id": shard.shard_id,
                    }
                )
                shard.duplicate_frame_kept = ownership[frame_index]
                continue
            global_grid[frame_index] = shard.matrix[offset]
            ownership[frame_index] = shard.shard_id

    uncovered = [index for index, owner in enumerate(ownership) if owner is None]
    return {
        "global_grid": global_grid,
        "frame_ownership": ownership,
        "loaded_shards": loaded,
        "failures": failures,
        "alignment_residuals": alignment_residuals,
        "compat_records": compat_records,
        "duplicate_notes": duplicate_notes,
        "uncovered_frames": uncovered,
        "header_version_set": sorted(
            {
                f"{shard.compat.grid_version}|{shard.compat.codebook_version}"
                for shard in loaded
            }
        ),
        "completed_shard_ids": [shard.shard_id for shard in loaded],
    }

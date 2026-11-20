"""Video decode isolation and per-frame metric computation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .constants import FPS
from .quality import compute_frame_metric, missing_frame_metric
from .util import atomic_write_bytes, resolve_workspace_path
from .y4m import Y4MReader, Y4MFrame


def _shot_map(shots: list[dict]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for shot in shots:
        for frame_index in range(int(shot["start_frame"]), int(shot["end_frame"]) + 1):
            mapping[frame_index] = int(shot["shot_id"])
    return mapping


def _jump_for_frame(timestamp_jumps: dict[Any, Any], frame_index: int) -> float:
    if str(frame_index) in timestamp_jumps:
        return float(timestamp_jumps[str(frame_index)])
    if frame_index in timestamp_jumps:
        return float(timestamp_jumps[frame_index])
    return 0.0


def compute_actual_frames(
    workspace: Path,
    truth: dict[str, Any],
    shots: list[dict],
    policy: dict[str, Any],
    selected_frames: set[int],
    injection: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Compute only selected frames while still consuming the video stream in order."""

    if not selected_frames:
        return {}
    source_path = resolve_workspace_path(workspace, truth["video"]["source_file"])
    encoded_path = resolve_workspace_path(workspace, truth["video"]["encoded_file"])
    frame_count = int(truth["video"]["frame_count"])
    shots_by_frame = _shot_map(shots)
    dropped = {int(value) for value in injection.get("dropped_frames", [])}
    timestamp_jumps = dict(injection.get("timestamp_jumps", {}))
    timestamp_tolerance = float(injection.get("timestamp_tolerance_ms", 1.0))
    records: dict[int, dict[str, Any]] = {}
    max_selected = max(selected_frames)
    with Y4MReader(source_path) as source_reader, Y4MReader(encoded_path) as encoded_reader:
        for frame_index in range(frame_count):
            if frame_index > max_selected:
                break
            source_frame, source_retries, source_error = source_reader.read_frame(retries=2)
            encoded_frame, encoded_retries, encoded_error = encoded_reader.read_frame(retries=2)
            if frame_index not in selected_frames:
                if source_frame is None or encoded_frame is None:
                    break
                continue
            shot_id = int(shots_by_frame[frame_index])
            expected_ms = frame_index * 1000.0 / FPS
            observed_ms = expected_ms + _jump_for_frame(timestamp_jumps, frame_index)
            retries = max(source_retries, encoded_retries)
            if frame_index in dropped:
                records[frame_index] = missing_frame_metric(
                    frame_index,
                    shot_id,
                    observed_ms,
                    "E_DROPPED_FRAME",
                    retries,
                )
                continue
            if abs(observed_ms - expected_ms) > timestamp_tolerance:
                records[frame_index] = missing_frame_metric(
                    frame_index,
                    shot_id,
                    observed_ms,
                    "E_TIMESTAMP_JUMP",
                    retries,
                )
                continue
            if source_frame is None or encoded_frame is None:
                reason = source_error if source_frame is None else encoded_error
                records[frame_index] = missing_frame_metric(
                    frame_index,
                    shot_id,
                    observed_ms,
                    f"E_DECODE:{reason or 'unknown'}",
                    retries,
                )
                if source_frame is None or encoded_frame is None:
                    # Further stream boundaries are not trustworthy after a failed read.
                    break
                continue
            records[frame_index] = compute_frame_metric(
                source_frame,
                encoded_frame,
                frame_index,
                shot_id,
                observed_ms,
                policy,
                retries,
            )
    missing_reason = "E_DECODE:stream_ended_early"
    for frame_index in sorted(selected_frames - set(records)):
        records[frame_index] = missing_frame_metric(
            frame_index,
            int(shots_by_frame[frame_index]),
            frame_index * 1000.0 / FPS + _jump_for_frame(timestamp_jumps, frame_index),
            missing_reason,
            2,
        )
    return records


def yuv_to_rgb(frame: Y4MFrame) -> np.ndarray:
    y = frame.y.astype(np.float32)
    u = np.repeat(np.repeat(frame.u.astype(np.float32), 2, axis=0), 2, axis=1) - 128.0
    v = np.repeat(np.repeat(frame.v.astype(np.float32), 2, axis=0), 2, axis=1) - 128.0
    red = y + 1.402 * v
    green = y - 0.344136 * u - 0.714136 * v
    blue = y + 1.772 * u
    return np.clip(np.stack((red, green, blue), axis=2), 0, 255).astype(np.uint8)


def write_ppm(path: Path, frame: Y4MFrame) -> None:
    rgb = yuv_to_rgb(frame)
    header = f"P6\n{frame.y.shape[1]} {frame.y.shape[0]}\n255\n".encode("ascii")
    atomic_write_bytes(path, header + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())


def export_preview(workspace: Path, truth: dict[str, Any], frame_index: int, output: Path) -> dict[str, Any]:
    source_path = resolve_workspace_path(workspace, truth["video"]["source_file"])
    encoded_path = resolve_workspace_path(workspace, truth["video"]["encoded_file"])
    if not (0 <= frame_index < int(truth["video"]["frame_count"])):
        raise ValueError("frame index outside the fixture")
    with Y4MReader(source_path) as source_reader, Y4MReader(encoded_path) as encoded_reader:
        source_frame = encoded_frame = None
        for index in range(frame_index + 1):
            source_frame, _retries, error = source_reader.read_frame(retries=2)
            if error:
                raise ValueError(error)
            encoded_frame, _retries, error = encoded_reader.read_frame(retries=2)
            if error:
                raise ValueError(error)
            if index == frame_index:
                break
    assert source_frame is not None and encoded_frame is not None
    write_ppm(output, encoded_frame)
    return {
        "frame_index": frame_index,
        "output": str(output),
        "width": int(encoded_frame.y.shape[1]),
        "height": int(encoded_frame.y.shape[0]),
        "pixel_format": "rgb24_ppm_decoded_from_yuv420p",
        "source_y_sha256": hashlib.sha256(source_frame.y.tobytes()).hexdigest(),
        "decoded_y_sha256": hashlib.sha256(encoded_frame.y.tobytes()).hexdigest(),
    }

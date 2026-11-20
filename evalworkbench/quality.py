"""Deterministic PSNR/SSIM and segment aggregation."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .constants import METRIC_ENCODING_VERSION
from .errors import BusinessError
from .util import hash_object


def validate_shots(shots: Iterable[dict], frame_count: int) -> list[dict]:
    normalized: list[dict] = []
    expected_start = 0
    for index, raw in enumerate(shots):
        try:
            shot_id = int(raw["shot_id"])
            start = int(raw["start_frame"])
            end = int(raw["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BusinessError("E_SCHEMA", "shot record is malformed", {"index": index}) from exc
        if shot_id != index:
            raise BusinessError("E_SCHEMA", "shot ids must be contiguous", {"expected": index, "actual": shot_id})
        if start != expected_start or end < start:
            raise BusinessError(
                "E_SCHEMA",
                "shot ranges must be contiguous and ordered",
                {"shot_id": shot_id, "start": start, "expected_start": expected_start, "end": end},
            )
        record = dict(raw)
        record["shot_id"] = shot_id
        record["start_frame"] = start
        record["end_frame"] = end
        record.setdefault("label", f"shot-{shot_id}")
        normalized.append(record)
        expected_start = end + 1
    if not normalized or expected_start != frame_count:
        raise BusinessError(
            "E_SCHEMA",
            "shot ranges do not cover the declared frame count",
            {"covered_until": expected_start, "frame_count": frame_count},
        )
    return normalized


def policy_hash(policy: dict) -> str:
    return hash_object(policy)


def _psnr_from_mse(mse: float, ceiling: float) -> float:
    if mse <= 0.0:
        return float(ceiling)
    return min(float(ceiling), float(10.0 * np.log10((255.0 * 255.0) / mse)))


def _ssim_plane(reference: np.ndarray, decoded: np.ndarray, window: int, c1: float, c2: float) -> float:
    height, width = reference.shape
    if window <= 0 or height % window or width % window:
        raise BusinessError(
            "E_SCHEMA",
            "SSIM window must evenly divide the plane",
            {"window": window, "width": width, "height": height},
        )
    ref = reference.astype(np.float64)
    dec = decoded.astype(np.float64)
    ref_blocks = ref.reshape(height // window, window, width // window, window).transpose(0, 2, 1, 3)
    dec_blocks = dec.reshape(height // window, window, width // window, window).transpose(0, 2, 1, 3)
    mu_ref = ref_blocks.mean(axis=(2, 3))
    mu_dec = dec_blocks.mean(axis=(2, 3))
    var_ref = ref_blocks.var(axis=(2, 3))
    var_dec = dec_blocks.var(axis=(2, 3))
    covariance = (ref_blocks * dec_blocks).mean(axis=(2, 3)) - mu_ref * mu_dec
    numerator = (2.0 * mu_ref * mu_dec + c1) * (2.0 * covariance + c2)
    denominator = (mu_ref * mu_ref + mu_dec * mu_dec + c1) * (var_ref + var_dec + c2)
    return float(np.mean(numerator / denominator))


def compute_frame_metric(
    source_frame: Any,
    decoded_frame: Any,
    frame_index: int,
    shot_id: int,
    timestamp_ms: float,
    policy: dict,
    retries: int = 0,
) -> dict:
    y_range = policy["pixel_range"]
    if list(y_range) != [0, 255]:
        raise BusinessError("E_SCHEMA", "only 8-bit [0,255] range is supported", {"pixel_range": y_range})
    weight_y = float(policy.get("channel_weights", {}).get("y", 1.0))
    if weight_y <= 0.0:
        raise BusinessError("E_SCHEMA", "Y channel weight must be positive", {"channel_weights": policy.get("channel_weights")})
    source_y = source_frame.y.astype(np.float64)
    decoded_y = decoded_frame.y.astype(np.float64)
    difference = source_y - decoded_y
    mse = float(np.mean(difference * difference))
    psnr = _psnr_from_mse(mse, float(policy.get("psnr_ceiling_db", 99.0)))
    window = int(policy.get("ssim_window", 8))
    ssim = _ssim_plane(
        source_frame.y,
        decoded_frame.y,
        window,
        float(policy.get("ssim_c1", 6.5025)),
        float(policy.get("ssim_c2", 58.5225)),
    )
    chroma_values: list[float] = []
    for source_plane, decoded_plane in ((source_frame.u, decoded_frame.u), (source_frame.v, decoded_frame.v)):
        plane_difference = source_plane.astype(np.float64) - decoded_plane.astype(np.float64)
        chroma_values.append(_psnr_from_mse(float(np.mean(plane_difference * plane_difference)), float(policy.get("psnr_ceiling_db", 99.0))))
    return {
        "frame_index": int(frame_index),
        "shot_id": int(shot_id),
        "timestamp_ms": round(float(timestamp_ms), 6),
        "psnr_db": round(float(psnr), 9),
        "ssim": round(float(ssim), 9),
        "chroma_psnr_db": round(float(np.mean(chroma_values)), 9),
        "missing": False,
        "reason": None,
        "retries": int(retries),
        "metric_encoding_version": METRIC_ENCODING_VERSION,
    }


def missing_frame_metric(frame_index: int, shot_id: int, timestamp_ms: float, reason: str, retries: int) -> dict:
    return {
        "frame_index": int(frame_index),
        "shot_id": int(shot_id),
        "timestamp_ms": round(float(timestamp_ms), 6),
        "psnr_db": None,
        "ssim": None,
        "chroma_psnr_db": None,
        "missing": True,
        "reason": reason,
        "retries": int(retries),
        "metric_encoding_version": METRIC_ENCODING_VERSION,
    }


def record_content_hash(record: dict) -> str:
    content = {key: value for key, value in record.items() if key not in {"record_hash", "deps_hash"}}
    return hash_object(content)


def summarize_segments(
    frames: list[dict],
    shots: list[dict],
    policy: dict,
    shot_ids: set[int] | None = None,
) -> list[dict]:
    by_index = {int(frame["frame_index"]): frame for frame in frames}
    summaries: list[dict] = []
    for shot in shots:
        if shot_ids is not None and int(shot["shot_id"]) not in shot_ids:
            continue
        start = int(shot["start_frame"])
        end = int(shot["end_frame"])
        members = [by_index[index] for index in range(start, end + 1) if index in by_index]
        valid = [frame for frame in members if not frame.get("missing") and frame.get("psnr_db") is not None]
        missing_indices = [int(frame["frame_index"]) for frame in members if frame.get("missing") or frame.get("psnr_db") is None]
        psnr_values = [float(frame["psnr_db"]) for frame in valid]
        ssim_values = [float(frame["ssim"]) for frame in valid]
        summary = {
            "shot_id": int(shot["shot_id"]),
            "label": shot.get("label", f"shot-{shot['shot_id']}"),
            "start_frame": start,
            "end_frame": end,
            "frame_count": end - start + 1,
            "valid_frame_count": len(valid),
            "missing_frames": missing_indices,
            "complete": len(missing_indices) == 0,
            "psnr_db": round(float(np.mean(psnr_values)), 9) if psnr_values else None,
            "ssim": round(float(np.mean(ssim_values)), 9) if ssim_values else None,
            "policy_version": policy["policy_version"],
            "policy_hash": policy_hash(policy),
            "metric_encoding_version": METRIC_ENCODING_VERSION,
            "aggregation": dict(policy.get("aggregation", {})),
        }
        summary["record_hash"] = hash_object({key: value for key, value in summary.items() if key != "record_hash"})
        summaries.append(summary)
    return summaries


def metric_ledger_digest(frames: list[dict], segments: list[dict], policy: dict) -> str:
    return hash_object(
        {
            "policy": policy,
            "frames": sorted(frames, key=lambda item: int(item["frame_index"])),
            "segments": sorted(segments, key=lambda item: int(item["shot_id"])),
        }
    )

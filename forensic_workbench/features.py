"""Frame features, perceptual signatures, and normalized suspicion metrics."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import PipelineConfig
from .errors import frame_format, region_bounds
from .util import sha256_bytes, sha256_json, stable_round


TILE_ROWS = 4
TILE_COLS = 4
SIGNATURE_ROWS = 8
SIGNATURE_COLS = 8


@dataclass(frozen=True)
class Selection:
    mask_digest: str
    sampling_digest: str
    selection_digest: str
    sampled_pixels: int
    excluded_pixels: int
    mask_pixels: int
    flat_indices: np.ndarray
    signature_cells: np.ndarray
    tile_cells: np.ndarray
    tile_indices: tuple[np.ndarray, ...]
    tile_counts: tuple[int, ...]


class SelectionResolver:
    """Builds and caches the exact pixel selection for each frame scope."""

    def __init__(self, width: int, height: int, config: PipelineConfig):
        self.width = width
        self.height = height
        self.config = config
        self._cache: dict[tuple[tuple[int, ...], tuple[int, ...]], Selection] = {}

    def for_frame(self, frame_index: int) -> Selection:
        active_sampling = tuple(
            index
            for index, region in enumerate(self.config.sampling.regions)
            if region.applies_to(frame_index)
        )
        active_mask = tuple(
            index
            for index, region in enumerate(self.config.watermark_mask.regions)
            if region.applies_to(frame_index)
        )
        key = (active_sampling, active_mask)
        selection = self._cache.get(key)
        if selection is None:
            selection = self._build(frame_index, active_sampling, active_mask)
            self._cache[key] = selection
        return selection

    def _build(
        self,
        frame_index: int,
        active_sampling: tuple[int, ...],
        active_mask: tuple[int, ...],
    ) -> Selection:
        sampling_mask = np.zeros((self.height, self.width), dtype=np.bool_)
        for index in active_sampling:
            region = self.config.sampling.regions[index]
            sampling_mask[
                region.y : region.y + region.height, region.x : region.x + region.width
            ] = True
        exclusion_mask = np.zeros((self.height, self.width), dtype=np.bool_)
        for index in active_mask:
            region = self.config.watermark_mask.regions[index]
            exclusion_mask[
                region.y : region.y + region.height, region.x : region.x + region.width
            ] = True
        selected = sampling_mask & ~exclusion_mask
        sampled_pixels = int(selected.sum())
        if sampled_pixels == 0:
            raise region_bounds(
                "sampling region and watermark mask exclude every pixel",
                frame_index=frame_index,
                sampling_regions=[idx for idx in active_sampling],
                mask_regions=[idx for idx in active_mask],
            )
        ys, xs = np.nonzero(selected)
        flat_indices = ys.astype(np.int64) * self.width + xs.astype(np.int64)
        signature_cells = (
            np.minimum((ys * SIGNATURE_ROWS) // self.height, SIGNATURE_ROWS - 1)
            * SIGNATURE_COLS
            + np.minimum((xs * SIGNATURE_COLS) // self.width, SIGNATURE_COLS - 1)
        ).astype(np.int16)
        tile_cells = (
            np.minimum((ys * TILE_ROWS) // self.height, TILE_ROWS - 1) * TILE_COLS
            + np.minimum((xs * TILE_COLS) // self.width, TILE_COLS - 1)
        ).astype(np.int16)
        tile_indices: list[np.ndarray] = []
        tile_counts: list[int] = []
        for tile in range(TILE_ROWS * TILE_COLS):
            indices = flat_indices[tile_cells == tile]
            tile_indices.append(indices)
            tile_counts.append(int(indices.size))
        packed_mask = np.packbits(selected.reshape(-1)).tobytes()
        sampling_payload = [
            self.config.sampling.regions[index].to_dict(include_scope=False)
            for index in active_sampling
        ]
        mask_payload = [
            self.config.watermark_mask.regions[index].to_dict(include_scope=False)
            for index in active_mask
        ]
        selection_bitmap_digest = sha256_bytes(
            b"forensic-selection-bitmap-v1"
            + struct.pack("<II", self.width, self.height)
            + packed_mask
        )
        sampling_digest = sha256_json(sampling_payload)
        mask_digest = sha256_json(mask_payload)
        selection_digest = sha256_json(
            {
                "bitmap": selection_bitmap_digest,
                "sampling": sampling_digest,
                "mask": mask_digest,
            }
        )
        return Selection(
            mask_digest=mask_digest,
            sampling_digest=sampling_digest,
            selection_digest=selection_digest,
            sampled_pixels=sampled_pixels,
            excluded_pixels=int((sampling_mask & exclusion_mask).sum()),
            mask_pixels=int(exclusion_mask.sum()),
            flat_indices=flat_indices,
            signature_cells=signature_cells,
            tile_cells=tile_cells,
            tile_indices=tuple(tile_indices),
            tile_counts=tuple(tile_counts),
        )


def _signature(gray: np.ndarray, cells: np.ndarray) -> str:
    sums = np.bincount(cells, weights=gray, minlength=SIGNATURE_ROWS * SIGNATURE_COLS)
    counts = np.bincount(cells, minlength=SIGNATURE_ROWS * SIGNATURE_COLS)
    means = sums / np.maximum(counts, 1)
    overall = float(gray.mean()) if gray.size else 0.0
    bits = (means >= overall) & (counts > 0)
    return np.packbits(bits).tobytes().hex()


def compute_frame_payload(
    *,
    frame_index: int,
    frame: np.ndarray,
    selection: Selection,
    shot_boundaries: frozenset[int],
    gop_size: int,
    feature_version: str,
) -> dict[str, Any]:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise frame_format(
            "frame format is invalid for feature extraction",
            frame_index=frame_index,
            shape=list(frame.shape),
            dtype=str(frame.dtype),
        )
    flat = frame.reshape(-1, 3)
    pixels = flat[selection.flat_indices]
    pixel_bytes = pixels.tobytes(order="C")
    content_digest = hashlib.sha256()
    content_digest.update(b"forensic-frame-content-v1")
    content_digest.update(struct.pack("<III", frame.shape[1], frame.shape[0], pixels.shape[0]))
    content_digest.update(pixel_bytes)
    gray = (
        pixels[:, 0].astype(np.uint32) * 29
        + pixels[:, 1].astype(np.uint32) * 150
        + pixels[:, 2].astype(np.uint32) * 77
    ) >> 8
    signature = _signature(gray, selection.signature_cells)
    tile_hashes: list[str | None] = []
    for tile, indices in enumerate(selection.tile_indices):
        if indices.size == 0:
            tile_hashes.append(None)
            continue
        tile_digest = hashlib.sha256()
        tile_digest.update(b"forensic-tile-v1")
        tile_digest.update(struct.pack("<H", tile))
        tile_digest.update(flat[indices].tobytes(order="C"))
        tile_hashes.append(tile_digest.hexdigest())
    gop_index = frame_index // gop_size
    return {
        "frame_index": frame_index,
        "feature_version": feature_version,
        "content_hash": content_digest.hexdigest(),
        "keyframe_signature": signature,
        "tile_hashes": tile_hashes,
        "selection_digest": selection.selection_digest,
        "sampling_digest": selection.sampling_digest,
        "mask_digest": selection.mask_digest,
        "sampled_pixels": selection.sampled_pixels,
        "excluded_pixels": selection.excluded_pixels,
        "mask_pixels": selection.mask_pixels,
        "gop_index": gop_index,
        "gop_boundary": frame_index % gop_size == 0,
        "scene_boundary": frame_index in shot_boundaries,
    }


def signature_offset(left: str, right: str) -> tuple[int, float]:
    distance = (int(left, 16) ^ int(right, 16)).bit_count()
    return distance, stable_round(distance / 64.0)


def tile_offset(left: list[str | None], right: list[str | None]) -> tuple[int, float]:
    total = max(len(left), len(right))
    if total == 0:
        return 0, 0.0
    mismatches = 0
    for index in range(total):
        left_value = left[index] if index < len(left) else None
        right_value = right[index] if index < len(right) else None
        if left_value != right_value:
            mismatches += 1
    return mismatches, stable_round(mismatches / total)


def normalized_score(signature_distance: float, tile_mismatch: float) -> float:
    return stable_round(0.65 * signature_distance + 0.35 * tile_mismatch)


def build_gop_summary(
    *,
    gop_index: int,
    features: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    if not features:
        raise region_bounds("GOP summary has no frame features", gop_index=gop_index)
    ordered = sorted(features, key=lambda item: int(item["frame_index"]))
    keyframe = ordered[0]
    frame_scores: list[dict[str, Any]] = []
    scores: list[float] = []
    for feature in ordered:
        signature_distance, signature_norm = signature_offset(
            str(keyframe["keyframe_signature"]), str(feature["keyframe_signature"])
        )
        tile_mismatches, tile_norm = tile_offset(
            list(keyframe["tile_hashes"]), list(feature["tile_hashes"])
        )
        score = normalized_score(signature_norm, tile_norm)
        scores.append(score)
        frame_scores.append(
            {
                "frame": int(feature["frame_index"]),
                "signature_hamming": signature_distance,
                "signature_offset": signature_norm,
                "tile_mismatches": tile_mismatches,
                "tile_mismatch": tile_norm,
                "score": score,
                "above_threshold": score >= threshold,
            }
        )
    return {
        "gop_index": gop_index,
        "frame_start": int(ordered[0]["frame_index"]),
        "frame_end": int(ordered[-1]["frame_index"]) + 1,
        "frame_count": len(ordered),
        "keyframe_frame": int(keyframe["frame_index"]),
        "keyframe_signature": str(keyframe["keyframe_signature"]),
        "keyframe_tile_hashes": list(keyframe["tile_hashes"]),
        "threshold": threshold,
        "suspicion_mean": stable_round(float(np.mean(scores))),
        "suspicion_max": stable_round(float(np.max(scores))),
        "suspicious_frames": [
            item["frame"] for item in frame_scores if item["above_threshold"]
        ],
        "frame_scores": frame_scores,
        "normalization": "score=0.65*(signature_hamming/64)+0.35*(mismatched_tiles/16)",
    }


def build_cross_gop_summary(
    *,
    left_gop: int,
    right_gop: int,
    left: dict[str, Any],
    right: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    signature_distance, signature_norm = signature_offset(
        str(left["keyframe_signature"]), str(right["keyframe_signature"])
    )
    tile_mismatches, tile_norm = tile_offset(
        list(left["keyframe_tile_hashes"]), list(right["keyframe_tile_hashes"])
    )
    score = normalized_score(signature_norm, tile_norm)
    return {
        "comparison_item": f"gop:{left_gop:03d}|gop:{right_gop:03d}",
        "left_gop": left_gop,
        "right_gop": right_gop,
        "left_keyframe_frame": int(left["keyframe_frame"]),
        "right_keyframe_frame": int(right["keyframe_frame"]),
        "signature_hamming": signature_distance,
        "signature_offset": signature_norm,
        "tile_mismatches": tile_mismatches,
        "tile_mismatch": tile_norm,
        "score": score,
        "threshold": threshold,
        "above_threshold": score >= threshold,
        "normalization": "score=0.65*(signature_hamming/64)+0.35*(mismatched_tiles/16)",
    }


def build_conclusion(
    *,
    window_start: int,
    window_end: int,
    gop_summaries: list[dict[str, Any]],
    cross_summaries: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    if not gop_summaries:
        raise region_bounds(
            "analysis window intersects no GOP",
            start_frame=window_start,
            end_frame=window_end,
        )
    gop_max = max(float(item["suspicion_max"]) for item in gop_summaries)
    gop_mean = float(np.mean([float(item["suspicion_mean"]) for item in gop_summaries]))
    cross_max = max((float(item["score"]) for item in cross_summaries), default=0.0)
    cross_mean = float(np.mean([float(item["score"]) for item in cross_summaries])) if cross_summaries else 0.0
    overall = stable_round(0.70 * gop_max + 0.30 * cross_max)
    suspicious_gops = [
        int(item["gop_index"]) for item in gop_summaries if float(item["suspicion_max"]) >= threshold
    ]
    suspicious_cross_items = [
        str(item["comparison_item"]) for item in cross_summaries if bool(item["above_threshold"])
    ]
    return {
        "analysis_window": {"start_frame": window_start, "end_frame": window_end},
        "threshold": threshold,
        "gop_count": len(gop_summaries),
        "cross_gop_count": len(cross_summaries),
        "gop_suspicion_mean": stable_round(gop_mean),
        "gop_suspicion_max": stable_round(gop_max),
        "cross_gop_suspicion_mean": stable_round(cross_mean),
        "cross_gop_suspicion_max": stable_round(cross_max),
        "overall_suspicion": overall,
        "suspicious_gops": suspicious_gops,
        "suspicious_cross_items": suspicious_cross_items,
        "integrity_status": "review" if overall >= threshold else "consistent",
        "normalization": "overall=0.70*max_gop_score+0.30*max_cross_gop_score; range=[0,1]",
    }

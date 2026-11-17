"""Configuration loading and deterministic configuration fingerprints."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

# Canonical current-codebook class order (alphabetical; doubles as the
# tie-break reversed order because a descending name sort keeps it explicit).
CANONICAL_CLASSES = ("alarm", "music", "noise", "speech")

DEFAULT_CONFIG: dict[str, Any] = {
    "audio": {
        "sample_rate_hz": 16000,
        "frame_length_ms": 20,
        "frame_shift_ms": 10,
        "total_duration_ms": 4000,
    },
    "classes": {
        "current_codebook_version": "codebook_v2",
        "current_grid_version": "frame_grid_v1",
        "canonical_order": list(CANONICAL_CLASSES),
        "codebooks": {
            "codebook_v1": {
                "0": "alarm",
                "1": "music",
                "2": "noise",
                "3": "speech",
            },
            "codebook_v2": {
                "10": "speech",
                "20": "music",
                "30": "alarm",
                "40": "noise",
                "50": "dog_bark",
            },
            "codebook_v3": {
                "100": "alarm",
                "200": "music",
                "300": "noise",
                "400": "speech",
            },
        },
        "grids": {
            "frame_grid_v1": {"frame_length_ms": 20, "frame_shift_ms": 10},
            # Foreign grid used by one synthetic shard to exercise resampling.
            "frame_grid_v2": {"frame_length_ms": 20, "frame_shift_ms": 8},
        },
    },
    "thresholds": {
        "frame_posterior": 0.45,
        "boundary_mean_min": 0.35,
        "boundary_mean_diff_max": 0.18,
        "overlap_ratio": 0.30,
        "min_trajectory_ms": 60,
        "min_gap_ms": 40,
    },
    "boundary": {
        "neighborhood_frames": 4,
    },
    "smoothing": {
        "method": "linear_crossfade",
        "window_frames": 3,
    },
    "runtime": {
        "max_workers": 4,
    },
    "paths": {
        "manifest": "samples/sample_manifest.yaml",
        "truth": "samples/global_truth.yaml",
        "shard_dir": "samples/shards",
        "wav": "samples/aed_synthetic.wav",
        "snapshot_dir": "output/snapshot",
        "report": "output/diagnostic_report.json",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
    return result


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            override = yaml.safe_load(handle) or {}
        config = _deep_merge(config, override)
    if tuple(config["classes"]["canonical_order"]) != CANONICAL_CLASSES:
        raise ValueError("canonical class order must be alarm/music/noise/speech")
    return config


def config_fingerprint(config: dict[str, Any]) -> str:
    """SHA-256 over all processing-affecting configuration fields.

    Paths and runtime worker counts do not affect per-field results, so they
    are excluded from the fingerprint on purpose.
    """

    material = {
        "audio": config["audio"],
        "classes": {
            key: config["classes"][key]
            for key in (
                "current_codebook_version",
                "current_grid_version",
                "canonical_order",
                "codebooks",
                "grids",
            )
        },
        "thresholds": config["thresholds"],
        "boundary": config["boundary"],
        "smoothing": config["smoothing"],
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def frame_count(total_duration_ms: int, frame_shift_ms: int) -> int:
    return total_duration_ms // frame_shift_ms + 1


def frame_to_ms(frame_index: int, frame_shift_ms: int) -> int:
    return frame_index * frame_shift_ms

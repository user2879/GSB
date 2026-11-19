"""Content fingerprints and derived cache keys (requirement 4)."""
from __future__ import annotations

from pathlib import Path

from .models import FORMAT_V1, FORMAT_V2
from .util import content_hash, file_hash


def frame_content_fingerprint(image_path: str | Path) -> str:
    return file_hash(image_path)


def detection_params_fingerprint(params_dict: dict) -> str:
    return content_hash({"kind": "detection_params", "params": params_dict})


def match_config_fingerprint(config_dict: dict) -> str:
    return content_hash({"kind": "match_config", "config": config_dict})


def format_version_fingerprint(fmt: str) -> str:
    if fmt not in (FORMAT_V1, FORMAT_V2):
        raise ValueError(f"unknown format {fmt}")
    return content_hash({"kind": "format_version", "format": fmt})


def sidecar_fingerprint(sidecar: dict | None) -> str:
    if sidecar is None:
        return content_hash({"kind": "sidecar", "sidecar": None})
    return content_hash({"kind": "sidecar", "sidecar": sidecar})


def detect_cache_key(
    frame_fp: str,
    detect_fp: str,
    match_fp: str,
) -> str:
    # Frame candidates depend only on pixels, detection params and the match
    # config that governs pool eviction.  Format/sidecar enter at match and
    # serialization keys so a version switch never re-runs detection.
    return content_hash(
        {
            "stage": "detect",
            "frame_content": frame_fp,
            "detection_params": detect_fp,
            "match_config": match_fp,
        }
    )


def frame_candidate_fingerprint(frame_dict: dict) -> str:
    return content_hash({"kind": "frame_candidates", "frame": frame_dict})


def track_cache_key(
    sequence: str,
    frame_fps: list[str],
    detect_fp: str,
    match_fp: str,
    format_fp: str,
    sidecar_fp: str,
) -> str:
    return content_hash(
        {
            "stage": "track",
            "sequence": sequence,
            "frame_fingerprints": frame_fps,
            "detection_params": detect_fp,
            "match_config": match_fp,
            "format_version": format_fp,
            "sidecar": sidecar_fp,
        }
    )


def serialization_cache_key(
    track_fps: list[str],
    detect_fp: str,
    match_fp: str,
    format_fp: str,
    sidecar_fp: str,
) -> str:
    return content_hash(
        {
            "stage": "serialization",
            "track_fingerprints": sorted(track_fps),
            "detection_params": detect_fp,
            "match_config": match_fp,
            "format_version": format_fp,
            "sidecar": sidecar_fp,
        }
    )


def global_config_hash(
    dataset_root: str,
    detection_params: dict,
    match_config: dict,
    class_table: dict,
    width: int,
    height: int,
) -> str:
    return content_hash(
        {
            "kind": "global_config",
            "dataset_root": str(dataset_root),
            "detection_params": detection_params,
            "match_config": match_config,
            "class_table": class_table,
            "width": width,
            "height": height,
        }
    )

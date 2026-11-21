from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import E_CONFIG_INVALID, BusinessError
from .util import finite_number, read_json, stable_hash


@dataclass(frozen=True)
class DetectionParams:
    raw: dict[str, Any]
    class_map: dict[int, str]
    pool_capacity: int
    fingerprint: str

    def class_name(self, code: int) -> str | None:
        return self.class_map.get(code)


@dataclass(frozen=True)
class MatchConfig:
    raw: dict[str, Any]
    iou_threshold: float
    max_gap_frames: int
    pool_eviction_rule: str
    fixed_decimal_places: int
    concurrency: int
    snapshot_commit_interval: int
    fingerprint: str
    full_hash: str

    @property
    def determinism_fingerprint(self) -> str:
        payload = {
            "iou_threshold": self.iou_threshold,
            "max_gap_frames": self.max_gap_frames,
            "pool_eviction_rule": self.pool_eviction_rule,
            "fixed_decimal_places": self.fixed_decimal_places,
        }
        return stable_hash(payload)


@dataclass(frozen=True)
class ExchangeConfig:
    active_version: str
    raw: dict[str, Any]


def _load_json(path: Path, stage: str = "config") -> dict[str, Any]:
    try:
        value = read_json(path)
    except FileNotFoundError as exc:
        raise BusinessError(E_CONFIG_INVALID, f"missing file: {path}", stage) from exc
    except Exception as exc:
        raise BusinessError(E_CONFIG_INVALID, f"invalid JSON {path}: {exc}", stage) from exc
    if not isinstance(value, dict):
        raise BusinessError(E_CONFIG_INVALID, f"JSON root must be an object: {path}", stage)
    return value


def load_detection_params(data_dir: str | Path) -> DetectionParams:
    path = Path(data_dir) / "detection_params.json"
    raw = _load_json(path)
    class_map_raw = raw.get("class_map")
    if not isinstance(class_map_raw, dict) or not class_map_raw:
        raise BusinessError(E_CONFIG_INVALID, "class_map must be a non-empty object", "config")
    class_map: dict[int, str] = {}
    names: set[str] = set()
    for key, value in class_map_raw.items():
        try:
            code = int(key)
        except (TypeError, ValueError) as exc:
            raise BusinessError(E_CONFIG_INVALID, f"invalid class code: {key}", "config") from exc
        if not isinstance(value, str) or not value:
            raise BusinessError(E_CONFIG_INVALID, f"invalid class name for code {code}", "config")
        if code in class_map or value in names:
            raise BusinessError(E_CONFIG_INVALID, "duplicate class code/name", "config")
        class_map[code] = value
        names.add(value)
    pool_capacity = raw.get("candidate_pool_capacity", 0)
    if not isinstance(pool_capacity, int) or isinstance(pool_capacity, bool) or pool_capacity <= 0:
        raise BusinessError(E_CONFIG_INVALID, "candidate_pool_capacity must be positive", "config")
    weights = raw.get("fusion_weights", {})
    if not isinstance(weights, dict):
        raise BusinessError(E_CONFIG_INVALID, "fusion_weights must be an object", "config")
    for value in weights.values():
        if not finite_number(value):
            # Per-frame overrides are validated in the detection stage. A malformed
            # default is still rejected by the detector for every frame.
            continue
    fingerprint_payload = dict(raw)
    fingerprint_payload.pop("runtime", None)
    fingerprint = stable_hash(fingerprint_payload)
    return DetectionParams(
        raw=raw,
        class_map=class_map,
        pool_capacity=pool_capacity,
        fingerprint=fingerprint,
    )


def validate_runtime_weight(raw: dict[str, Any], series_name: str, frame_index: int) -> None:
    override = (
        raw.get("frame_overrides", {})
        .get(series_name, {})
        .get(str(frame_index), {})
    )
    weights = override.get("fusion_weights") if isinstance(override, dict) else None
    if weights is None:
        weights = raw.get("fusion_weights", {})
    if isinstance(weights, dict):
        for key, value in weights.items():
            if not finite_number(value):
                raise BusinessError(
                    "E_NON_FINITE",
                    f"non-finite fusion weight {key}={value!r}",
                    "detection",
                    series_name=series_name,
                    frame_index=frame_index,
                )


def load_match_config(data_dir: str | Path) -> MatchConfig:
    path = Path(data_dir) / "match_config.json"
    raw = _load_json(path)
    try:
        iou_threshold = float(raw["iou_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessError(E_CONFIG_INVALID, "invalid iou_threshold", "config") from exc
    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise BusinessError(E_CONFIG_INVALID, "iou_threshold must be in [0, 1]", "config")
    try:
        max_gap_frames = int(raw["max_gap_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessError(E_CONFIG_INVALID, "invalid max_gap_frames", "config") from exc
    if max_gap_frames < 0:
        raise BusinessError(E_CONFIG_INVALID, "max_gap_frames must be non-negative", "config")
    try:
        fixed_decimal_places = int(raw["fixed_decimal_places"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessError(E_CONFIG_INVALID, "invalid fixed_decimal_places", "config") from exc
    if not 0 <= fixed_decimal_places <= 6:
        raise BusinessError(E_CONFIG_INVALID, "fixed_decimal_places must be 0..6", "config")
    try:
        concurrency = int(raw["concurrency"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessError(E_CONFIG_INVALID, "invalid concurrency", "config") from exc
    if concurrency < 1:
        raise BusinessError(E_CONFIG_INVALID, "concurrency must be >= 1", "config")
    try:
        snapshot_commit_interval = int(raw["snapshot_commit_interval"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessError(E_CONFIG_INVALID, "invalid snapshot_commit_interval", "config") from exc
    if snapshot_commit_interval < 1:
        raise BusinessError(E_CONFIG_INVALID, "snapshot_commit_interval must be >= 1", "config")
    rule = str(raw.get("pool_eviction_rule", ""))
    if rule not in {"score_desc_then_coord_lex", "score_desc_coord_lex"}:
        raise BusinessError(E_CONFIG_INVALID, "unsupported pool_eviction_rule", "config")
    determinism_payload = {
        "iou_threshold": iou_threshold,
        "max_gap_frames": max_gap_frames,
        "pool_eviction_rule": rule,
        "fixed_decimal_places": fixed_decimal_places,
    }
    return MatchConfig(
        raw=raw,
        iou_threshold=iou_threshold,
        max_gap_frames=max_gap_frames,
        pool_eviction_rule=rule,
        fixed_decimal_places=fixed_decimal_places,
        concurrency=concurrency,
        snapshot_commit_interval=snapshot_commit_interval,
        fingerprint=stable_hash(determinism_payload),
        full_hash=stable_hash(raw),
    )


def load_exchange_config(data_dir: str | Path, override: str | None = None) -> ExchangeConfig:
    path = Path(data_dir) / "exchange_config.json"
    raw = _load_json(path) if path.exists() else {"active_version": "v1"}
    version = override or os.environ.get("CANDIDATE_EXCHANGE_VERSION") or raw.get("active_version", "v1")
    if version not in {"v1", "v2"}:
        raise BusinessError(
            E_CONFIG_INVALID, f"active_version must be v1 or v2, got {version!r}", "config"
        )
    return ExchangeConfig(active_version=version, raw=raw)

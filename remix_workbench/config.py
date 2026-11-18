from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .errors import MixConfigError
from .fingerprint import canonical_json, digest_json
from .models import DEFAULT_TRACKS, FORMAT_VERSIONS


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: str | Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("remix config must be a JSON object")
    return normalize_config(data)


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(raw)
    config.setdefault("sample_rate", 16000)
    config.setdefault("gain_quantization", 1024)
    config.setdefault("render_frame_length", 512)
    config.setdefault("thread_pool_size", 4)
    if "snapshot_interval_fragments" in config:
        config["snapshot_interval_clips"] = config["snapshot_interval_fragments"]
    else:
        config.setdefault("snapshot_interval_clips", 2)
        config["snapshot_interval_fragments"] = config["snapshot_interval_clips"]
    config.setdefault("defaults", {})
    config.setdefault("conversions", [])
    defaults = config["defaults"]
    if not isinstance(defaults, dict):
        raise ValueError("config.defaults must be an object")
    defaults.setdefault("start_ms", 0.0)
    defaults.setdefault("end_ms", None)
    defaults.setdefault("tracks", {})
    clips = config.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("config.clips must be a non-empty array")
    seen: set[str] = set()
    normalized_clips = []
    for clip in clips:
        if not isinstance(clip, dict) or not isinstance(clip.get("clip_id"), str):
            raise ValueError("each clip must have a string clip_id")
        clip = copy.deepcopy(clip)
        clip_id = clip["clip_id"]
        if clip_id in seen:
            raise ValueError(f"duplicate clip_id {clip_id}")
        seen.add(clip_id)
        clip.setdefault("source_format", "v1")
        clip.setdefault("start_ms", defaults["start_ms"])
        clip.setdefault("end_ms", defaults["end_ms"])
        tracks = clip.get("tracks", {})
        if not isinstance(tracks, dict):
            raise ValueError(f"{clip_id}.tracks must be an object")
        normalized_tracks: dict[str, Any] = {}
        for track in DEFAULT_TRACKS:
            base = copy.deepcopy(defaults["tracks"].get(track, {}))
            override = copy.deepcopy(tracks.get(track, {}))
            if not isinstance(base, dict) or not isinstance(override, dict):
                raise ValueError(f"{clip_id}.{track} track config must be an object")
            merged = {**base, **override}
            merged.setdefault("gain", 1.0)
            merged.setdefault("mute", False)
            merged.setdefault("start_ms", 0.0)
            merged.setdefault("end_ms", None)
            normalized_tracks[track] = merged
        for track, override in tracks.items():
            if track in normalized_tracks:
                continue
            if not isinstance(override, dict):
                raise ValueError(f"{clip_id}.{track} track config must be an object")
            merged = copy.deepcopy(override)
            merged.setdefault("gain", 1.0)
            merged.setdefault("mute", False)
            merged.setdefault("start_ms", 0.0)
            merged.setdefault("end_ms", None)
            normalized_tracks[track] = merged
        clip["tracks"] = normalized_tracks
        normalized_clips.append(clip)
    config["clips"] = normalized_clips
    overrides = config.setdefault("track_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("config.track_overrides must be an object")
    conversions = config.get("conversions", [])
    if not isinstance(conversions, list):
        raise ValueError("config.conversions must be an array")
    for index, item in enumerate(conversions):
        if not isinstance(item, dict):
            raise ValueError("conversion entries must be objects")
        item.setdefault("id", f"conversion_{index:03d}")
        item.setdefault("source_format", item.get("from", "v1"))
        item.setdefault("target_format", item.get("to", "v3"))
        if item["target_format"] not in FORMAT_VERSIONS:
            raise ValueError(f"invalid conversion target {item['target_format']}")
    return config


def config_fingerprint(config: dict[str, Any]) -> str:
    return digest_json(config)


def config_text(config: dict[str, Any]) -> str:
    return canonical_json(config)


def _finite_number(value: Any, label: str, *, track: str | None = None) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}:
            raise MixConfigError(f"{label} must be finite", track=track)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MixConfigError(f"{label} must be numeric", track=track)
    number = float(value)
    if not math.isfinite(number):
        raise MixConfigError(f"{label} must be finite", track=track)
    return number


def validate_global(config: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    try:
        sample_rate = config["sample_rate"]
        if (
            not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or sample_rate <= 0
        ):
            raise MixConfigError("sample_rate must be a positive integer")
        quantization = config["gain_quantization"]
        if (
            not isinstance(quantization, int)
            or isinstance(quantization, bool)
            or quantization <= 0
            or quantization > 16_777_216
        ):
            raise MixConfigError(
                "gain_quantization must be an integer in [1, 16777216]"
            )
        frame_length = config["render_frame_length"]
        if (
            not isinstance(frame_length, int)
            or isinstance(frame_length, bool)
            or frame_length <= 0
        ):
            raise MixConfigError("render_frame_length must be a positive integer")
        workers = config["thread_pool_size"]
        if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
            raise MixConfigError("thread_pool_size must be a positive integer")
        interval = config["snapshot_interval_clips"]
        if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
            raise MixConfigError(
                "snapshot_interval_fragments/snapshot_interval_clips must be a positive integer"
            )
    except (KeyError, MixConfigError) as exc:
        track = getattr(exc, "track", None)
        errors.append(
            {
                "clip_id": "<global>",
                "track": track,
                "stage": "config",
                "code": "E_MIX_CONFIG",
                "message": str(exc),
            }
        )
    return errors


def validate_track_settings(
    clip: dict[str, Any],
    track: str,
    settings: dict[str, Any],
) -> tuple[int, bool, float, float | None]:
    gain = _finite_number(settings.get("gain"), f"{clip['clip_id']}.{track}.gain", track=track)
    if gain < 0:
        raise MixConfigError(
            f"{clip['clip_id']}.{track}.gain must be non-negative", track=track
        )
    mute = settings.get("mute")
    if not isinstance(mute, bool):
        raise MixConfigError(
            f"{clip['clip_id']}.{track}.mute must be boolean", track=track
        )
    start_ms = _finite_number(
        settings.get("start_ms", 0),
        f"{clip['clip_id']}.{track}.start_ms",
        track=track,
    )
    if start_ms < 0:
        raise MixConfigError(
            f"{clip['clip_id']}.{track}.start_ms must be non-negative", track=track
        )
    end_ms_raw = settings.get("end_ms")
    end_ms = (
        None
        if end_ms_raw is None
        else _finite_number(
            end_ms_raw, f"{clip['clip_id']}.{track}.end_ms", track=track
        )
    )
    if end_ms is not None and end_ms <= start_ms:
        raise MixConfigError(
            f"{clip['clip_id']}.{track} trim end must be after start", track=track
        )
    quantization = int(clip["_gain_quantization"])
    gain_q = int(math.floor(gain * quantization + 0.5))
    return gain_q, mute, start_ms, end_ms


def ms_to_samples(milliseconds: float, sample_rate: int) -> int:
    return int(math.floor(float(milliseconds) * sample_rate / 1000.0 + 0.5))


def clip_interval(
    clip: dict[str, Any], frames: int, sample_rate: int
) -> tuple[int, int]:
    start_ms = _finite_number(clip.get("start_ms", 0), f"{clip['clip_id']}.start_ms")
    if start_ms < 0:
        raise MixConfigError(f"{clip['clip_id']}.start_ms must be non-negative")
    end_raw = clip.get("end_ms")
    end_ms = (
        None
        if end_raw is None
        else _finite_number(end_raw, f"{clip['clip_id']}.end_ms")
    )
    start = ms_to_samples(start_ms, sample_rate)
    end = frames if end_ms is None else ms_to_samples(end_ms, sample_rate)
    if start < 0 or start >= end or start >= frames:
        raise MixConfigError(
            f"{clip['clip_id']} trim [{start_ms}, {end_ms}] is outside 0..{frames} frames"
        )
    if end > frames:
        raise MixConfigError(
            f"{clip['clip_id']} trim end {end} exceeds {frames} source frames"
        )
    return start, end


def track_interval(
    clip_id: str,
    track: str,
    settings: dict[str, Any],
    frames: int,
    sample_rate: int,
) -> tuple[int, int]:
    start_ms = float(settings["start_ms"])
    end_raw = settings["end_ms"]
    end_ms = None if end_raw is None else float(end_raw)
    start = ms_to_samples(start_ms, sample_rate)
    end = frames if end_ms is None else ms_to_samples(end_ms, sample_rate)
    if start < 0 or start >= end or start >= frames:
        raise MixConfigError(
            f"{clip_id}.{track} trim [{start_ms}, {end_ms}] is outside the clip",
            track=track,
        )
    if end > frames:
        raise MixConfigError(
            f"{clip_id}.{track} trim end {end} exceeds {frames} source frames",
            track=track,
        )
    return start, end


def write_config(path: str | Path, config: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

"""JSON manifest parsing and strict validation.

Validation failures raise :class:`HapticError` carrying an ``E_xxx`` code.
Validation always runs before the state directory is touched, so an illegal
input never produces any state file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .errors import (
    E_CURVE,
    E_DEVICE,
    E_EVENT,
    E_FORMAT,
    E_GLOBAL,
    E_JSON,
    HapticError,
)
from .models import CurvePoint, Device, Event, GlobalConfig, Manifest, SyncAnchor

EVENT_TYPES = frozenset({"envelope", "texture", "tap"})


def load_manifest(path: str | Path) -> Manifest:
    file_path = Path(path)
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HapticError(E_JSON, f"cannot read manifest: {exc}", path=str(file_path)) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HapticError(
            E_JSON,
            "manifest is not valid JSON",
            path=str(file_path),
            line=exc.lineno,
            column=exc.colno,
        ) from exc
    return parse_manifest(data)


def parse_manifest(data: Any) -> Manifest:
    if not isinstance(data, dict):
        raise HapticError(E_FORMAT, "manifest root must be a JSON object")

    devices = _parse_devices(data.get("devices"))
    device_ids = {device.device_id for device in devices}
    events = _parse_events(data.get("events", []), devices)
    anchors = _parse_anchors(data.get("sync_anchors", []), device_ids)
    config = _parse_global(data.get("global", {}), devices)

    state_dir = _require_nonempty_string(data, "state_dir", E_FORMAT)
    output_file = _require_nonempty_string(data, "output_file", E_FORMAT)

    known = {"devices", "events", "sync_anchors", "global", "state_dir", "output_file"}
    extra = {key: value for key, value in data.items() if key not in known}
    return Manifest(
        devices=tuple(devices),
        events=tuple(events),
        sync_anchors=tuple(anchors),
        global_config=config,
        state_dir=state_dir,
        output_file=output_file,
        extra=extra,
    )


def _parse_devices(raw: Any) -> list[Device]:
    if not isinstance(raw, list) or not raw:
        raise HapticError(E_DEVICE, "devices must be a non-empty array")
    devices: list[Device] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise HapticError(E_DEVICE, "device entry must be an object", index=index)
        device_id = _require_nonempty_string(item, "device_id", E_DEVICE, index=index)
        if device_id in seen:
            raise HapticError(E_DEVICE, "duplicate device id", device_id=device_id)
        seen.add(device_id)

        sample_rate = _require_positive_int(item, "sample_rate_hz", E_DEVICE, index)
        frame_duration = _require_positive_int(item, "frame_duration_ms", E_DEVICE, index)
        max_commands = _require_positive_int(item, "max_commands_per_frame", E_DEVICE, index)
        levels = _require_int(item, "levels", E_DEVICE, index)
        if levels < 2:
            raise HapticError(E_DEVICE, "levels must be >= 2", device_id=device_id, levels=levels)

        min_intensity = _require_finite_number(item, "min_intensity", E_DEVICE, device_id)
        max_intensity = _require_finite_number(item, "max_intensity", E_DEVICE, device_id)
        if not min_intensity < max_intensity:
            raise HapticError(
                E_DEVICE,
                "min_intensity must be strictly less than max_intensity",
                device_id=device_id,
            )
        devices.append(
            Device(
                device_id=device_id,
                sample_rate_hz=sample_rate,
                frame_duration_ms=frame_duration,
                min_intensity=float(min_intensity),
                max_intensity=float(max_intensity),
                levels=levels,
                max_commands_per_frame=max_commands,
            )
        )
    return devices


def _parse_events(raw: Any, devices: list[Device]) -> list[Event]:
    if not isinstance(raw, list):
        raise HapticError(E_EVENT, "events must be an array")
    by_id = {device.device_id: device for device in devices}
    events: list[Event] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise HapticError(E_EVENT, "event entry must be an object", index=index)
        event_id = _require_nonempty_string(item, "event_id", E_EVENT, index=index)
        if event_id in seen:
            raise HapticError(E_EVENT, "duplicate event id", event_id=event_id)
        seen.add(event_id)

        event_type = _require_nonempty_string(item, "event_type", E_EVENT, index=index)
        if event_type not in EVENT_TYPES:
            raise HapticError(
                E_EVENT,
                "unsupported event_type",
                event_id=event_id,
                event_type=event_type,
                allowed=sorted(EVENT_TYPES),
            )
        start_ms = _require_int(item, "start_ms", E_EVENT, index)
        if start_ms < 0:
            raise HapticError(E_EVENT, "start_ms must be non-negative", event_id=event_id)
        duration_ms = _require_int(item, "duration_ms", E_EVENT, index)
        if duration_ms <= 0:
            raise HapticError(E_EVENT, "duration_ms must be positive", event_id=event_id)

        target = _require_nonempty_string(item, "target_device", E_EVENT, index=index)
        if target not in by_id:
            raise HapticError(E_EVENT, "unknown target_device", event_id=event_id, target_device=target)

        curve = _parse_curve(item.get("curve"), by_id[target], event_id, start_ms, duration_ms)
        events.append(
            Event(
                event_id=event_id,
                event_type=event_type,
                start_ms=start_ms,
                duration_ms=duration_ms,
                target_device=target,
                curve=curve,
            )
        )
    return events


def _parse_curve(
    raw: Any,
    device: Device,
    event_id: str,
    start_ms: int,
    duration_ms: int,
) -> tuple[CurvePoint, ...]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise HapticError(E_CURVE, "curve must contain at least two points", event_id=event_id)
    points: list[CurvePoint] = []
    previous_time: int | None = None
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise HapticError(E_CURVE, "curve point must be an object", event_id=event_id, index=index)
        time_ms = _require_int(item, "time_ms", E_CURVE, index)
        if previous_time is not None and time_ms <= previous_time:
            raise HapticError(
                E_CURVE,
                "curve time_ms values must be strictly increasing",
                event_id=event_id,
                index=index,
            )
        previous_time = time_ms
        intensity = _require_finite_number(item, "intensity", E_CURVE, event_id)
        if not device.min_intensity <= intensity <= device.max_intensity:
            raise HapticError(
                E_CURVE,
                "curve intensity outside device range",
                event_id=event_id,
                index=index,
                intensity=intensity,
                range=[device.min_intensity, device.max_intensity],
            )
        points.append(CurvePoint(time_ms=time_ms, intensity=float(intensity)))

    end_ms = start_ms + duration_ms
    if points[0].time_ms > start_ms or points[-1].time_ms < end_ms:
        raise HapticError(
            E_CURVE,
            "curve must cover the full event interval [start_ms, start_ms+duration_ms]",
            event_id=event_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    return tuple(points)


def _parse_anchors(raw: Any, device_ids: set[str]) -> list[SyncAnchor]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise HapticError(E_FORMAT, "sync_anchors must be an array")
    anchors: list[SyncAnchor] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise HapticError(E_FORMAT, "sync anchor must be an object", index=index)
        anchor_id = _require_nonempty_string(item, "anchor_id", E_FORMAT, index=index)
        if anchor_id in seen:
            raise HapticError(E_FORMAT, "duplicate sync anchor id", anchor_id=anchor_id)
        seen.add(anchor_id)
        time_ms = _require_int(item, "time_ms", E_FORMAT, index)
        if time_ms < 0:
            raise HapticError(E_FORMAT, "anchor time_ms must be non-negative", anchor_id=anchor_id)
        anchor_devices = item.get("device_ids")
        if not isinstance(anchor_devices, list) or not anchor_devices:
            raise HapticError(E_FORMAT, "anchor device_ids must be a non-empty array", anchor_id=anchor_id)
        anchor_device_ids: list[str] = []
        for device_id in anchor_devices:
            if not isinstance(device_id, str) or not device_id.strip():
                raise HapticError(E_FORMAT, "anchor device id must be a non-empty string", anchor_id=anchor_id)
            if device_id not in device_ids:
                raise HapticError(
                    E_FORMAT,
                    "anchor references unknown device",
                    anchor_id=anchor_id,
                    device_id=device_id,
                )
            if device_id not in anchor_device_ids:
                anchor_device_ids.append(device_id)
        reference = _require_nonempty_string(item, "reference_device", E_FORMAT, index=index)
        if reference not in anchor_device_ids:
            raise HapticError(
                E_FORMAT,
                "reference_device must be one of anchor device_ids",
                anchor_id=anchor_id,
                reference_device=reference,
            )
        allowed = _require_int(item, "allowed_samples", E_FORMAT, index)
        if allowed < 0:
            raise HapticError(E_FORMAT, "allowed_samples must be >= 0", anchor_id=anchor_id)
        anchors.append(
            SyncAnchor(
                anchor_id=anchor_id,
                time_ms=time_ms,
                device_ids=tuple(anchor_device_ids),
                reference_device=reference,
                allowed_samples=allowed,
            )
        )
    return anchors


def _parse_global(raw: Any, devices: list[Device]) -> GlobalConfig:
    if not isinstance(raw, dict):
        raise HapticError(E_GLOBAL, "global must be an object")
    concurrency = _require_positive_int(raw, "concurrency", E_GLOBAL)
    inflight_limit = _require_positive_int(raw, "inflight_limit", E_GLOBAL)
    max_frame_budget = max(device.max_commands_per_frame for device in devices)
    if inflight_limit < max_frame_budget:
        raise HapticError(
            E_GLOBAL,
            "inflight_limit must be >= the largest max_commands_per_frame so progress is possible",
            inflight_limit=inflight_limit,
            required=max_frame_budget,
        )
    return GlobalConfig(concurrency=concurrency, inflight_limit=inflight_limit)


def _require_int(container: dict, key: str, code: str, index: int | None = None) -> int:
    value = container.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise HapticError(code, f"{key} must be an integer", key=key, index=index)
    return value


def _require_positive_int(container: dict, key: str, code: str, index: int | None = None) -> int:
    value = _require_int(container, key, code, index)
    if value <= 0:
        raise HapticError(code, f"{key} must be a positive integer", key=key, index=index)
    return value


def _require_nonempty_string(
    container: dict,
    key: str,
    code: str,
    index: int | None = None,
) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HapticError(code, f"{key} must be a non-empty string", key=key, index=index)
    return value


def _require_finite_number(
    container: dict,
    key: str,
    code: str,
    event_id: str | None = None,
) -> float:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise HapticError(
            code,
            f"{key} must be a finite number",
            key=key,
            event_id=event_id,
        )
    return float(value)

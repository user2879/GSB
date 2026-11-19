from __future__ import annotations

import base64
import binascii
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .errors import E_MASK_FORMAT_INVALID, Failure
from .io_utils import read_json
from .models import (
    InstanceMask,
    PaletteEntry,
    ParseBundle,
    instance_cache_key,
    instance_key,
    metadata_fingerprint,
)
from .rle import run_content_fingerprint, runs_to_mask, validate_runs
from .scheduler import OrderedTask, ordered_parallel_map


V1_MAGIC = b"PANO1\x00\x00\x00"
V1_VERSION = 1
V1_HEADER = struct.Struct("<8sHHIII")
V1_RECORD = struct.Struct("<III")
V1_RUN = struct.Struct("<II")
V2_FORMAT_VERSION = "v2"
FALLBACK_PALETTE: PaletteEntry = (128, 128, 128, 255)


@dataclass(frozen=True)
class V1RawRecord:
    instance_id: int
    class_id: int
    runs: tuple[tuple[int, int], ...]


@dataclass
class MetadataDocument:
    by_key: dict[str, dict[str, Any]]
    failures: list[Failure]

    def lookup(self, source_format: str, instance_id: int) -> dict[str, Any]:
        return self.by_key.get(
            instance_key(source_format, instance_id),
            self.by_key.get(str(instance_id), {}),
        )


def _failure(instance: str, stage: str, message: str) -> Failure:
    return Failure(instance, E_MASK_FORMAT_INVALID, stage, message)


def _valid_palette(value: Any) -> PaletteEntry:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("palette must contain four RGBA channels")
    channels = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError("palette channel is outside [0, 255]")
    return channels


def load_metadata_document(path: str | Path) -> MetadataDocument:
    failures: list[Failure] = []
    by_key: dict[str, dict[str, Any]] = {}
    try:
        raw = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return MetadataDocument(
            by_key={},
            failures=[_failure("metadata:package", "metadata", str(exc))],
        )
    entries = raw.get("instances", raw) if isinstance(raw, dict) else raw
    if isinstance(entries, dict):
        iterable = entries.items()
    elif isinstance(entries, list):
        iterable = ((str(item.get("instance_id", "")), item) for item in entries)
    else:
        return MetadataDocument(
            by_key={},
            failures=[_failure("metadata:package", "metadata", "instances must be a list or object")],
        )
    for key, entry in iterable:
        if not isinstance(entry, dict):
            failures.append(
                _failure(f"metadata:{key}", "metadata", "metadata entry must be an object")
            )
            continue
        parsed = dict(entry)
        source = str(parsed.get("source_format", parsed.get("source", "v1")))
        raw_id = parsed.get("instance_id", key)
        try:
            parsed["instance_id"] = int(raw_id)
        except (TypeError, ValueError):
            failures.append(
                _failure(f"metadata:{key}", "metadata", "invalid instance_id")
            )
            continue
        if source not in {"v1", "v2"}:
            failures.append(
                _failure(
                    instance_key(source, parsed["instance_id"]),
                    "metadata",
                    "source_format must be v1 or v2",
                )
            )
            continue
        parsed["source_format"] = source
        by_key[instance_key(source, parsed["instance_id"])] = parsed
        by_key.setdefault(str(parsed["instance_id"]), parsed)
    return MetadataDocument(by_key=by_key, failures=failures)


def _build_instance(
    *,
    source_format: str,
    instance_id: int,
    class_id: int,
    mask: np.ndarray,
    confidence: float,
    crowd: bool,
    palette: PaletteEntry,
    width: int,
    height: int,
    defaulted_fields: tuple[str, ...],
) -> InstanceMask:
    from .rle import mask_to_runs

    runs = mask_to_runs(mask)
    run_fp = run_content_fingerprint(runs, width=width, height=height)
    meta_fp = metadata_fingerprint(
        class_id=class_id,
        confidence=confidence,
        crowd=crowd,
        palette=palette,
    )
    cache_key = instance_cache_key(
        source_format=source_format,
        run_fingerprint=run_fp,
        class_id=class_id,
        metadata_fp=meta_fp,
    )
    return InstanceMask(
        source_format=source_format,
        instance_id=instance_id,
        class_id=class_id,
        mask=mask,
        confidence=confidence,
        crowd=crowd,
        palette=palette,
        run_fingerprint=run_fp,
        metadata_fingerprint=meta_fp,
        cache_key=cache_key,
        defaulted_fields=defaulted_fields,
        width=width,
        height=height,
    )


def _metadata_values(
    metadata: dict[str, Any],
    *,
    config: PipelineConfig,
    class_id: int,
    require_class_match: bool,
) -> tuple[float, bool, PaletteEntry, tuple[str, ...], str | None]:
    defaulted: list[str] = []
    if require_class_match and metadata.get("class_id", metadata.get("class")) is not None:
        try:
            metadata_class = int(metadata.get("class_id", metadata.get("class")))
        except (TypeError, ValueError):
            return 0.0, False, FALLBACK_PALETTE, (), "class_id is not an integer"
        if metadata_class != class_id:
            return 0.0, False, FALLBACK_PALETTE, (), "class_id disagrees with package"
    if metadata.get("confidence") is None:
        confidence = config.confidence_default
        defaulted.append("confidence")
    else:
        try:
            confidence = float(metadata["confidence"])
        except (TypeError, ValueError):
            return 0.0, False, FALLBACK_PALETTE, (), "confidence is not numeric"
    if not 0.0 <= confidence <= 1.0:
        return 0.0, False, FALLBACK_PALETTE, (), "confidence is outside [0, 1]"
    if metadata.get("crowd") is None:
        crowd = config.default_crowd
        defaulted.append("crowd")
    else:
        if type(metadata["crowd"]) is not bool:
            return 0.0, False, FALLBACK_PALETTE, (), "crowd must be boolean"
        crowd = bool(metadata["crowd"])
    if metadata.get("palette") is not None:
        try:
            palette = _valid_palette(metadata["palette"])
        except (TypeError, ValueError) as exc:
            return 0.0, False, FALLBACK_PALETTE, (), str(exc)
    elif class_id in config.palette:
        palette = config.palette[class_id]
        defaulted.append("palette")
    else:
        palette = FALLBACK_PALETTE
        defaulted.append("palette")
    return confidence, crowd, palette, tuple(defaulted), None


def parse_v1_package(
    path: str | Path,
    *,
    config: PipelineConfig,
    metadata: MetadataDocument | None = None,
    max_workers: int | None = None,
) -> ParseBundle:
    workers = config.max_workers if max_workers is None else max_workers
    metadata = metadata or MetadataDocument({}, [])
    failures: list[Failure] = list(metadata.failures)
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        return ParseBundle(
            "v1",
            0,
            0,
            failures=sorted(
                failures + [_failure("v1:package", "decode_v1", str(exc))],
                key=lambda item: item.stable_sort_key,
            ),
        )
    if len(payload) < V1_HEADER.size:
        return ParseBundle(
            "v1",
            0,
            0,
            failures=sorted(
                failures
                + [_failure("v1:package", "decode_v1", "truncated v1 header")],
                key=lambda item: item.stable_sort_key,
            ),
        )
    magic, version, flags, width, height, count = V1_HEADER.unpack_from(payload, 0)
    if magic != V1_MAGIC:
        failures.append(_failure("v1:package", "decode_v1", "invalid v1 magic"))
        return ParseBundle("v1", 0, 0, failures=sorted(failures, key=lambda item: item.stable_sort_key))
    if version != V1_VERSION or flags != 0:
        failures.append(_failure("v1:package", "decode_v1", "unsupported v1 version or flags"))
        return ParseBundle("v1", width, height, failures=sorted(failures, key=lambda item: item.stable_sort_key))
    if width <= 0 or height <= 0:
        failures.append(_failure("v1:package", "decode_v1", "invalid image dimensions"))
        return ParseBundle("v1", width, height, failures=sorted(failures, key=lambda item: item.stable_sort_key))
    pixel_count = width * height
    offset = V1_HEADER.size
    raw_records: list[V1RawRecord] = []
    seen_ids: set[int] = set()
    for record_index in range(count):
        if len(payload) - offset < V1_RECORD.size:
            failures.append(
                _failure(
                    f"v1:record-{record_index}",
                    "decode_v1",
                    "truncated v1 instance record",
                )
            )
            break
        instance_id, class_id, run_count = V1_RECORD.unpack_from(payload, offset)
        offset += V1_RECORD.size
        key = instance_key("v1", instance_id)
        if instance_id <= 0 or class_id <= 0:
            failures.append(_failure(key, "decode_v1", "instance ID and class ID must be positive"))
            break
        if instance_id in seen_ids:
            failures.append(_failure(key, "decode_v1", "duplicate instance ID"))
            break
        seen_ids.add(instance_id)
        if run_count > pixel_count + 1:
            failures.append(_failure(key, "decode_v1", "run count exceeds pixel count"))
            break
        required = run_count * V1_RUN.size
        if len(payload) - offset < required:
            failures.append(_failure(key, "decode_v1", "truncated v1 run data"))
            break
        runs = tuple(
            V1_RUN.unpack_from(payload, offset + index * V1_RUN.size)
            for index in range(run_count)
        )
        offset += required
        raw_records.append(V1RawRecord(instance_id, class_id, runs))

    def decode_record(record: V1RawRecord) -> tuple[InstanceMask | None, Failure | None]:
        key = instance_key("v1", record.instance_id)
        try:
            validated_runs = validate_runs(record.runs, pixel_count)
            mask = runs_to_mask(validated_runs, pixel_count).reshape(height, width)
        except (ValueError, OverflowError) as exc:
            return None, _failure(key, "decode_v1", f"invalid run list: {exc}")
        item_metadata = metadata.lookup("v1", record.instance_id)
        confidence, crowd, palette, defaulted, error = _metadata_values(
            item_metadata,
            config=config,
            class_id=record.class_id,
            require_class_match=True,
        )
        if error:
            return None, _failure(key, "decode_v1", error)
        instance = _build_instance(
            source_format="v1",
            instance_id=record.instance_id,
            class_id=record.class_id,
            mask=mask,
            confidence=confidence,
            crowd=crowd,
            palette=palette,
            width=width,
            height=height,
            defaulted_fields=defaulted,
        )
        return instance, None

    decoded = ordered_parallel_map(
        [
            OrderedTask(
                instance_id=record.instance_id,
                stage="decode_v1",
                payload=record,
                source_format="v1",
            )
            for record in raw_records
        ],
        decode_record,
        workers,
    )
    instances: list[InstanceMask] = []
    for instance, failure in decoded:
        if instance is not None:
            instances.append(instance)
        if failure is not None:
            failures.append(failure)
    return ParseBundle(
        "v1",
        width,
        height,
        instances=sorted(
            instances,
            key=lambda item: (item.source_format, item.instance_id, item.key),
        ),
        failures=sorted(failures, key=lambda item: item.stable_sort_key),
    )


def _decode_v2_instance(
    entry: Any,
    *,
    width: int,
    height: int,
    config: PipelineConfig,
) -> tuple[InstanceMask | None, Failure | None]:
    if not isinstance(entry, dict):
        return None, _failure("v2:unknown", "decode_v2", "instance entry must be an object")
    raw_id = entry.get("instance_id")
    key = f"v2:{raw_id}" if raw_id is not None else "v2:unknown"
    required = {"instance_id", "class_id", "confidence", "crowd", "palette", "mask_base64"}
    missing = sorted(required.difference(entry))
    if missing:
        return None, _failure(key, "decode_v2", f"missing fields: {', '.join(missing)}")
    try:
        instance_id = int(entry["instance_id"])
        class_id = int(entry["class_id"])
        confidence = float(entry["confidence"])
    except (TypeError, ValueError):
        return None, _failure(key, "decode_v2", "instance_id, class_id or confidence is invalid")
    key = instance_key("v2", instance_id)
    if instance_id <= 0 or class_id <= 0:
        return None, _failure(key, "decode_v2", "instance ID and class ID must be positive")
    if not 0.0 <= confidence <= 1.0:
        return None, _failure(key, "decode_v2", "confidence is outside [0, 1]")
    if type(entry["crowd"]) is not bool:
        return None, _failure(key, "decode_v2", "crowd must be boolean")
    try:
        palette = _valid_palette(entry["palette"])
    except (TypeError, ValueError) as exc:
        return None, _failure(key, "decode_v2", str(exc))
    encoded_mask = entry["mask_base64"]
    if not isinstance(encoded_mask, str):
        return None, _failure(key, "decode_v2", "mask_base64 must be a string")
    try:
        mask_bytes = base64.b64decode(encoded_mask, validate=True)
    except (binascii.Error, ValueError) as exc:
        return None, _failure(key, "decode_v2", f"invalid base64 mask: {exc}")
    expected = width * height
    if len(mask_bytes) != expected:
        return None, _failure(
            key,
            "decode_v2",
            f"decoded mask has {len(mask_bytes)} pixels, expected {expected}",
        )
    mask_flat = np.frombuffer(mask_bytes, dtype=np.uint8)
    if np.any(mask_flat > 1):
        return None, _failure(key, "decode_v2", "mask bytes must be 0 or 1")
    mask = mask_flat.astype(bool).reshape(height, width)
    return (
        _build_instance(
            source_format="v2",
            instance_id=instance_id,
            class_id=class_id,
            mask=mask,
            confidence=confidence,
            crowd=bool(entry["crowd"]),
            palette=palette,
            width=width,
            height=height,
            defaulted_fields=(),
        ),
        None,
    )


def parse_v2_package(
    path: str | Path,
    *,
    config: PipelineConfig,
    max_workers: int | None = None,
) -> ParseBundle:
    workers = config.max_workers if max_workers is None else max_workers
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return ParseBundle(
            "v2",
            0,
            0,
            failures=[_failure("v2:package", "decode_v2", str(exc))],
        )
    if not isinstance(payload, dict):
        return ParseBundle(
            "v2",
            0,
            0,
            failures=[_failure("v2:package", "decode_v2", "package root must be an object")],
        )
    try:
        width = int(payload["width"])
        height = int(payload["height"])
    except (KeyError, TypeError, ValueError):
        return ParseBundle(
            "v2",
            0,
            0,
            failures=[_failure("v2:package", "decode_v2", "missing or invalid width/height")],
        )
    if payload.get("format_version") != V2_FORMAT_VERSION:
        return ParseBundle(
            "v2",
            width,
            height,
            failures=[_failure("v2:package", "decode_v2", "format_version must be v2")],
        )
    if width <= 0 or height <= 0:
        return ParseBundle(
            "v2",
            width,
            height,
            failures=[_failure("v2:package", "decode_v2", "invalid image dimensions")],
        )
    entries = payload.get("instances")
    if not isinstance(entries, list):
        return ParseBundle(
            "v2",
            width,
            height,
            failures=[_failure("v2:package", "decode_v2", "instances must be a list")],
        )
    v2_tasks: list[OrderedTask[Any]] = []
    for index, entry in enumerate(entries):
        raw_id = entry.get("instance_id") if isinstance(entry, dict) else None
        try:
            task_id = int(raw_id)
        except (TypeError, ValueError):
            task_id = -(index + 1)
        v2_tasks.append(
            OrderedTask(
                instance_id=task_id,
                stage="decode_v2",
                payload=entry,
                source_format="v2",
            )
        )
    decoded = ordered_parallel_map(
        v2_tasks,
        lambda entry: _decode_v2_instance(
            entry,
            width=width,
            height=height,
            config=config,
        ),
        workers,
    )
    instances: list[InstanceMask] = []
    failures: list[Failure] = []
    seen_ids: set[int] = set()
    for instance, failure in decoded:
        if instance is not None:
            if instance.instance_id in seen_ids:
                failures.append(
                    _failure(instance.key, "decode_v2", "duplicate instance ID")
                )
                continue
            seen_ids.add(instance.instance_id)
            instances.append(instance)
        if failure is not None:
            failures.append(failure)
    return ParseBundle(
        "v2",
        width,
        height,
        instances=sorted(
            instances,
            key=lambda item: (item.source_format, item.instance_id, item.key),
        ),
        failures=sorted(failures, key=lambda item: item.stable_sort_key),
    )

"""Loaders for source declarations and window parameter files."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .errors import FormatError
from .models import SUPPORTED_SCALES, SourceDeclaration, WindowParams


@dataclass(frozen=True, slots=True)
class SourceSetConfig:
    """Observed source collection declared on disk."""

    source_set_version: int
    sources: tuple[SourceDeclaration, ...]
    base_dir: str = ""

    def by_id(self) -> dict[str, SourceDeclaration]:
        return {src.source_id: src for src in self.sources}


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """All session parameters (windowing, scheduling, leasing)."""

    windows: WindowParams
    global_concurrency: int = 2
    per_source_concurrency: int = 1
    lease_ttl_ms: int = 60_000
    # Version tag of the fixed-point scale convention used by this session.
    scale_convention_version: int = 1


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FormatError(f"config file not found: {path}",
                          field="file") from exc
    except json.JSONDecodeError as exc:
        raise FormatError(f"invalid JSON in {path}: {exc.msg}",
                          field="file", line_no=exc.lineno) from exc
    if not isinstance(data, dict):
        raise FormatError(f"top-level JSON object expected in {path}",
                          field="file")
    return data


def load_source_set(path: str) -> SourceSetConfig:
    data = _read_json(path)
    base_dir = os.path.dirname(os.path.abspath(path))

    version = data.get("source_set_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise FormatError("source_set_version must be a positive integer",
                          field="source_set_version")
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise FormatError("sources must be a non-empty array",
                          field="sources")

    sources: list[SourceDeclaration] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_sources):
        prefix = f"sources[{index}]"
        if not isinstance(raw, dict):
            raise FormatError(f"{prefix} must be an object", field="sources")
        source_id = raw.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise FormatError(f"{prefix}.source_id must be a non-empty string",
                              field="source_id")
        if source_id in seen:
            raise FormatError(f"duplicate source_id {source_id!r}",
                              field="source_id")
        seen.add(source_id)

        format_version = raw.get("format_version")
        if not isinstance(format_version, int) or isinstance(
                format_version, bool) or format_version not in (1, 2, 3):
            raise FormatError(
                f"{prefix}.format_version must be one of 1, 2, 3",
                field="format_version")
        metric_key = raw.get("default_metric_key")
        if not isinstance(metric_key, str) or not metric_key:
            raise FormatError(
                f"{prefix}.default_metric_key must be a non-empty string",
                field="default_metric_key")
        default_scale = raw.get("default_scale")
        if not isinstance(default_scale, int) or isinstance(
                default_scale, bool) or default_scale not in SUPPORTED_SCALES:
            raise FormatError(
                f"{prefix}.default_scale must be in {list(SUPPORTED_SCALES)}",
                field="default_scale")
        mapping = raw.get("metric_mapping", {})
        if not isinstance(mapping, dict) or not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in mapping.items()):
            raise FormatError(
                f"{prefix}.metric_mapping must map string to string",
                field="metric_mapping")
        file_path = raw.get("file_path", "")
        if not isinstance(file_path, str) or not file_path:
            raise FormatError(f"{prefix}.file_path must be a non-empty string",
                              field="file_path")

        sources.append(SourceDeclaration(
            source_id=source_id,
            format_version=format_version,
            default_metric_key=metric_key,
            default_scale=default_scale,
            metric_mapping=dict(mapping),
            file_path=file_path,
        ))

    return SourceSetConfig(source_set_version=version,
                           sources=tuple(sources), base_dir=base_dir)


def load_engine_config(path: str) -> EngineConfig:
    data = _read_json(path)
    window_size = data.get("window_size")
    slide_step = data.get("slide_step", 1)
    scale = data.get("scale")
    if not isinstance(window_size, int) or isinstance(window_size, bool) \
            or window_size < 1:
        raise FormatError("window_size must be a positive integer",
                          field="window_size")
    if not isinstance(slide_step, int) or isinstance(slide_step, bool) \
            or slide_step < 1:
        raise FormatError("slide_step must be a positive integer",
                          field="slide_step")
    if not isinstance(scale, int) or isinstance(scale, bool) \
            or scale not in SUPPORTED_SCALES:
        raise FormatError(f"scale must be in {list(SUPPORTED_SCALES)}",
                          field="scale")
    windows = WindowParams(window_size=window_size, slide_step=slide_step,
                           scale=scale)
    windows.validate()

    global_concurrency = data.get("global_concurrency", 2)
    per_source_concurrency = data.get("per_source_concurrency", 1)
    lease_ttl_ms = data.get("lease_ttl_ms", 60_000)
    scale_convention_version = data.get("scale_convention_version", 1)
    for name, value in (("global_concurrency", global_concurrency),
                        ("per_source_concurrency", per_source_concurrency),
                        ("lease_ttl_ms", lease_ttl_ms)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise FormatError(f"{name} must be a positive integer",
                              field=name)
    if not isinstance(scale_convention_version, int) or isinstance(
            scale_convention_version, bool) or scale_convention_version < 1:
        raise FormatError("scale_convention_version must be a positive int",
                          field="scale_convention_version")
    if per_source_concurrency > global_concurrency:
        raise FormatError(
            "per_source_concurrency cannot exceed global_concurrency",
            field="per_source_concurrency")

    return EngineConfig(
        windows=windows,
        global_concurrency=global_concurrency,
        per_source_concurrency=per_source_concurrency,
        lease_ttl_ms=lease_ttl_ms,
        scale_convention_version=scale_convention_version,
    )

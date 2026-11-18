from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ConfigFormatError
from .models import (
    ConfigDocument,
    Derivation,
    LIMIT_DIMENSION_BYTES,
    LIMIT_DIMENSION_OPERATIONS,
    LIMIT_DIMENSIONS,
    VolumeConfig,
)
from .utils import atomic_write_json


CURRENT_FORMAT_VERSION = 3
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _format_error(
    message: str,
    *,
    volume_id: str | None = None,
    field_path: str | None = None,
) -> ConfigFormatError:
    return ConfigFormatError(message, volume_id=volume_id, field_path=field_path)


def _required_nonnegative_int(
    container: dict[str, Any],
    key: str,
    *,
    volume_id: str,
    field_path: str,
) -> int:
    if key not in container:
        raise _format_error(
            f"missing required field {key}",
            volume_id=volume_id,
            field_path=field_path,
        )
    value = container[key]
    if not _is_int(value):
        raise _format_error(
            f"field {key} must be an integer",
            volume_id=volume_id,
            field_path=field_path,
        )
    if value < 0:
        raise _format_error(
            f"field {key} must not be negative",
            volume_id=volume_id,
            field_path=field_path,
        )
    return value


def _detect_version(document: dict[str, Any]) -> tuple[int, str]:
    key = None
    for candidate in ("format_version", "version", "schema_version"):
        if candidate in document:
            key = candidate
            break
    if key is None:
        raise _format_error(
            "missing format version",
            field_path="$.format_version",
        )
    value = document[key]
    path = f"$.{key}"
    if not _is_int(value):
        raise _format_error("format version must be an integer", field_path=path)
    if value not in SUPPORTED_FORMAT_VERSIONS:
        raise _format_error(f"unknown format version {value!r}", field_path=path)
    return value, path


def _normalize_limit_dimension(value: Any, *, volume_id: str, path: str) -> str:
    if not isinstance(value, str):
        raise _format_error(
            "limit_dimension must be a string",
            volume_id=volume_id,
            field_path=path,
        )
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "bytes": LIMIT_DIMENSION_BYTES,
        "byte": LIMIT_DIMENSION_BYTES,
        "write_bytes": LIMIT_DIMENSION_BYTES,
        "write": LIMIT_DIMENSION_BYTES,
        "ops": LIMIT_DIMENSION_OPERATIONS,
        "op": LIMIT_DIMENSION_OPERATIONS,
        "operations": LIMIT_DIMENSION_OPERATIONS,
        "operation": LIMIT_DIMENSION_OPERATIONS,
        "operation_count": LIMIT_DIMENSION_OPERATIONS,
    }
    if normalized not in aliases:
        raise _format_error(
            f"limit_dimension must be one of {list(LIMIT_DIMENSIONS)}",
            volume_id=volume_id,
            field_path=path,
        )
    return aliases[normalized]


def _parse_derivation(volume: dict[str, Any], volume_id: str, index: int) -> Derivation | None:
    base = f"$.volumes[{index}]"
    if "derivation" in volume:
        raw = volume["derivation"]
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise _format_error(
                "derivation must be an object or null",
                volume_id=volume_id,
                field_path=f"{base}.derivation",
            )
        required = ("upstream_id", "factor", "offset")
        for key in required:
            if key not in raw:
                raise _format_error(
                    f"missing required derivation field {key}",
                    volume_id=volume_id,
                    field_path=f"{base}.derivation.{key}",
                )
        upstream = raw["upstream_id"]
        if not isinstance(upstream, str) or not upstream.strip():
            raise _format_error(
                "derivation.upstream_id must be a non-empty string",
                volume_id=volume_id,
                field_path=f"{base}.derivation.upstream_id",
            )
        factor = _required_nonnegative_int(
            raw,
            "factor",
            volume_id=volume_id,
            field_path=f"{base}.derivation.factor",
        )
        offset = _required_nonnegative_int(
            raw,
            "offset",
            volume_id=volume_id,
            field_path=f"{base}.derivation.offset",
        )
        return Derivation(upstream_id=upstream.strip(), factor=factor, offset=offset)

    # Accept a flat form for hand-authored v3 files, while keeping the nested
    # form as the canonical encoding written by the application.
    flat_upstream = "derived_from" if "derived_from" in volume else "upstream_id"
    if flat_upstream not in volume:
        return None
    upstream = volume[flat_upstream]
    factor_key = "derive_factor" if "derive_factor" in volume else "factor"
    offset_key = "derive_offset" if "derive_offset" in volume else "offset"
    if not isinstance(upstream, str) or not upstream.strip():
        raise _format_error(
            f"{flat_upstream} must be a non-empty string",
            volume_id=volume_id,
            field_path=f"{base}.{flat_upstream}",
        )
    for key in (factor_key, offset_key):
        if key not in volume:
            raise _format_error(
                f"missing required derivation field {key}",
                volume_id=volume_id,
                field_path=f"{base}.{key}",
            )
    factor = _required_nonnegative_int(
        volume,
        factor_key,
        volume_id=volume_id,
        field_path=f"{base}.{factor_key}",
    )
    offset = _required_nonnegative_int(
        volume,
        offset_key,
        volume_id=volume_id,
        field_path=f"{base}.{offset_key}",
    )
    return Derivation(upstream_id=upstream.strip(), factor=factor, offset=offset)


def parse_document(
    raw: Any,
    *,
    source_path: str | None = None,
) -> ConfigDocument:
    """Validate and normalize a v1/v2/v3 document to the current schema.

    Validation is all-or-nothing.  The first offending volume and JSON-style
    field path are attached to every ``E_FORMAT`` error.
    """

    if not isinstance(raw, dict):
        raise _format_error("configuration document must be a JSON object", field_path="$")

    version, version_path = _detect_version(raw)

    revision = 1
    if "revision" in raw:
        revision_value = raw["revision"]
        if not _is_int(revision_value):
            raise _format_error("revision must be an integer", field_path="$.revision")
        if revision_value < 1:
            raise _format_error("revision must be at least 1", field_path="$.revision")
        revision = revision_value

    name = raw.get("name", "storage-config")
    if not isinstance(name, str) or not name.strip():
        raise _format_error("name must be a non-empty string", field_path="$.name")

    original_version = version
    if version == 3 and "upgraded_from" in raw:
        upgraded_from = raw["upgraded_from"]
        if not _is_int(upgraded_from) or upgraded_from not in SUPPORTED_FORMAT_VERSIONS:
            raise _format_error(
                "upgraded_from must be one of 1, 2, or 3",
                field_path="$.upgraded_from",
            )
        original_version = upgraded_from

    if "volumes" not in raw:
        raise _format_error("missing volumes", field_path="$.volumes")
    raw_volumes = raw["volumes"]
    if not isinstance(raw_volumes, list):
        raise _format_error("volumes must be an array", field_path="$.volumes")
    if not raw_volumes:
        raise _format_error("volumes must not be empty", field_path="$.volumes")

    volumes: list[VolumeConfig] = []
    seen_ids: set[str] = set()
    for index, raw_volume in enumerate(raw_volumes):
        base = f"$.volumes[{index}]"
        if not isinstance(raw_volume, dict):
            raise _format_error(
                "volume entry must be an object",
                field_path=base,
            )
        if "id" in raw_volume:
            id_key = "id"
        elif "volume_id" in raw_volume:
            id_key = "volume_id"
        else:
            raise _format_error(
                "missing volume id",
                field_path=f"{base}.id",
            )
        volume_id_value = raw_volume[id_key]
        if not isinstance(volume_id_value, str) or not volume_id_value.strip():
            raise _format_error(
                "volume id must be a non-empty string",
                field_path=f"{base}.{id_key}",
            )
        volume_id = volume_id_value.strip()
        if volume_id in seen_ids:
            raise _format_error(
                f"duplicate volume id {volume_id!r}",
                volume_id=volume_id,
                field_path=f"{base}.{id_key}",
            )
        seen_ids.add(volume_id)

        capacity = _required_nonnegative_int(
            raw_volume,
            "capacity_bytes",
            volume_id=volume_id,
            field_path=f"{base}.capacity_bytes",
        )
        ops_per_sec = 0
        burst_capacity = 0
        limit_dimension = LIMIT_DIMENSION_BYTES
        derivation: Derivation | None = None

        if version >= 2:
            ops_per_sec = _required_nonnegative_int(
                raw_volume,
                "ops_per_sec",
                volume_id=volume_id,
                field_path=f"{base}.ops_per_sec",
            )
            burst_capacity = _required_nonnegative_int(
                raw_volume,
                "burst_capacity",
                volume_id=volume_id,
                field_path=f"{base}.burst_capacity",
            )
        if version >= 3:
            if "limit_dimension" not in raw_volume:
                raise _format_error(
                    "missing required field limit_dimension",
                    volume_id=volume_id,
                    field_path=f"{base}.limit_dimension",
                )
            limit_dimension = _normalize_limit_dimension(
                raw_volume["limit_dimension"],
                volume_id=volume_id,
                path=f"{base}.limit_dimension",
            )
            derivation = _parse_derivation(raw_volume, volume_id, index)

        volumes.append(
            VolumeConfig(
                volume_id=volume_id,
                capacity_bytes=capacity,
                ops_per_sec=ops_per_sec,
                burst_capacity=burst_capacity,
                limit_dimension=limit_dimension,
                derivation=derivation,
            )
        )

    metadata = raw.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise _format_error("metadata must be an object", field_path="$.metadata")

    return ConfigDocument(
        revision=revision,
        volumes=tuple(volumes),
        schema_version=CURRENT_FORMAT_VERSION,
        original_version=original_version,
        name=name.strip(),
        source_path=source_path,
        metadata=dict(metadata),
    )


def load_document(path: str | Path) -> ConfigDocument:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise _format_error(
            f"cannot read configuration: {exc}",
            field_path="$",
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _format_error(
            f"invalid JSON: {exc.msg}",
            field_path=f"$ (line {exc.lineno}, column {exc.colno})",
        ) from exc
    document = parse_document(raw, source_path=str(source))
    return document


def write_document(path: str | Path, document: ConfigDocument) -> None:
    atomic_write_json(path, document.to_json())


def peek_version(path: str | Path) -> int:
    """Read just enough JSON to identify the declared format version."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _format_error(f"cannot read configuration version: {exc}", field_path="$") from exc
    if not isinstance(raw, dict):
        raise _format_error("configuration document must be a JSON object", field_path="$")
    version, _ = _detect_version(raw)
    return version

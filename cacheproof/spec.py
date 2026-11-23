"""Strict loading and normalization of the local cache declaration."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .canonical import digest_value, utf8_key
from .content import entry_digest
from .errors import spec_error
from .models import (
    CacheSpec,
    EntrySpec,
    SegmentSpec,
    SnapshotEntrySpec,
    ViewSegmentSpec,
    ViewSpec,
)


_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, field: str, segment_id: str | None = None) -> dict:
    if not isinstance(value, dict):
        raise spec_error("expected a JSON object", segment_id=segment_id, field=field)
    return value


def _require_list(value: Any, field: str, segment_id: str | None = None) -> list:
    if not isinstance(value, list):
        raise spec_error("expected a JSON array", segment_id=segment_id, field=field)
    return value


def _require_text(
    value: Any,
    field: str,
    *,
    segment_id: str | None = None,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and value == ""):
        raise spec_error(
            "expected a non-empty string" if not allow_empty else "expected a string",
            segment_id=segment_id,
            field=field,
        )
    return value


def _normalize_entry_digest(
    value: Any,
    *,
    view_id: str,
    segment_id: str,
    field: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise spec_error(
            "digest must be sha256:<64 lowercase hex characters>",
            segment_id=segment_id,
            field=f"views[{view_id}].{field}",
        )
    return value


def load_spec(path: str | Path) -> CacheSpec:
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise spec_error(
            f"cannot read declaration file: {exc}",
            field="source_path",
            details={"path": str(source)},
        ) from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise spec_error(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            field="source_path",
            details={"path": str(source), "line": exc.lineno, "column": exc.colno},
        ) from exc
    return parse_spec(document, source_path=source)


def parse_spec(document: Any, *, source_path: Path | None = None) -> CacheSpec:
    root = _require_object(document, "$")
    format_version = root.get("format_version")
    if not _is_int(format_version) or format_version != 1:
        raise spec_error(
            "format_version must be the integer 1", field="format_version"
        )

    raw_segments = _require_list(root.get("segments"), "segments")
    segments: list[SegmentSpec] = []
    segment_by_id: dict[str, SegmentSpec] = {}
    declared_digests: dict[tuple[str, str], str] = {}

    for segment_position, raw_segment in enumerate(raw_segments):
        segment_obj = _require_object(
            raw_segment, f"segments[{segment_position}]"
        )
        segment_id_value = segment_obj.get("segment_id")
        if not isinstance(segment_id_value, str) or segment_id_value == "":
            raise spec_error(
                "segment_id must be a non-empty string",
                field="segment_id",
                details={"index": segment_position},
            )
        segment_id = segment_id_value
        if segment_id in segment_by_id:
            raise spec_error(
                "duplicate segment_id",
                segment_id=segment_id,
                field="segment_id",
                details={"index": segment_position},
            )

        capacity = segment_obj.get("capacity_bytes")
        if not _is_int(capacity) or capacity <= 0:
            raise spec_error(
                "capacity_bytes must be a positive integer",
                segment_id=segment_id,
                field="capacity_bytes",
            )

        raw_entries = _require_list(
            segment_obj.get("entries"), "entries", segment_id=segment_id
        )
        entries: list[EntrySpec] = []
        entry_keys: set[str] = set()
        for entry_position, raw_entry in enumerate(raw_entries):
            entry_obj = _require_object(
                raw_entry,
                f"entries[{entry_position}]",
                segment_id=segment_id,
            )
            key = _require_text(
                entry_obj.get("key"), "key", segment_id=segment_id
            )
            if key in entry_keys:
                raise spec_error(
                    "duplicate entry key within segment",
                    segment_id=segment_id,
                    entry_key=key,
                    field="entries.key",
                )
            size = entry_obj.get("size_bytes")
            if not _is_int(size) or size < 0:
                raise spec_error(
                    "size_bytes must be a non-negative integer",
                    segment_id=segment_id,
                    entry_key=key,
                    field="entries.size_bytes",
                )
            seed_value = entry_obj.get("content_seed")
            if not isinstance(seed_value, str):
                raise spec_error(
                    "content_seed must be a string",
                    segment_id=segment_id,
                    entry_key=key,
                    field="entries.content_seed",
                )
            entry = EntrySpec(key=key, size_bytes=size, content_seed=seed_value)
            entries.append(entry)
            entry_keys.add(key)
            declared_digests[(segment_id, key)] = entry_digest(
                key, size, seed_value
            )

        declared_used = sum(entry.size_bytes for entry in entries)
        if declared_used > capacity:
            raise spec_error(
                "declared entry bytes exceed segment capacity",
                segment_id=segment_id,
                field="capacity_bytes",
                details={
                    "declared_used_bytes": declared_used,
                    "capacity_bytes": capacity,
                },
            )
        segment = SegmentSpec(
            segment_id=segment_id,
            capacity_bytes=capacity,
            entries=tuple(entries),
        )
        segments.append(segment)
        segment_by_id[segment_id] = segment

    raw_views = _require_list(root.get("views"), "views")
    views: list[ViewSpec] = []
    last_version: dict[str, int] = {}
    seen_view_versions: set[tuple[str, int]] = set()
    for view_position, raw_view in enumerate(raw_views):
        view_obj = _require_object(raw_view, f"views[{view_position}]")
        view_id = _require_text(
            view_obj.get("view_id"), "view_id"
        )
        version = view_obj.get("version")
        if not _is_int(version) or version < 0:
            raise spec_error(
                "view version must be a non-negative integer",
                field=f"views[{view_id}].version",
                details={"view_id": view_id},
            )
        if view_id in last_version and version <= last_version[view_id]:
            raise spec_error(
                "view versions must be strictly increasing per view_id in declaration order",
                field=f"views[{view_id}].version",
                details={
                    "view_id": view_id,
                    "previous_version": last_version[view_id],
                    "version": version,
                },
            )
        if (view_id, version) in seen_view_versions:
            raise spec_error(
                "duplicate view_id/version pair",
                field=f"views[{view_id}].version",
                details={"view_id": view_id, "version": version},
            )
        last_version[view_id] = version
        seen_view_versions.add((view_id, version))

        raw_view_segments = _require_list(
            view_obj.get("segments"),
            f"views[{view_id}].segments",
        )
        view_segments: list[ViewSegmentSpec] = []
        seen_view_segments: set[str] = set()
        for view_segment_position, raw_view_segment in enumerate(
            raw_view_segments
        ):
            view_segment_obj = _require_object(
                raw_view_segment,
                f"views[{view_id}].segments[{view_segment_position}]",
            )
            segment_id = _require_text(
                view_segment_obj.get("segment_id"),
                "segment_id",
            )
            if segment_id not in segment_by_id:
                raise spec_error(
                    "view references an undeclared segment",
                    segment_id=segment_id,
                    field=f"views[{view_id}].segments.segment_id",
                )
            if segment_id in seen_view_segments:
                raise spec_error(
                    "duplicate segment reference within view",
                    segment_id=segment_id,
                    field=f"views[{view_id}].segments.segment_id",
                )
            seen_view_segments.add(segment_id)

            raw_snapshot_entries = _require_list(
                view_segment_obj.get("entries"),
                f"views[{view_id}].segments[{segment_id}].entries",
            )
            snapshot_entries: list[SnapshotEntrySpec] = []
            seen_snapshot_keys: set[str] = set()
            for snapshot_entry_position, raw_snapshot_entry in enumerate(
                raw_snapshot_entries
            ):
                if isinstance(raw_snapshot_entry, str):
                    key = _require_text(
                        raw_snapshot_entry,
                        f"views[{view_id}].segments[{segment_id}].entries.key",
                        segment_id=segment_id,
                    )
                    explicit_digest = None
                else:
                    snapshot_entry_obj = _require_object(
                        raw_snapshot_entry,
                        f"views[{view_id}].segments[{segment_id}].entries[{snapshot_entry_position}]",
                        segment_id=segment_id,
                    )
                    key = _require_text(
                        snapshot_entry_obj.get("key"),
                        f"views[{view_id}].segments[{segment_id}].entries.key",
                        segment_id=segment_id,
                    )
                    explicit_digest = _normalize_entry_digest(
                        snapshot_entry_obj.get("digest"),
                        view_id=view_id,
                        segment_id=segment_id,
                        field=f"segments[{segment_id}].entries[{key}].digest",
                    )
                if key in seen_snapshot_keys:
                    raise spec_error(
                        "duplicate snapshot entry key within a view segment",
                        segment_id=segment_id,
                        entry_key=key,
                        field=f"views[{view_id}].segments[{segment_id}].entries.key",
                    )
                seen_snapshot_keys.add(key)
                resolved_digest = explicit_digest
                if resolved_digest is None:
                    resolved_digest = declared_digests.get((segment_id, key))
                snapshot_entries.append(
                    SnapshotEntrySpec(key=key, digest=resolved_digest)
                )

            snapshot_entries.sort(key=lambda item: utf8_key(item.key))
            view_segments.append(
                ViewSegmentSpec(
                    segment_id=segment_id,
                    entries=tuple(snapshot_entries),
                )
            )
        view_segments.sort(key=lambda item: utf8_key(item.segment_id))
        views.append(
            ViewSpec(
                view_id=view_id,
                version=version,
                segments=tuple(view_segments),
            )
        )

    return CacheSpec(
        format_version=format_version,
        segments=tuple(segments),
        views=tuple(views),
        source_path=source_path,
    )


def normalized_spec_payload(spec: CacheSpec) -> dict:
    segments = []
    for segment in sorted(spec.segments, key=lambda item: utf8_key(item.segment_id)):
        entries = [
            {
                "key": entry.key,
                "size_bytes": entry.size_bytes,
                "content_seed": entry.content_seed,
            }
            for entry in sorted(segment.entries, key=lambda item: utf8_key(item.key))
        ]
        segments.append(
            {
                "segment_id": segment.segment_id,
                "capacity_bytes": segment.capacity_bytes,
                "entries": entries,
            }
        )

    views = []
    for view in sorted(spec.views, key=lambda item: (utf8_key(item.view_id), item.version)):
        view_segments = []
        for view_segment in view.segments:
            view_segments.append(
                {
                    "segment_id": view_segment.segment_id,
                    "entries": [
                        {"key": entry.key, "digest": entry.digest}
                        for entry in view_segment.entries
                    ],
                }
            )
        views.append(
            {
                "view_id": view.view_id,
                "version": view.version,
                "segments": view_segments,
            }
        )
    return {"format_version": spec.format_version, "segments": segments, "views": views}


def spec_digest(spec: CacheSpec) -> str:
    return digest_value(normalized_spec_payload(spec))


def latest_views(spec: CacheSpec) -> list[ViewSpec]:
    latest: dict[str, ViewSpec] = {}
    for view in spec.views:
        current = latest.get(view.view_id)
        if current is None or view.version > current.version:
            latest[view.view_id] = view
    return sorted(latest.values(), key=lambda item: utf8_key(item.view_id))

"""Read-view snapshot construction and stable difference calculation."""

from __future__ import annotations

from typing import Any, Iterable

from .canonical import digest_value, utf8_key


VIEW_FORMAT_VERSION = 1


def _normalized_segments(segments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for segment in sorted(segments, key=lambda item: utf8_key(str(item["segment_id"]))):
        entries = [
            {
                "key": str(entry["key"]),
                "digest": entry.get("digest"),
            }
            for entry in sorted(segment.get("entries", []), key=lambda item: utf8_key(str(item["key"])))
        ]
        normalized.append(
            {
                "segment_id": str(segment["segment_id"]),
                "segment_version": int(segment["segment_version"]),
                "entries": entries,
            }
        )
    return normalized


def build_view_snapshot(
    *,
    view_id: str,
    view_version: int,
    segment_set_version: int,
    segments: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": VIEW_FORMAT_VERSION,
        "view_id": view_id,
        "view_version": view_version,
        "segment_set_version": segment_set_version,
        "segments": _normalized_segments(segments),
    }
    payload["content_digest"] = digest_value(payload)
    return payload


def validate_view_snapshot(snapshot: Any) -> dict[str, Any]:
    from .errors import DiagnosticError

    if not isinstance(snapshot, dict):
        raise DiagnosticError(
            "E_SNAPSHOT", "read-view snapshot must be an object", field="snapshot"
        )
    stored_digest = snapshot.get("content_digest")
    payload = {key: value for key, value in snapshot.items() if key != "content_digest"}
    expected = digest_value(payload)
    if stored_digest != expected:
        raise DiagnosticError(
            "E_SNAPSHOT",
            "read-view snapshot content digest mismatch",
            field="snapshot.content_digest",
            details={
                "stored_digest": stored_digest,
                "computed_digest": expected,
            },
        )
    return snapshot


def _entries_map(segment: dict[str, Any]) -> dict[str, Any]:
    return {str(entry["key"]): entry.get("digest") for entry in segment.get("entries", [])}


def view_is_stale(old: dict[str, Any], current: dict[str, Any]) -> bool:
    old_segments = {segment["segment_id"]: segment for segment in old.get("segments", [])}
    current_segments = {
        segment["segment_id"]: segment for segment in current.get("segments", [])
    }
    if set(old_segments) != set(current_segments):
        return True
    for segment_id in old_segments:
        old_segment = old_segments[segment_id]
        current_segment = current_segments[segment_id]
        if old_segment.get("segment_version") != current_segment.get("segment_version"):
            return True
        if _entries_map(old_segment) != _entries_map(current_segment):
            return True
    return False


def compute_view_diff(old: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    old_segments = {segment["segment_id"]: segment for segment in old.get("segments", [])}
    current_segments = {
        segment["segment_id"]: segment for segment in current.get("segments", [])
    }
    differences: list[dict[str, Any]] = []
    for segment_id in sorted(
        set(old_segments) | set(current_segments), key=utf8_key
    ):
        old_segment = old_segments.get(segment_id)
        current_segment = current_segments.get(segment_id)
        old_entries = _entries_map(old_segment or {"entries": []})
        current_entries = _entries_map(current_segment or {"entries": []})
        added = [
            {"key": key, "digest": current_entries[key]}
            for key in sorted(set(current_entries) - set(old_entries), key=utf8_key)
        ]
        deleted = [
            {"key": key, "digest": old_entries[key]}
            for key in sorted(set(old_entries) - set(current_entries), key=utf8_key)
        ]
        digest_changed = [
            {
                "key": key,
                "old_digest": old_entries[key],
                "new_digest": current_entries[key],
            }
            for key in sorted(set(old_entries) & set(current_entries), key=utf8_key)
            if old_entries[key] != current_entries[key]
        ]
        old_version = None if old_segment is None else old_segment.get("segment_version")
        new_version = (
            None if current_segment is None else current_segment.get("segment_version")
        )
        if (
            added
            or deleted
            or digest_changed
            or old_version != new_version
            or old_segment is None
            or current_segment is None
        ):
            differences.append(
                {
                    "segment_id": segment_id,
                    "old_segment_version": old_version,
                    "new_segment_version": new_version,
                    "added": added,
                    "deleted": deleted,
                    "digest_changed": digest_changed,
                }
            )
    return {
        "old_view_version": old.get("view_version"),
        "new_view_version": current.get("view_version"),
        "old_segment_set_version": old.get("segment_set_version"),
        "new_segment_set_version": current.get("segment_set_version"),
        "segments": differences,
    }


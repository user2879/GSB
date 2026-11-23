"""Session snapshot sealing, diffing and deterministic replay checks."""

from __future__ import annotations

from typing import Any, Iterable

from .canonical import canonical_text, digest_value, utf8_key
from .errors import DiagnosticError
from .proof import build_segment_proof, verify_proof_digest
from .views import (
    build_view_snapshot,
    compute_view_diff,
    validate_view_snapshot,
    view_is_stale,
)


SESSION_FORMAT_VERSION = 1


def seal_session_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = {key: value for key, value in payload.items() if key != "content_digest"}
    sealed["content_digest"] = digest_value(sealed)
    return sealed


def verify_session_snapshot(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise DiagnosticError(
            "E_SNAPSHOT", "session snapshot must be a JSON object", field="snapshot"
        )
    if snapshot.get("format_version") != SESSION_FORMAT_VERSION:
        raise DiagnosticError(
            "E_SNAPSHOT",
            "unsupported session snapshot format version",
            field="snapshot.format_version",
            details={
                "stored": snapshot.get("format_version"),
                "supported": SESSION_FORMAT_VERSION,
            },
        )
    stored = snapshot.get("content_digest")
    payload = {key: value for key, value in snapshot.items() if key != "content_digest"}
    computed = digest_value(payload)
    if stored != computed:
        raise DiagnosticError(
            "E_SNAPSHOT",
            "session snapshot content digest mismatch",
            field="snapshot.content_digest",
            details={"stored_digest": stored, "computed_digest": computed},
        )
    required_ints = (
        "segment_set_version",
        "ledger_batch",
        "completed_cursor",
        "last_operation_seq",
    )
    for field_name in required_ints:
        value = snapshot.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise DiagnosticError(
                "E_SNAPSHOT",
                "session snapshot field must be a non-negative integer",
                field=f"snapshot.{field_name}",
            )
    if not isinstance(snapshot.get("segments"), list):
        raise DiagnosticError(
            "E_SNAPSHOT", "session snapshot segments must be a list", field="snapshot.segments"
        )
    if not isinstance(snapshot.get("views"), list):
        raise DiagnosticError(
            "E_SNAPSHOT", "session snapshot views must be a list", field="snapshot.views"
        )
    seen_segments: set[str] = set()
    for proof in snapshot["segments"]:
        try:
            verify_proof_digest(proof)
        except DiagnosticError as exc:
            raise DiagnosticError(
                "E_SNAPSHOT",
                "snapshot contains an invalid capacity proof",
                segment_id=exc.segment_id,
                field="snapshot.segments.proof",
                details={"cause": exc.to_dict()},
            ) from exc
        segment_id = proof.get("segment_id")
        if segment_id in seen_segments:
            raise DiagnosticError(
                "E_SNAPSHOT",
                "session snapshot contains duplicate segment proofs",
                segment_id=segment_id,
                field="snapshot.segments.segment_id",
            )
        seen_segments.add(segment_id)
        if proof.get("used_bytes", 0) + proof.get("remaining_bytes", 0) != proof.get(
            "capacity_bytes"
        ):
            raise DiagnosticError(
                "E_SNAPSHOT",
                "snapshot proof bytes do not sum to capacity",
                segment_id=proof.get("segment_id"),
                field="snapshot.segments.used_bytes",
            )
    seen_views: set[str] = set()
    for view in snapshot["views"]:
        if not isinstance(view, dict):
            raise DiagnosticError(
                "E_SNAPSHOT",
                "session snapshot view must be an object",
                field="snapshot.views",
            )
        view_id = view.get("view_id")
        if not isinstance(view_id, str) or not view_id:
            raise DiagnosticError(
                "E_SNAPSHOT",
                "session snapshot view_id is invalid",
                field="snapshot.views.view_id",
            )
        if view_id in seen_views:
            raise DiagnosticError(
                "E_SNAPSHOT",
                "session snapshot contains duplicate views",
                field=f"snapshot.views[{view_id}]",
            )
        seen_views.add(view_id)
        view_snapshot = validate_view_snapshot(view.get("snapshot"))
        if view_snapshot.get("view_id") != view_id:
            raise DiagnosticError(
                "E_SNAPSHOT",
                "session view_id differs from its embedded snapshot",
                field=f"snapshot.views[{view_id}].snapshot.view_id",
            )
        if view_snapshot.get("view_version") != view.get("view_version"):
            raise DiagnosticError(
                "E_SNAPSHOT",
                "session view version differs from its embedded snapshot",
                field=f"snapshot.views[{view_id}].view_version",
            )
        if view.get("snapshot_digest") != view_snapshot.get("content_digest"):
            raise DiagnosticError(
                "E_SNAPSHOT",
                "session view snapshot digest mismatch",
                field=f"snapshot.views[{view_id}].snapshot_digest",
            )
    return snapshot


def _entry_map(segment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(entry["key"]): {
            "size_bytes": int(entry["size_bytes"]),
            "digest": str(entry["digest"]),
        }
        for entry in segment.get("entries", [])
    }


def snapshot_diff(old: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    changes: list[dict[str, Any]] = []
    for field_name in (
        "segment_set_version",
        "completed_cursor",
        "last_operation_seq",
        "ledger_batch",
    ):
        if old.get(field_name) != current.get(field_name):
            changes.append(
                {
                    "scope": "session",
                    "field": field_name,
                    "old": old.get(field_name),
                    "new": current.get(field_name),
                }
            )

    old_segments = {proof["segment_id"]: proof for proof in old.get("segments", [])}
    current_segments = {
        proof["segment_id"]: proof for proof in current.get("segments", [])
    }
    for segment_id in sorted(set(old_segments) | set(current_segments), key=utf8_key):
        old_proof = old_segments.get(segment_id)
        current_proof = current_segments.get(segment_id)
        if old_proof is None or current_proof is None:
            changes.append(
                {
                    "scope": "segment",
                    "segment_id": segment_id,
                    "field": "presence",
                    "old": old_proof is not None,
                    "new": current_proof is not None,
                }
            )
            continue
        for field_name in (
            "capacity_bytes",
            "used_bytes",
            "remaining_bytes",
            "entry_count",
            "segment_version",
            "segment_set_version",
            "content_digest",
        ):
            if old_proof.get(field_name) != current_proof.get(field_name):
                changes.append(
                    {
                        "scope": "segment",
                        "segment_id": segment_id,
                        "field": field_name,
                        "old": old_proof.get(field_name),
                        "new": current_proof.get(field_name),
                    }
                )
        old_entries = _entry_map(old_proof)
        current_entries = _entry_map(current_proof)
        for key in sorted(set(old_entries) | set(current_entries), key=utf8_key):
            if key not in old_entries:
                changes.append(
                    {
                        "scope": "entry",
                        "segment_id": segment_id,
                        "entry_key": key,
                        "field": "presence",
                        "old": None,
                        "new": "present",
                    }
                )
            elif key not in current_entries:
                changes.append(
                    {
                        "scope": "entry",
                        "segment_id": segment_id,
                        "entry_key": key,
                        "field": "presence",
                        "old": "present",
                        "new": None,
                    }
                )
            else:
                for field_name in ("size_bytes", "digest"):
                    if old_entries[key][field_name] != current_entries[key][field_name]:
                        changes.append(
                            {
                                "scope": "entry",
                                "segment_id": segment_id,
                                "entry_key": key,
                                "field": field_name,
                                "old": old_entries[key][field_name],
                                "new": current_entries[key][field_name],
                            }
                        )

    old_views = {view["view_id"]: view for view in old.get("views", [])}
    current_views = {view["view_id"]: view for view in current.get("views", [])}
    for view_id in sorted(set(old_views) | set(current_views), key=utf8_key):
        old_view = old_views.get(view_id)
        current_view = current_views.get(view_id)
        if old_view is None or current_view is None:
            changes.append(
                {
                    "scope": "view",
                    "view_id": view_id,
                    "field": "presence",
                    "old": old_view is not None,
                    "new": current_view is not None,
                }
            )
            continue
        for field_name in ("view_version", "segment_set_version", "snapshot_digest"):
            if old_view.get(field_name) != current_view.get(field_name):
                changes.append(
                    {
                        "scope": "view",
                        "view_id": view_id,
                        "field": field_name,
                        "old": old_view.get(field_name),
                        "new": current_view.get(field_name),
                    }
                )
    return {"changed": bool(changes), "changes": changes}


def _replay_error(
    message: str,
    *,
    segment_id: str | None = None,
    entry_key: str | None = None,
    field: str | None = None,
    expected: Any = None,
    actual: Any = None,
    reason: str | None = None,
) -> DiagnosticError:
    details: dict[str, Any] = {"expected": expected, "actual": actual}
    if reason:
        details["reason"] = reason
    return DiagnosticError(
        "E_REPLAY",
        message,
        segment_id=segment_id,
        entry_key=entry_key,
        field=field,
        details=details,
    )


def replay_operations(
    snapshot: dict[str, Any],
    operations: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Replay journal rows after the snapshot cursor without reading data files."""

    segments: dict[str, dict[str, Any]] = {}
    for proof in snapshot["segments"]:
        entries = _entry_map(proof)
        segments[proof["segment_id"]] = {
            "segment_id": proof["segment_id"],
            "capacity_bytes": int(proof["capacity_bytes"]),
            "segment_version": int(proof["segment_version"]),
            "entries": entries,
        }
    global_version = int(snapshot["segment_set_version"])
    cursor = int(snapshot["completed_cursor"])
    code_results: list[dict[str, Any]] = []

    for operation in sorted(operations, key=lambda row: row["operation_seq"]):
        kind = operation["kind"]
        status = operation["status"]
        segment_id = operation.get("segment_id") or ""
        entry_key = operation.get("entry_key") or ""
        segment = segments.get(segment_id)
        computed_code = "OK"

        if status == "unknown":
            computed_code = "E_UNKNOWN"
            code_results.append(
                {
                    "operation_seq": operation["operation_seq"],
                    "operation_id": operation["operation_id"],
                    "status": status,
                    "recorded_code": operation["result_code"],
                    "computed_code": computed_code,
                }
            )
            continue
        if status not in {"committed", "rejected"}:
            raise _replay_error(
                "journal contains a non-terminal operation after recovery",
                segment_id=segment_id,
                entry_key=entry_key,
                field="operations.status",
                expected="committed|rejected|unknown",
                actual=status,
            )
        if segment is None:
            computed_code = "E_NOT_FOUND"
        elif kind == "put":
            if entry_key in segment["entries"]:
                computed_code = "E_SPEC"
            else:
                used = sum(item["size_bytes"] for item in segment["entries"].values())
                available = segment["capacity_bytes"] - used
                size_bytes = operation.get("size_bytes")
                if not isinstance(size_bytes, int):
                    raise _replay_error(
                        "journal put has no integer size",
                        segment_id=segment_id,
                        entry_key=entry_key,
                        field="operations.size_bytes",
                        expected="integer",
                        actual=size_bytes,
                    )
                computed_code = "E_CAPACITY" if size_bytes > available else "OK"
                if status == "committed" and computed_code == "OK":
                    segment["entries"][entry_key] = {
                        "size_bytes": size_bytes,
                        "digest": operation.get("expected_digest"),
                    }
                    segment["segment_version"] += 1
                    global_version += 1
        elif kind == "delete":
            computed_code = "OK" if entry_key in segment["entries"] else "E_NOT_FOUND"
            if status == "committed" and computed_code == "OK":
                del segment["entries"][entry_key]
                segment["segment_version"] += 1
                global_version += 1
        else:
            raise _replay_error(
                "journal contains an unsupported operation kind",
                segment_id=segment_id,
                entry_key=entry_key,
                field="operations.kind",
                expected="put|delete",
                actual=kind,
            )

        if computed_code != operation["result_code"]:
            raise _replay_error(
                "replayed business code differs from the ledger",
                segment_id=segment_id or None,
                entry_key=entry_key or None,
                field="result_code",
                expected=computed_code,
                actual=operation["result_code"],
                reason="business code comparison",
            )
        if status == "committed":
            expected_cursor = cursor + 1
            if operation.get("committed_cursor") != expected_cursor:
                raise _replay_error(
                    "committed cursor is not contiguous",
                    segment_id=segment_id,
                    entry_key=entry_key,
                    field="committed_cursor",
                    expected=expected_cursor,
                    actual=operation.get("committed_cursor"),
                )
            cursor = expected_cursor
        code_results.append(
            {
                "operation_seq": operation["operation_seq"],
                "operation_id": operation["operation_id"],
                "status": status,
                "recorded_code": operation["result_code"],
                "computed_code": computed_code,
            }
        )

    proofs = []
    for segment_id in sorted(segments, key=utf8_key):
        segment = segments[segment_id]
        entries = [
            {
                "entry_key": key,
                "size_bytes": value["size_bytes"],
                "digest": value["digest"],
            }
            for key, value in sorted(
                segment["entries"].items(), key=lambda item: utf8_key(item[0])
            )
        ]
        proofs.append(
            build_segment_proof(
                segment_id=segment_id,
                capacity_bytes=segment["capacity_bytes"],
                segment_set_version=global_version,
                segment_version=segment["segment_version"],
                entries=entries,
            )
        )
    return {
        "segments": proofs,
        "segment_set_version": global_version,
        "completed_cursor": cursor,
        "operation_codes": code_results,
    }


def _first_state_mismatch(
    replay: dict[str, Any], live: dict[str, Any]
) -> None:
    expected_segments = {proof["segment_id"]: proof for proof in replay["segments"]}
    live_segments = {proof["segment_id"]: proof for proof in live["segments"]}
    for segment_id in sorted(
        set(expected_segments) | set(live_segments), key=utf8_key
    ):
        expected = expected_segments.get(segment_id)
        actual = live_segments.get(segment_id)
        if expected is None or actual is None:
            raise _replay_error(
                "replayed segment presence differs from the ledger",
                segment_id=segment_id,
                field="segments",
                expected=expected is not None,
                actual=actual is not None,
            )
        for field_name in (
            "capacity_bytes",
            "used_bytes",
            "remaining_bytes",
            "entry_count",
            "segment_version",
        ):
            if expected.get(field_name) != actual.get(field_name):
                raise _replay_error(
                    "replayed segment field differs from the ledger",
                    segment_id=segment_id,
                    field=field_name,
                    expected=expected.get(field_name),
                    actual=actual.get(field_name),
                )
        expected_entries = _entry_map(expected)
        actual_entries = _entry_map(actual)
        for key in sorted(
            set(expected_entries) | set(actual_entries), key=utf8_key
        ):
            if key not in expected_entries:
                raise _replay_error(
                    "ledger contains an entry absent from replay",
                    segment_id=segment_id,
                    entry_key=key,
                    field="entries.key",
                    expected=None,
                    actual=actual_entries[key]["digest"],
                )
            if key not in actual_entries:
                raise _replay_error(
                    "replay contains an entry absent from the ledger",
                    segment_id=segment_id,
                    entry_key=key,
                    field="entries.key",
                    expected=expected_entries[key]["digest"],
                    actual=None,
                )
            for field_name in ("size_bytes", "digest"):
                if expected_entries[key][field_name] != actual_entries[key][field_name]:
                    raise _replay_error(
                        "replayed entry field differs from the ledger",
                        segment_id=segment_id,
                        entry_key=key,
                        field=f"entries.{field_name}",
                        expected=expected_entries[key][field_name],
                        actual=actual_entries[key][field_name],
                    )
        if expected.get("content_digest") != actual.get("content_digest"):
            raise _replay_error(
                "replayed capacity proof digest differs from the ledger",
                segment_id=segment_id,
                field="proof.content_digest",
                expected=expected.get("content_digest"),
                actual=actual.get("content_digest"),
            )
    if replay["segment_set_version"] != live["segment_set_version"]:
        raise _replay_error(
            "replayed segment set version differs from the ledger",
            field="segment_set_version",
            expected=replay["segment_set_version"],
            actual=live["segment_set_version"],
        )
    if replay["completed_cursor"] != live["completed_cursor"]:
        raise _replay_error(
            "replayed completed operation cursor differs from the ledger",
            field="completed_cursor",
            expected=replay["completed_cursor"],
            actual=live["completed_cursor"],
        )


def expected_replayed_views(
    snapshot: dict[str, Any], replay: dict[str, Any]
) -> list[dict[str, Any]]:
    replay_segments = {proof["segment_id"]: proof for proof in replay["segments"]}
    expected_views = []
    for view in sorted(snapshot["views"], key=lambda item: utf8_key(item["view_id"])):
        old_snapshot = view["snapshot"]
        current_segments = []
        for old_segment in old_snapshot.get("segments", []):
            segment = replay_segments.get(old_segment["segment_id"])
            if segment is None:
                raise _replay_error(
                    "read view references a segment absent from replay",
                    segment_id=old_segment["segment_id"],
                    field="views.segments",
                )
            current_segments.append(
                {
                    "segment_id": segment["segment_id"],
                    "segment_version": segment["segment_version"],
                    "entries": [
                        {"key": entry["key"], "digest": entry["digest"]}
                        for entry in segment["entries"]
                    ],
                }
            )
        current_at_old_version = build_view_snapshot(
            view_id=view["view_id"],
            view_version=view["view_version"],
            segment_set_version=replay["segment_set_version"],
            segments=current_segments,
        )
        if view_is_stale(old_snapshot, current_at_old_version):
            rebuilt = build_view_snapshot(
                view_id=view["view_id"],
                view_version=view["view_version"] + 1,
                segment_set_version=replay["segment_set_version"],
                segments=current_segments,
            )
            differences = compute_view_diff(old_snapshot, rebuilt)
            expected_views.append(
                {
                    "view_id": view["view_id"],
                    "view_version": rebuilt["view_version"],
                    "snapshot": rebuilt,
                    "last_differences": differences,
                    "rebuilt": True,
                }
            )
        else:
            expected_views.append(
                {
                    "view_id": view["view_id"],
                    "view_version": old_snapshot["view_version"],
                    "snapshot": old_snapshot,
                    "last_differences": view.get("last_differences", {}),
                    "rebuilt": False,
                }
            )
    return expected_views


def compare_replayed_views(
    expected_views: list[dict[str, Any]], live_views: list[dict[str, Any]]
) -> None:
    expected_by_id = {view["view_id"]: view for view in expected_views}
    live_by_id = {view["view_id"]: view for view in live_views}
    for view_id in sorted(set(expected_by_id) | set(live_by_id), key=utf8_key):
        expected = expected_by_id.get(view_id)
        actual = live_by_id.get(view_id)
        if expected is None or actual is None:
            raise _replay_error(
                "replayed read-view presence differs from the ledger",
                field=f"read_views[{view_id}].presence",
                expected=expected is not None,
                actual=actual is not None,
            )
        if expected["view_version"] != actual["view_version"]:
            raise _replay_error(
                "replayed read-view version differs from the ledger",
                field=f"read_views[{view_id}].view_version",
                expected=expected["view_version"],
                actual=actual["view_version"],
            )
        expected_digest = expected["snapshot"]["content_digest"]
        actual_digest = actual["snapshot"]["content_digest"]
        if expected_digest != actual_digest:
            raise _replay_error(
                "replayed read-view snapshot differs from the ledger",
                field=f"read_views[{view_id}].snapshot_digest",
                expected=expected_digest,
                actual=actual_digest,
            )
        if canonical_text(expected["last_differences"]) != canonical_text(
            actual.get("last_differences", {})
        ):
            raise _replay_error(
                "replayed read-view difference report differs from the ledger",
                field=f"read_views[{view_id}].last_differences",
                expected=expected["last_differences"],
                actual=actual.get("last_differences", {}),
            )


def compare_replayed_state(replay: dict[str, Any], live: dict[str, Any]) -> None:
    _first_state_mismatch(replay, live)

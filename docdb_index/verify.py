"""Independent structural integrity checks for segments and stale ledgers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import canonical_json, loads_strict
from .errors import BoundError, LedgerError, SegmentError
from .models import segment_content_blob, segment_digest
from .storage import (
    ledger_rows,
    load_collection_specs,
    load_segments,
    log_event,
    open_session,
    read_meta,
    read_segment_entries,
    table_exists,
)


def _segment_error(
    code: str,
    message: str,
    *,
    collection: str,
    index_path: str,
    segment_id: str,
    field: str,
    expected: Any,
    actual: Any,
) -> Exception:
    cls = {
        "E_SEGMENT": SegmentError,
        "E_BOUND": BoundError,
    }[code]
    return cls(
        message,
        collection=collection,
        field_path=index_path,
        segment_id=segment_id,
        field=field,
        expected=expected,
        actual=actual,
    )


def verify_structure(session_dir: str | Path) -> dict[str, Any]:
    conn = open_session(session_dir)
    try:
        specs = load_collection_specs(conn)
        checked_segments = 0
        checked_entries = 0
        for spec in specs:
            for index in spec.indexes:
                segments = load_segments(conn, spec.name, index.path, with_entries=False)
                previous_high: tuple[Any, ...] | None = None
                for segment in sorted(segments, key=lambda item: item.segment_id):
                    checked_segments += 1
                    if not table_exists(conn, segment.table_name):
                        raise _segment_error(
                            "E_SEGMENT",
                            "segment table is missing",
                            collection=spec.name,
                            index_path=index.path,
                            segment_id=segment.segment_id,
                            field="table_name",
                            expected=segment.table_name,
                            actual=None,
                        )
                    segment.entries = read_segment_entries(conn, segment.table_name)
                    checked_entries += len(segment.entries)
                    recomputed_blob = segment_content_blob(
                        segment.segment_id,
                        segment.version,
                        segment.entries,
                    )
                    recomputed_digest = segment_digest(recomputed_blob)
                    if recomputed_digest != segment.digest:
                        raise _segment_error(
                            "E_SEGMENT",
                            "segment digest does not cover its entry sequence and version",
                            collection=spec.name,
                            index_path=index.path,
                            segment_id=segment.segment_id,
                            field="digest",
                            expected=recomputed_digest,
                            actual=segment.digest,
                        )
                    if recomputed_blob != segment.content_blob:
                        raise _segment_error(
                            "E_SEGMENT",
                            "segment content blob differs from its authoritative entries",
                            collection=spec.name,
                            index_path=index.path,
                            segment_id=segment.segment_id,
                            field="content_blob",
                            expected=recomputed_blob.hex(),
                            actual=segment.content_blob.hex(),
                        )
                    if segment.entry_count > index.segment_capacity:
                        raise _segment_error(
                            "E_BOUND",
                            "segment exceeds its fixed capacity",
                            collection=spec.name,
                            index_path=index.path,
                            segment_id=segment.segment_id,
                            field="capacity",
                            expected=index.segment_capacity,
                            actual=segment.entry_count,
                        )
                    if segment.entry_count != len(segment.entries):
                        raise _segment_error(
                            "E_BOUND",
                            "segment entry count does not match its table",
                            collection=spec.name,
                            index_path=index.path,
                            segment_id=segment.segment_id,
                            field="entry_count",
                            expected=len(segment.entries),
                            actual=segment.entry_count,
                        )
                    if not segment.entries:
                        raise _segment_error(
                            "E_BOUND",
                            "segment is empty",
                            collection=spec.name,
                            index_path=index.path,
                            segment_id=segment.segment_id,
                            field="entry_count",
                            expected=">0",
                            actual=0,
                        )
                    sorted_entries = sorted(segment.entries, key=lambda item: item.sort_key)
                    if [item.identity() for item in sorted_entries] != [
                        item.identity() for item in segment.entries
                    ]:
                        raise _segment_error(
                            "E_BOUND",
                            "segment table is not strictly ordered by key then primary key",
                            collection=spec.name,
                            index_path=index.path,
                            segment_id=segment.segment_id,
                            field="entry_order",
                            expected=[item.identity() for item in sorted_entries],
                            actual=[item.identity() for item in segment.entries],
                        )
                    low = segment.entries[0]
                    high = segment.entries[-1]
                    if (
                        segment.low_value_json != low.value_json
                        or segment.low_pk_json != low.pk_json
                    ):
                        raise _segment_error(
                            "E_BOUND",
                            "segment low boundary does not match its first entry",
                            collection=spec.name,
                            index_path=index.path,
                            segment_id=segment.segment_id,
                            field="low_boundary",
                            expected=[low.value_json, low.pk_json],
                            actual=[segment.low_value_json, segment.low_pk_json],
                        )
                    if (
                        segment.high_value_json != high.value_json
                        or segment.high_pk_json != high.pk_json
                    ):
                        raise _segment_error(
                            "E_BOUND",
                            "segment high boundary does not match its last entry",
                            collection=spec.name,
                            index_path=index.path,
                            segment_id=segment.segment_id,
                            field="high_boundary",
                            expected=[high.value_json, high.pk_json],
                            actual=[segment.high_value_json, segment.high_pk_json],
                        )
                previous_high = None
                for segment in sorted(segments, key=lambda item: item.low_key):
                    if previous_high is not None and not previous_high < segment.low_key:
                        raise _segment_error(
                            "E_BOUND",
                            "segment boundaries overlap or are out of order",
                            collection=spec.name,
                            index_path=index.path,
                            segment_id=segment.segment_id,
                            field="segment_order",
                            expected="low_key greater than previous high_key",
                            actual={
                                "previous_high": previous_high,
                                "current_low": segment.low_key,
                            },
                        )
                    previous_high = segment.high_key

        actual_ledger = ledger_rows(conn, "stale_ledger")
        expected_ledger = ledger_rows(conn, "stale_ledger_expected")
        actual_map = {
            (
                row["collection"],
                row["index_path"],
                row["pk_json"],
                row["stale_key_json"],
            ): row
            for row in actual_ledger
        }
        expected_map = {
            (
                row["collection"],
                row["index_path"],
                row["pk_json"],
                row["stale_key_json"],
            ): row
            for row in expected_ledger
        }
        for identity in sorted(set(expected_map) | set(actual_map)):
            expected = expected_map.get(identity)
            actual = actual_map.get(identity)
            if canonical_json(expected) != canonical_json(actual):
                collection, index_path, pk_json, stale_value_json = identity
                raise LedgerError(
                    "stale-key ledger does not match its replay expectation",
                    collection=collection,
                    field_path=index_path,
                    primary_key=loads_strict(pk_json),
                    stale_key=loads_strict(stale_value_json),
                    field="stale_ledger",
                    expected=expected,
                    actual=actual,
                )
        for spec in specs:
            for index in spec.indexes:
                expected_digest = read_meta(
                    conn,
                    f"ledger_digest:{spec.name}:{index.path}",
                )
                if expected_digest is None:
                    continue
                rows = ledger_rows(
                    conn,
                    collection_name=spec.name,
                    index_path=index.path,
                )
                from .canonical import digest_json

                actual_digest = digest_json(rows)
                if actual_digest != expected_digest:
                    raise LedgerError(
                        "stale-key ledger digest differs from its expected digest",
                        collection=spec.name,
                        field_path=index.path,
                        field="ledger_digest",
                        expected=expected_digest,
                        actual=actual_digest,
                    )
        return {
            "ok": True,
            "collections": len(specs),
            "segments_checked": checked_segments,
            "entries_checked": checked_entries,
            "ledger_entries_checked": len(actual_ledger),
        }
    finally:
        conn.close()


def verify_session(
    session_dir: str | Path,
    *,
    include_replay: bool = True,
) -> dict[str, Any]:
    try:
        structure = verify_structure(session_dir)
        result: dict[str, Any] = {"ok": True, "structure": structure}
        if include_replay:
            from .replay import replay_validate

            result["replay"] = replay_validate(session_dir)
        log_event(
            session_dir,
            level="info",
            code="OK",
            message="session verification passed",
            context={"include_replay": include_replay},
        )
        return result
    except Exception as exc:
        code = getattr(exc, "code", "E_STATE")
        details = getattr(exc, "details", {})
        log_event(
            session_dir,
            level="error",
            code=code,
            message=str(exc),
            context=details,
        )
        try:
            setattr(exc, "_docdb_logged", True)
        except Exception:
            pass
        raise

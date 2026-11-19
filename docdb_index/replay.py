"""Replay the append-only audit journal into a fresh session and compare state."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .canonical import canonical_json, loads_strict
from .engine import apply_changes, initialize_session
from .errors import ReplayError
from .models import (
    CollectionSpec,
    IndexSpec,
    segment_content_blob,
    segment_digest,
)
from .storage import (
    ledger_rows,
    load_segments,
    open_session,
    state_manifest,
)


def _core_state(conn) -> dict[str, Any]:
    state = state_manifest(conn)
    records_by_key: dict[tuple[str, str, str], Any] = {}
    index_keys = {
        (segment["collection"], segment["index_path"])
        for segment in state["segments"]
    }
    for collection_name, index_path in sorted(index_keys):
        for record in load_segments(
            conn,
            collection_name,
            index_path,
            with_entries=True,
        ):
            records_by_key[(collection_name, index_path, record.segment_id)] = record
    computed_segments: list[dict[str, Any]] = []
    for segment in state["segments"]:
        record = records_by_key[
            (segment["collection"], segment["index_path"], segment["segment_id"])
        ]
        computed_blob = segment_content_blob(
            record.segment_id,
            record.version,
            record.entries,
        )
        computed_digest = segment_digest(computed_blob)
        low = record.entries[0] if record.entries else None
        high = record.entries[-1] if record.entries else None
        computed_segments.append(
            {
                **segment,
                "digest": computed_digest,
                "metadata_digest": segment["digest"],
                "entry_count": len(record.entries),
                "low_key": low.value if low is not None else None,
                "low_pk": low.pk if low is not None else None,
                "high_key": high.value if high is not None else None,
                "high_pk": high.pk if high is not None else None,
            }
        )
    return {
        "collection_versions": state["collection_versions"],
        "index_versions": state["index_versions"],
        "segments": computed_segments,
        "segment_digests": {
            f"{item['collection']}|{item['index_path']}|{item['segment_id']}": item["digest"]
            for item in computed_segments
        },
        "ledger_digest": state["ledger_digest"],
        "ledger": ledger_rows(conn),
    }


def _core_from_json(state: dict[str, Any], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "collection_versions": state.get("collection_versions", {}),
        "index_versions": state.get("index_versions", {}),
        "segments": state.get("segments", []),
        "segment_digests": state.get("segment_digests", {}),
        "ledger_digest": state.get("ledger_digest"),
        "ledger": ledger,
    }


def _first_segment_difference(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    batch_id: str,
    context: str,
) -> ReplayError | None:
    expected_segments = {
        f"{item['collection']}|{item['index_path']}|{item['segment_id']}": item
        for item in expected.get("segments", [])
    }
    actual_segments = {
        f"{item['collection']}|{item['index_path']}|{item['segment_id']}": item
        for item in actual.get("segments", [])
    }
    for key in sorted(set(expected_segments) | set(actual_segments)):
        expected_item = expected_segments.get(key)
        actual_item = actual_segments.get(key)
        if expected_item is None or actual_item is None:
            return ReplayError(
                "replay found a segment set difference",
                batch_id=batch_id,
                context=context,
                segment_id=key.rsplit("|", 1)[-1],
                segment_key=key,
                field="segment_presence",
                expected=expected_item,
                actual=actual_item,
            )
        for field in (
            "digest",
            "version",
            "entry_count",
            "low_key",
            "low_pk",
            "high_key",
            "high_pk",
        ):
            if expected_item.get(field) != actual_item.get(field):
                return ReplayError(
                    "replay found a segment metadata difference",
                    batch_id=batch_id,
                    context=context,
                    collection=expected_item["collection"],
                    field_path=expected_item["index_path"],
                    segment_id=expected_item["segment_id"],
                    segment_key=key,
                    field=field,
                    expected=expected_item.get(field),
                    actual=actual_item.get(field),
                )
    if expected.get("collection_versions") != actual.get("collection_versions"):
        return ReplayError(
            "replay found a collection-version difference",
            batch_id=batch_id,
            context=context,
            segment_id="<collection>",
            field="collection_versions",
            expected=expected.get("collection_versions"),
            actual=actual.get("collection_versions"),
        )
    if expected.get("index_versions") != actual.get("index_versions"):
        return ReplayError(
            "replay found an index-version difference",
            batch_id=batch_id,
            context=context,
            segment_id="<index>",
            field="index_versions",
            expected=expected.get("index_versions"),
            actual=actual.get("index_versions"),
        )
    expected_ledger = {
        (
            row["collection"],
            row["index_path"],
            row["pk_json"],
            row["stale_key_json"],
            row["reason"],
            row["original_segment"],
            row["discovered_collection_version"],
        )
        for row in expected.get("ledger", [])
    }
    actual_ledger = {
        (
            row["collection"],
            row["index_path"],
            row["pk_json"],
            row["stale_key_json"],
            row["reason"],
            row["original_segment"],
            row["discovered_collection_version"],
        )
        for row in actual.get("ledger", [])
    }
    if expected_ledger != actual_ledger:
        return ReplayError(
            "replay found a stale-key ledger difference",
            batch_id=batch_id,
            context=context,
            segment_id="<ledger>",
            field="stale_keys",
            expected=sorted(expected_ledger),
            actual=sorted(actual_ledger),
        )
    return None


def _load_baseline(conn) -> list[CollectionSpec]:
    specs: list[CollectionSpec] = []
    for row in conn.execute(
        "SELECT collection_name, declaration_json, documents_json "
        "FROM audit_baseline ORDER BY collection_name"
    ):
        declaration = loads_strict(str(row["declaration_json"]))
        documents = loads_strict(str(row["documents_json"]))
        specs.append(
            CollectionSpec(
                name=str(declaration["name"]),
                version=int(declaration["version"]),
                primary_key=str(declaration["primary_key"]),
                documents=tuple(documents),
                indexes=tuple(
                    IndexSpec(
                        path=str(index["path"]),
                        segment_capacity=int(index["segment_capacity"]),
                        allow_null=bool(index["allow_null"]),
                    )
                    for index in declaration["indexes"]
                ),
            )
        )
    return specs


def replay_validate(session_dir: str | Path) -> dict[str, Any]:
    actual_conn = open_session(session_dir)
    try:
        baseline = _load_baseline(actual_conn)
        if not baseline:
            raise ReplayError(
                "session has no audit baseline to replay",
                segment_id="<audit>",
                field="audit_baseline",
            )
        batches = [
            {
                "seq": int(row["seq"]),
                "batch_id": str(row["batch_id"]),
                "collection_name": str(row["collection_name"]),
                "changes": loads_strict(str(row["changes_json"])),
                "options": loads_strict(str(row["options_json"])),
                "version_before": int(row["version_before"]),
                "version_after": int(row["version_after"]),
                "before": loads_strict(str(row["before_json"])),
                "after": loads_strict(str(row["after_json"])),
                "affected": loads_strict(str(row["affected_json"])),
                "ledger_after": loads_strict(str(row["ledger_after_json"])),
            }
            for row in actual_conn.execute(
                "SELECT seq, batch_id, collection_name, changes_json, options_json, "
                "version_before, version_after, before_json, after_json, affected_json, "
                "ledger_after_json FROM audit_batches ORDER BY seq"
            )
        ]
        actual_final = _core_state(actual_conn)
    finally:
        actual_conn.close()

    with tempfile.TemporaryDirectory(prefix="docdb-replay-") as temp_dir:
        replay_session = Path(temp_dir) / "session"
        initialize_session(replay_session, baseline)
        batches_checked = 0
        unaffected_checked = 0
        for batch in batches:
            conn = open_session(replay_session)
            try:
                replay_before = _core_state(conn)
            finally:
                conn.close()
            # before_json does not embed ledger rows; compare segment/version state.
            expected_before = {
                "collection_versions": batch["before"].get("collection_versions", {}),
                "index_versions": batch["before"].get("index_versions", {}),
                "segments": batch["before"].get("segments", []),
            }
            actual_before = {
                "collection_versions": replay_before["collection_versions"],
                "index_versions": replay_before["index_versions"],
                "segments": replay_before["segments"],
            }
            error = _first_segment_difference(
                expected_before,
                actual_before,
                batch_id=batch["batch_id"],
                context="before_batch",
            )
            if error is not None:
                raise error

            for index_path, affected in sorted(batch["affected"].items()):
                affected_ids = set(affected.get("affected_segment_ids", []))
                before_map = {
                    item["segment_id"]: item
                    for item in batch["before"].get("segments", [])
                    if item["collection"] == batch["collection_name"]
                    and item["index_path"] == index_path
                }
                after_map = {
                    item["segment_id"]: item
                    for item in batch["after"].get("segments", [])
                    if item["collection"] == batch["collection_name"]
                    and item["index_path"] == index_path
                }
                for segment_id in sorted(set(before_map) - affected_ids):
                    unaffected_checked += 1
                    before_digest = before_map[segment_id]["digest"]
                    after_digest = after_map.get(segment_id, {}).get("digest")
                    if before_digest != after_digest:
                        raise ReplayError(
                            "an unaffected segment changed across the recorded batch",
                            batch_id=batch["batch_id"],
                            collection=batch["collection_name"],
                            field_path=index_path,
                            segment_id=segment_id,
                            field="unaffected_digest",
                            expected=before_digest,
                            actual=after_digest,
                        )

            options = batch["options"]
            apply_changes(
                replay_session,
                batch["collection_name"],
                batch["changes"],
                batch_id=batch["batch_id"],
                batch_size=options.get("batch_size"),
                retain_stale=bool(options.get("retain_stale", False)),
                target_version=batch["version_after"],
                record_audit=False,
            )
            conn = open_session(replay_session)
            try:
                replay_after = _core_state(conn)
            finally:
                conn.close()
            expected_after = _core_from_json(batch["after"], batch["ledger_after"])
            error = _first_segment_difference(
                expected_after,
                replay_after,
                batch_id=batch["batch_id"],
                context="after_batch",
            )
            if error is not None:
                raise error
            batches_checked += 1

        if not batches:
            conn = open_session(replay_session)
            try:
                replay_final = _core_state(conn)
            finally:
                conn.close()
        else:
            conn = open_session(replay_session)
            try:
                replay_final = _core_state(conn)
            finally:
                conn.close()
        error = _first_segment_difference(
            actual_final,
            replay_final,
            batch_id="<final>",
            context="current_vs_replay",
        )
        if error is not None:
            raise error

    return {
        "ok": True,
        "baseline_collections": len(baseline),
        "batches_checked": batches_checked,
        "unaffected_segments_checked": unaffected_checked,
        "segments_compared": len(actual_final["segments"]),
        "stale_keys_compared": len(actual_final["ledger"]),
    }

"""Core range-index engine: build, incremental maintenance and queries."""

from __future__ import annotations

import bisect
import math
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .canonical import (
    canonical_json,
    digest_json,
    is_number,
    loads_strict,
)
from .errors import DocumentError, StaleError, StateError
from .models import CollectionSpec, IndexEntry, IndexSpec, SegmentRecord
from .storage import (
    allocate_segment_ids,
    append_audit_batch,
    append_query_history,
    connect,
    delete_segment,
    document_key,
    ensure_session_dirs,
    get_collection,
    get_index,
    increment_isolation_stats,
    index_versions,
    init_schema,
    insert_stale_ledger,
    isolation_stats,
    ledger_digest,
    ledger_rows,
    load_collection_specs,
    load_documents,
    load_segments,
    open_session,
    query_history_digest,
    read_meta,
    segment_manifest,
    session_db_path,
    state_manifest,
    sync_expected_ledger,
    transaction,
    write_audit_baseline,
    write_meta,
    write_segment,
)
from .validation import field_path_parts, get_path, set_path, validate_document


def _deep_copy(value: Any) -> Any:
    return loads_strict(canonical_json(value))


def _entry_for_document(
    document: dict[str, Any],
    primary_key: str,
    index: IndexSpec,
) -> IndexEntry | None:
    pk, _ = validate_document_returning_pk(document, primary_key)
    present, value = get_path(document, index.path)
    if not present or value is None:
        return None
    if not is_number(value):
        raise DocumentError(
            "indexed field must be a finite number",
            primary_key=pk,
            field_path=index.path,
        )
    return IndexEntry(value=value, pk=pk)


def validate_document_returning_pk(document: dict[str, Any], primary_key: str) -> tuple[Any, str]:
    present, pk = get_path(document, primary_key)
    if not present:
        raise DocumentError(
            "primary key field is missing",
            primary_key=None,
            field_path=primary_key,
        )
    return pk, canonical_json(pk)


def entries_for_documents(
    documents: dict[str, dict[str, Any]],
    primary_key: str,
    index: IndexSpec,
) -> dict[tuple[str, str], IndexEntry]:
    result: dict[tuple[str, str], IndexEntry] = {}
    for doc in documents.values():
        entry = _entry_for_document(doc, primary_key, index)
        if entry is not None:
            if entry.identity() in result:
                raise StateError(
                    "duplicate index entry generated for one document",
                    primary_key=entry.pk,
                    field_path=index.path,
                )
            result[entry.identity()] = entry
    return result


def _chunks(entries: list[IndexEntry], capacity: int) -> list[list[IndexEntry]]:
    if not entries:
        return []
    if len(entries) <= capacity:
        return [entries]
    count = math.ceil(len(entries) / capacity)
    base, extra = divmod(len(entries), count)
    result: list[list[IndexEntry]] = []
    position = 0
    for number in range(count):
        size = base + (1 if number < extra else 0)
        result.append(entries[position : position + size])
        position += size
    return result


def _route_entry(entry: IndexEntry, segments: list[SegmentRecord]) -> str | None:
    if not segments:
        return None
    lows = [segment.low_key for segment in segments]
    position = bisect.bisect_right(lows, entry.sort_key) - 1
    if position < 0:
        position = 0
    return segments[position].segment_id


def _manifest_by_id(
    conn: sqlite3.Connection,
    collection_name: str,
    index_path: str,
) -> dict[str, dict[str, Any]]:
    return {
        item["segment_id"]: item
        for item in segment_manifest(conn, collection_name, index_path)
    }


def _cleanup_runtime_for_reset(session_dir: Path) -> None:
    for name in ("session.db-wal", "session.db-shm", "session.db"):
        path = session_dir / name
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    for name in ("snapshots", "rollback"):
        path = session_dir / name
        if path.exists():
            for child in sorted(path.rglob("*"), reverse=True):
                try:
                    child.chmod(0o700)
                except OSError:
                    pass
            shutil.rmtree(path, ignore_errors=True)
    log_path = session_dir / "logs" / "events.jsonl"
    try:
        log_path.unlink()
    except FileNotFoundError:
        pass


def initialize_session(
    session_dir: str | Path,
    specs: Iterable[CollectionSpec],
    *,
    reset: bool = False,
) -> dict[str, Any]:
    """Validate-first, temp-DB build followed by an atomic file replacement."""

    specs = tuple(specs)
    if not specs:
        raise DocumentError("at least one collection is required", field_path="$")
    session_path = ensure_session_dirs(session_dir)
    target = session_db_path(session_path)
    if target.exists() and not reset:
        raise StateError(
            "session already exists; use --reset to replace it",
            session=str(session_path),
        )

    temp = session_path / f".session.db.tmp-{os.getpid()}"
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(temp) + suffix)
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
    conn = connect(temp)
    try:
        init_schema(conn)
        with transaction(conn):
            for spec in specs:
                conn.execute(
                    "INSERT INTO collections(name, primary_key, collection_version) "
                    "VALUES(?,?,?)",
                    (spec.name, spec.primary_key, spec.version),
                )
                for index in spec.indexes:
                    conn.execute(
                        "INSERT INTO indexes("
                        "collection_name, path, segment_capacity, allow_null, "
                        "index_version, next_segment_number) VALUES(?,?,?,?,?,1)",
                        (
                            spec.name,
                            index.path,
                            index.segment_capacity,
                            int(index.allow_null),
                            spec.version,
                        ),
                    )
                for document in sorted(
                    spec.documents,
                    key=lambda item: canonical_json(
                        get_path(item, spec.primary_key)[1]
                    ),
                ):
                    pk, pk_json = document_key(document, spec.primary_key)
                    conn.execute(
                        "INSERT INTO documents(collection_name, pk_json, doc_json) "
                        "VALUES(?,?,?)",
                        (spec.name, pk_json, canonical_json(document)),
                    )

            write_audit_baseline(conn, specs)

            for spec in specs:
                documents = {
                    canonical_json(get_path(doc, spec.primary_key)[1]): doc
                    for doc in spec.documents
                }
                for index in spec.indexes:
                    expected = sorted(
                        entries_for_documents(
                            documents,
                            spec.primary_key,
                            index,
                        ).values(),
                        key=lambda entry: entry.sort_key,
                    )
                    chunks = _chunks(expected, index.segment_capacity)
                    ids = allocate_segment_ids(conn, spec.name, index.path, len(chunks))
                    for segment_id, chunk in zip(ids, chunks):
                        write_segment(
                            conn,
                            spec.name,
                            index.path,
                            segment_id,
                            1,
                            chunk,
                        )
            sync_expected_ledger(conn)
        conn.close()
        conn = None

        if reset:
            _cleanup_runtime_for_reset(session_path)
        os.replace(temp, target)
        # Remove any transient journal sidecars left from the temp file.
        for suffix in ("-wal", "-shm"):
            try:
                Path(str(temp) + suffix).unlink()
            except FileNotFoundError:
                pass
    finally:
        if conn is not None:
            conn.close()
        for path in (temp, Path(str(temp) + "-wal"), Path(str(temp) + "-shm")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    return manifest(session_path)


def _parse_index_value(
    document: dict[str, Any],
    primary_key: str,
    index: IndexSpec,
) -> IndexEntry | None:
    return _entry_for_document(document, primary_key, index)


def _apply_change(
    documents: dict[str, dict[str, Any]],
    collection: sqlite3.Row,
    index_specs: list[IndexSpec],
    change: dict[str, Any],
    position: int,
) -> tuple[str, Any]:
    if not isinstance(change, dict):
        raise DocumentError("change must be an object", field_path=f"changes[{position}]")
    task_collection = change.get("collection")
    if task_collection is not None and task_collection != collection["name"]:
        raise DocumentError(
            "change targets a different collection",
            field_path=f"changes[{position}].collection",
        )
    op = change.get("op", change.get("operation"))
    if not isinstance(op, str):
        raise DocumentError("change op is required", field_path=f"changes[{position}].op")
    op = op.lower()
    primary_key = str(collection["primary_key"])
    if op == "update":
        op = "replace" if isinstance(change.get("document", change.get("doc")), dict) else "set_path"
    elif op in {"modify", "patch"}:
        op = "set_path"

    def document_from_change() -> dict[str, Any]:
        document = change.get("document", change.get("doc"))
        if not isinstance(document, dict):
            raise DocumentError(
                "change document must be an object",
                field_path=f"changes[{position}].document",
            )
        return _deep_copy(document)

    def pk_from_change() -> tuple[Any, str]:
        raw = change.get("pk", change.get("primary_key"))
        if raw is None and primary_key in change:
            raw = change[primary_key]
        if raw is None and isinstance(change.get("document", change.get("doc")), dict):
            present, raw = get_path(change["document"] if "document" in change else change["doc"], primary_key)
            if not present:
                raw = None
        if raw is None:
            raise DocumentError(
                "change primary key is required",
                field_path=f"changes[{position}].pk",
            )
        return raw, canonical_json(raw)

    if op in {"insert", "add", "upsert"}:
        document = document_from_change()
        pk, pk_json = document_key(document, primary_key)
        if op in {"insert", "add"} and pk_json in documents:
            raise DocumentError(
                "insert target already exists",
                primary_key=pk,
                field_path=primary_key,
            )
        existed = pk_json in documents
        validate_document(document, primary_key=primary_key, indexes=tuple(index_specs))
        documents[pk_json] = document
        return ("upsert" if existed else "insert"), pk

    if op in {"delete", "remove"}:
        pk, pk_json = pk_from_change()
        if pk_json not in documents:
            raise DocumentError(
                "delete target does not exist",
                primary_key=pk,
                field_path=primary_key,
            )
        del documents[pk_json]
        return "delete", pk

    if op in {"replace", "update_document"}:
        document = document_from_change()
        pk, pk_json = document_key(document, primary_key)
        if pk_json not in documents:
            raise DocumentError(
                "update target does not exist",
                primary_key=pk,
                field_path=primary_key,
            )
        validate_document(document, primary_key=primary_key, indexes=tuple(index_specs))
        documents[pk_json] = document
        return "replace", pk

    if op in {"set_path", "set_field", "set"}:
        pk, pk_json = pk_from_change()
        if pk_json not in documents:
            raise DocumentError(
                "field update target does not exist",
                primary_key=pk,
                field_path=primary_key,
            )
        document = _deep_copy(documents[pk_json])
        fields = change.get("fields", change.get("set"))
        updates: list[tuple[str, Any]] = []
        if isinstance(fields, dict):
            updates.extend((str(path), value) for path, value in fields.items())
        path = change.get("path", change.get("field", change.get("field_path")))
        if path is not None:
            field_path_parts(path, primary_key=pk)
            updates.append((str(path), change.get("value")))
        if not updates:
            raise DocumentError(
                "field update requires a path/value or fields object",
                primary_key=pk,
                field_path=f"changes[{position}].path",
            )
        for update_path, update_value in updates:
            field_path_parts(update_path, primary_key=pk)
            set_path(document, update_path, update_value)
        new_pk, new_pk_json = document_key(document, primary_key)
        if new_pk_json != pk_json:
            raise DocumentError(
                "field update may not change the primary key",
                primary_key=pk,
                field_path=primary_key,
            )
        validate_document(document, primary_key=primary_key, indexes=tuple(index_specs))
        documents[pk_json] = document
        return "set_path", pk

    if op == "noop":
        pk, _ = pk_from_change()
        return "noop", pk

    raise DocumentError(
        "unsupported change op",
        field_path=f"changes[{position}].op",
        op=op,
    )


def _materialize_changes(
    documents: dict[str, dict[str, Any]],
    collection: sqlite3.Row,
    index_specs: list[IndexSpec],
    changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for position, change in enumerate(changes):
        before_pks = set(documents)
        before_docs = {key: _deep_copy(value) for key, value in documents.items()}
        kind, pk = _apply_change(
            documents,
            collection,
            index_specs,
            change,
            position,
        )
        trace.append(
            {
                "position": position,
                "op": kind,
                "primary_key": pk,
                "resulting_document_count": len(documents),
                "changed_document": set(documents) != before_pks
                or any(
                    canonical_json(documents[key]) != canonical_json(before_docs[key])
                    for key in set(documents) & set(before_docs)
                ),
            }
        )
    return trace


def _segment_content_identity(record: SegmentRecord) -> dict[str, Any]:
    return {
        "version": record.version,
        "entry_count": record.entry_count,
        "low_value_json": record.low_value_json,
        "low_pk_json": record.low_pk_json,
        "high_value_json": record.high_value_json,
        "high_pk_json": record.high_pk_json,
        "digest": record.digest,
        "content_blob": record.content_blob.hex(),
    }


def _rebuild_index(
    conn: sqlite3.Connection,
    collection: sqlite3.Row,
    index_row: sqlite3.Row,
    documents: dict[str, dict[str, Any]],
    version_after: int,
    *,
    retain_stale: bool,
) -> dict[str, Any]:
    collection_name = str(collection["name"])
    primary_key = str(collection["primary_key"])
    index_path = str(index_row["path"])
    index_spec = IndexSpec(
        path=index_path,
        segment_capacity=int(index_row["segment_capacity"]),
        allow_null=bool(index_row["allow_null"]),
    )
    before_segments = load_segments(conn, collection_name, index_path, with_entries=True)
    before_by_id = {segment.segment_id: segment for segment in before_segments}
    before_identity = {
        segment.segment_id: _segment_content_identity(segment)
        for segment in before_segments
    }
    expected = entries_for_documents(documents, primary_key, index_spec)
    expected_ids = set(expected)
    final_pk_jsons = set(documents)

    physical_entry_segments: dict[tuple[str, str], list[str]] = {}
    for segment in before_segments:
        for entry in segment.entries:
            physical_entry_segments.setdefault(entry.identity(), []).append(segment.segment_id)

    stale_entries: list[tuple[str, IndexEntry, str]] = []
    stale_by_segment: dict[str, list[IndexEntry]] = {}
    for segment in before_segments:
        for entry in segment.entries:
            if entry.identity() in expected_ids:
                continue
            reason = (
                "document_delete"
                if entry.pk_json not in final_pk_jsons
                else "key_migration"
            )
            stale_entries.append((segment.segment_id, entry, reason))
            stale_by_segment.setdefault(segment.segment_id, []).append(entry)

    existing_ledger = {
        (row["pk_json"], row["stale_key_json"])
        for row in ledger_rows(conn, collection_name=collection_name, index_path=index_path)
    }
    for segment_id, entry, reason in stale_entries:
        insert_stale_ledger(
            conn,
            collection_name=collection_name,
            index_path=index_path,
            pk_json=entry.pk_json,
            stale_value_json=entry.value_json,
            reason=reason,
            original_segment=segment_id,
            collection_version=version_after,
        )

    physical_after: dict[str, list[IndexEntry]] = {
        segment.segment_id: list(segment.entries) for segment in before_segments
    }
    if not retain_stale:
        for segment in before_segments:
            if segment.segment_id in stale_by_segment:
                stale_ids = {entry.identity() for entry in stale_by_segment[segment.segment_id]}
                physical_after[segment.segment_id] = [
                    entry for entry in segment.entries if entry.identity() not in stale_ids
                ]

    physical_ids = {
        entry.identity()
        for entries in physical_after.values()
        for entry in entries
    }
    missing = [entry for identity, entry in expected.items() if identity not in physical_ids]
    missing.sort(key=lambda entry: entry.sort_key)

    new_by_segment: dict[str, list[IndexEntry]] = {}
    if before_segments:
        for entry in missing:
            target = _route_entry(entry, before_segments)
            if target is None:
                raise StateError("cannot route an entry without a target segment")
            new_by_segment.setdefault(target, []).append(entry)
    else:
        new_by_segment[""] = missing

    affected_ids: set[str] = set()
    if not retain_stale:
        affected_ids.update(stale_by_segment)
    affected_ids.update(
        target for target in new_by_segment if target in before_by_id
    )
    affected_reports: list[dict[str, Any]] = []

    for segment in before_segments:
        if segment.segment_id not in affected_ids:
            continue
        base = physical_after.get(segment.segment_id, [])
        additions = new_by_segment.get(segment.segment_id, [])
        combined = sorted(base + additions, key=lambda entry: entry.sort_key)
        old_manifest = before_identity[segment.segment_id]
        if not combined:
            delete_segment(
                conn,
                collection_name,
                index_path,
                segment.segment_id,
                segment.table_name,
            )
            affected_reports.append(
                {
                    "segment_id": segment.segment_id,
                    "action": "removed",
                    "old_version": segment.version,
                    "new_version": None,
                    "old_digest": segment.digest,
                    "new_digest": None,
                }
            )
            continue

        chunks = _chunks(combined, index_spec.segment_capacity)
        extra_ids = allocate_segment_ids(
            conn,
            collection_name,
            index_path,
            max(0, len(chunks) - 1),
        )
        produced: list[str] = []
        for chunk_number, chunk in enumerate(chunks):
            if chunk_number == 0:
                segment_id = segment.segment_id
            else:
                segment_id = extra_ids[chunk_number - 1]
            record = write_segment(
                conn,
                collection_name,
                index_path,
                segment_id,
                segment.version + 1,
                chunk,
            )
            produced.append(segment_id)
            affected_reports.append(
                {
                    "segment_id": segment_id,
                    "source_segment_id": segment.segment_id,
                    "action": "split" if len(chunks) > 1 else "replaced",
                    "old_version": segment.version,
                    "new_version": record.version,
                    "old_digest": segment.digest if chunk_number == 0 else None,
                    "new_digest": record.digest,
                    "produced_segments": produced if chunk_number == len(chunks) - 1 else None,
                }
            )
        # Make the produced list visible exactly once and keep records stable.
        for report in affected_reports:
            if (
                report.get("source_segment_id") == segment.segment_id
                and report.get("action") == "split"
            ):
                report["produced_segments"] = produced
        if len(old_manifest) == 0:  # defensive; prevents accidental use of stale vars
            raise StateError("internal segment identity failure")

    if not before_segments and new_by_segment.get("", []):
        chunks = _chunks(
            sorted(new_by_segment[""], key=lambda entry: entry.sort_key),
            index_spec.segment_capacity,
        )
        ids = allocate_segment_ids(conn, collection_name, index_path, len(chunks))
        for segment_id, chunk in zip(ids, chunks):
            record = write_segment(
                conn,
                collection_name,
                index_path,
                segment_id,
                1,
                chunk,
            )
            affected_reports.append(
                {
                    "segment_id": segment_id,
                    "action": "created",
                    "old_version": None,
                    "new_version": record.version,
                    "old_digest": None,
                    "new_digest": record.digest,
                }
            )

    # Verify the explicit non-interference contract for every unaffected segment.
    after_segments = load_segments(conn, collection_name, index_path, with_entries=False)
    after_by_id = {segment.segment_id: segment for segment in after_segments}
    unchanged: list[str] = []
    for segment in before_segments:
        if segment.segment_id in affected_ids:
            continue
        after = after_by_id.get(segment.segment_id)
        if after is None:
            raise StateError(
                "an unaffected segment disappeared",
                segment_id=segment.segment_id,
            )
        if _segment_content_identity(after) != before_identity[segment.segment_id]:
            raise StateError(
                "an unaffected segment changed",
                segment_id=segment.segment_id,
            )
        unchanged.append(segment.segment_id)

    after_sorted = load_segments(conn, collection_name, index_path, with_entries=True)
    for previous, current in zip(after_sorted, after_sorted[1:]):
        if not previous.high_key < current.low_key:
            raise StateError(
                "segment key ranges overlap after rebuild",
                previous_segment=previous.segment_id,
                next_segment=current.segment_id,
            )

    if not retain_stale:
        actual_ids = {
            entry.identity()
            for segment in after_sorted
            for entry in segment.entries
        }
        if actual_ids != expected_ids:
            missing_ids = sorted(expected_ids - actual_ids)
            extra_ids = sorted(actual_ids - expected_ids)
            raise StateError(
                "reclaimed index content does not equal the current document projection",
                field_path=index_path,
                missing=len(missing_ids),
                extra=len(extra_ids),
            )

    ledger_after = ledger_rows(
        conn,
        collection_name=collection_name,
        index_path=index_path,
    )
    ledger_added = [
        row
        for row in ledger_after
        if (row["pk_json"], row["stale_key_json"]) not in existing_ledger
    ]

    old_entry_by_pk: dict[str, IndexEntry] = {}
    for identity in physical_entry_segments:
        value_json, entry_pk_json = identity
        old_entry_by_pk.setdefault(
            entry_pk_json,
            IndexEntry(loads_strict(value_json), loads_strict(entry_pk_json)),
        )
    new_entry_by_pk = {entry.pk_json: entry for entry in expected.values()}
    derivation: list[dict[str, Any]] = []
    for pk_json in sorted(set(old_entry_by_pk) | set(new_entry_by_pk)):
        old_entry = old_entry_by_pk.get(pk_json)
        new_entry = new_entry_by_pk.get(pk_json)
        if old_entry is None and new_entry is None:
            continue
        if old_entry is not None and new_entry is not None and old_entry.identity() == new_entry.identity():
            continue
        source = physical_entry_segments.get(old_entry.identity(), []) if old_entry else []
        target = _route_entry(new_entry, before_segments) if new_entry else None
        reason = None
        if old_entry is not None and new_entry is None:
            reason = "document_delete" if pk_json not in final_pk_jsons else "key_migration"
        elif old_entry is not None and new_entry is not None:
            reason = "key_migration"
        else:
            reason = "document_insert"
        derivation.append(
            {
                "primary_key": loads_strict(pk_json),
                "old_key": old_entry.value if old_entry else None,
                "new_key": new_entry.value if new_entry else None,
                "source_segments": source,
                "target_segment": target,
                "affected_segments": sorted(set(source) | ({target} if target else set())),
                "reason": reason,
            }
        )

    conn.execute(
        "UPDATE indexes SET index_version=? WHERE collection_name=? AND path=?",
        (version_after, collection_name, index_path),
    )

    return {
        "collection": collection_name,
        "field_path": index_path,
        "version_before": int(collection["collection_version"]),
        "version_after": version_after,
        "affected_segments": affected_reports,
        "affected_segment_ids": sorted(
            affected_ids | {report["segment_id"] for report in affected_reports}
        ),
        "unchanged_segment_ids": unchanged,
        "stale_entries_found": len(stale_entries),
        "stale_entries_reclaimed": 0 if retain_stale else len(stale_entries),
        "ledger_added": ledger_added,
        "ledger_count": len(ledger_after),
        "derivation_chain": derivation,
    }


def _persist_document_diff(
    conn: sqlite3.Connection,
    collection_name: str,
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> None:
    for pk_json in sorted(set(before) | set(after)):
        old_doc = before.get(pk_json)
        new_doc = after.get(pk_json)
        if old_doc is None and new_doc is not None:
            conn.execute(
                "INSERT INTO documents(collection_name, pk_json, doc_json) VALUES(?,?,?)",
                (collection_name, pk_json, canonical_json(new_doc)),
            )
        elif old_doc is not None and new_doc is None:
            conn.execute(
                "DELETE FROM documents WHERE collection_name=? AND pk_json=?",
                (collection_name, pk_json),
            )
        elif old_doc is not None and new_doc is not None:
            new_json = canonical_json(new_doc)
            old_json = canonical_json(old_doc)
            if new_json != old_json:
                conn.execute(
                    "UPDATE documents SET doc_json=? WHERE collection_name=? AND pk_json=?",
                    (new_json, collection_name, pk_json),
                )


def check_consistency(
    conn: sqlite3.Connection,
    collection_name: str,
    path: str,
) -> tuple[int, int]:
    collection = get_collection(conn, collection_name)
    index_row = get_index(conn, collection_name, path)
    collection_version = int(collection["collection_version"])
    index_version = int(index_row["index_version"])
    if collection_version != index_version:
        raise StaleError(
            "index version does not match collection version",
            stale_dimension="index_version_vs_collection_version",
            collection=collection_name,
            field_path=path,
            collection_version=collection_version,
            index_version=index_version,
        )
    return collection_version, index_version


def apply_changes(
    session_dir: str | Path,
    collection_name: str,
    changes: list[dict[str, Any]],
    *,
    batch_id: str = "batch",
    batch_size: int | None = None,
    retain_stale: bool = False,
    target_version: int | None = None,
    record_audit: bool = True,
) -> dict[str, Any]:
    if batch_size is not None and batch_size <= 0:
        raise DocumentError("batch size must be a positive integer", field_path="batch_size")
    conn = open_session(session_dir)
    try:
        with transaction(conn):
            collection = get_collection(conn, collection_name)
            index_rows = list(
                conn.execute(
                    "SELECT collection_name, path, segment_capacity, allow_null, "
                    "index_version, next_segment_number FROM indexes "
                    "WHERE collection_name=? ORDER BY path",
                    (collection_name,),
                )
            )
            if not index_rows:
                raise StateError("collection has no index", collection=collection_name)
            version_before = int(collection["collection_version"])
            for row in index_rows:
                if int(row["index_version"]) != version_before:
                    raise StaleError(
                        "index and collection versions differ before apply",
                        stale_dimension="index_version_vs_collection_version",
                        collection=collection_name,
                        field_path=str(row["path"]),
                        collection_version=version_before,
                        index_version=int(row["index_version"]),
                    )
            index_specs = [
                IndexSpec(
                    path=str(row["path"]),
                    segment_capacity=int(row["segment_capacity"]),
                    allow_null=bool(row["allow_null"]),
                )
                for row in index_rows
            ]
            before_docs = load_documents(conn, collection_name)
            final_docs = {key: _deep_copy(value) for key, value in before_docs.items()}
            change_trace = _materialize_changes(
                final_docs,
                collection,
                index_specs,
                changes,
            )

            if changes:
                version_after = (
                    version_before + 1
                    if target_version is None
                    else target_version
                )
                if version_after < version_before + 1:
                    raise DocumentError(
                        "target version must advance for a non-empty batch",
                        field_path="target_version",
                    )
            else:
                version_after = version_before if target_version is None else target_version
                if version_after != version_before:
                    raise DocumentError(
                        "an empty maintenance batch may not change the collection version",
                        field_path="target_version",
                    )

            before_state = state_manifest(conn)
            _persist_document_diff(conn, collection_name, before_docs, final_docs)
            conn.execute(
                "UPDATE collections SET collection_version=? WHERE name=?",
                (version_after, collection_name),
            )
            collection = get_collection(conn, collection_name)

            index_reports: list[dict[str, Any]] = []
            affected_map: dict[str, Any] = {}
            for index_row in index_rows:
                report = _rebuild_index(
                    conn,
                    collection,
                    index_row,
                    final_docs,
                    version_after,
                    retain_stale=retain_stale,
                )
                index_reports.append(report)
                affected_map[str(index_row["path"])] = {
                    "affected_segment_ids": report["affected_segment_ids"],
                    "unchanged_segment_ids": report["unchanged_segment_ids"],
                    "affected_segments": report["affected_segments"],
                }
            sync_expected_ledger(conn)
            after_state = state_manifest(conn)

            if record_audit:
                append_audit_batch(
                    conn,
                    batch_id=batch_id,
                    collection_name=collection_name,
                    changes=changes,
                    options={
                        "batch_size": batch_size,
                        "retain_stale": retain_stale,
                        "target_version": target_version,
                    },
                    version_before=version_before,
                    version_after=version_after,
                    before_state=before_state,
                    after_state=after_state,
                    affected=affected_map,
                    ledger_after=ledger_rows(
                        conn,
                        collection_name=collection_name,
                    ),
                )
        processing_groups = (
            0
            if not changes
            else math.ceil(len(changes) / (batch_size or len(changes) or 1))
        )
        return {
            "ok": True,
            "collection": collection_name,
            "batch_id": batch_id,
            "collection_version_before": version_before,
            "collection_version_after": version_after,
            "change_count": len(changes),
            "batch_size": batch_size,
            "processing_groups": processing_groups,
            "retain_stale": retain_stale,
            "change_trace": change_trace,
            "indexes": index_reports,
            "state": after_state,
        }
    finally:
        conn.close()


def _numeric_bounds(
    *,
    eq: Any = None,
    lo: Any = None,
    hi: Any = None,
) -> tuple[int | float | None, int | float | None, bool]:
    equality = eq is not None
    if equality:
        if not is_number(eq):
            raise DocumentError("equality query key must be a finite number", field_path="eq")
        return eq, eq, True
    if lo is not None and not is_number(lo):
        raise DocumentError("range lower bound must be a finite number", field_path="lo")
    if hi is not None and not is_number(hi):
        raise DocumentError("range upper bound must be a finite number", field_path="hi")
    if lo is not None and hi is not None and lo > hi:
        raise DocumentError("range lower bound must not exceed upper bound", field_path="range")
    return lo, hi, False


def _candidate_segments(
    segments: list[SegmentRecord],
    lo: int | float | None,
    hi: int | float | None,
    equality: bool,
) -> list[SegmentRecord]:
    result: list[SegmentRecord] = []
    for segment in segments:
        segment_lo = loads_strict(segment.low_value_json)
        segment_hi = loads_strict(segment.high_value_json)
        if lo is not None and segment_hi < lo:
            continue
        if hi is not None and segment_lo > hi:
            continue
        result.append(segment)
    return result


def _current_entry(
    document: dict[str, Any] | None,
    index: IndexSpec,
) -> tuple[bool, Any]:
    if document is None:
        return False, None
    present, value = get_path(document, index.path)
    if not present:
        return False, None
    return True, value


def _query_items(
    conn: sqlite3.Connection,
    collection_name: str,
    index_spec: IndexSpec,
    docs: dict[str, dict[str, Any]],
    *,
    eq: Any = None,
    lo: Any = None,
    hi: Any = None,
    scan: str = "index",
) -> tuple[list[dict[str, Any]], list[str], int, dict[str, int]]:
    lower, upper, equality = _numeric_bounds(eq=eq, lo=lo, hi=hi)
    items: list[dict[str, Any]] = []
    candidates: list[str] = []
    filtered_by_segment: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()

    if scan == "full":
        for pk_json, document in docs.items():
            present, value = _current_entry(document, index_spec)
            if not present or value is None or not is_number(value):
                continue
            if lower is not None and value < lower:
                continue
            if upper is not None and value > upper:
                continue
            if equality and value != lower:
                continue
            pk = loads_strict(pk_json)
            entry = IndexEntry(value=value, pk=pk)
            if entry.identity() in seen:
                continue
            seen.add(entry.identity())
            items.append(
                {
                    "_sort": entry.sort_key,
                    "value": value,
                    "primary_key": pk,
                    "document": document,
                }
            )
    else:
        segments = load_segments(conn, collection_name, index_spec.path, with_entries=True)
        selected = _candidate_segments(segments, lower, upper, equality)
        candidates = [segment.segment_id for segment in selected]
        for segment in selected:
            for entry in segment.entries:
                if lower is not None and entry.value < lower:
                    continue
                if upper is not None and entry.value > upper:
                    continue
                if equality and entry.value != lower:
                    continue
                present, current_value = _current_entry(docs.get(entry.pk_json), index_spec)
                if (
                    not present
                    or current_value is None
                    or not is_number(current_value)
                    or current_value != entry.value
                ):
                    filtered_by_segment[segment.segment_id] = (
                        filtered_by_segment.get(segment.segment_id, 0) + 1
                    )
                    continue
                if entry.identity() in seen:
                    continue
                seen.add(entry.identity())
                items.append(
                    {
                        "_sort": entry.sort_key,
                        "value": entry.value,
                        "primary_key": entry.pk,
                        "document": docs[entry.pk_json],
                    }
                )
    items.sort(key=lambda item: item["_sort"])
    for item in items:
        del item["_sort"]
    return items, candidates, sum(filtered_by_segment.values()), filtered_by_segment


def query_index(
    session_dir: str | Path,
    collection_name: str,
    path: str,
    *,
    eq: Any = None,
    lo: Any = None,
    hi: Any = None,
    offset: int = 0,
    limit: int | None = None,
    scan: str = "index",
) -> dict[str, Any]:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise DocumentError("offset must be a non-negative integer", field_path="offset")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
    ):
        raise DocumentError("limit must be a non-negative integer", field_path="limit")
    if scan not in {"index", "full"}:
        raise DocumentError("scan must be 'index' or 'full'", field_path="scan")
    conn = open_session(session_dir)
    try:
        with transaction(conn):
            _, _ = check_consistency(conn, collection_name, path)
            collection = get_collection(conn, collection_name)
            index_row = get_index(conn, collection_name, path)
            index_spec = IndexSpec(
                path=path,
                segment_capacity=int(index_row["segment_capacity"]),
                allow_null=bool(index_row["allow_null"]),
            )
            docs = load_documents(conn, collection_name)
            items, candidates, filtered, filtered_by_segment = _query_items(
                conn,
                collection_name,
                index_spec,
                docs,
                eq=eq,
                lo=lo,
                hi=hi,
                scan=scan,
            )
            total = len(items)
            end = None if limit is None else offset + limit
            page = items[offset:end]
            next_offset = None
            if end is not None and end < total:
                next_offset = end
            next_offset = next_offset if page else None
            query_payload = {
                "collection": collection_name,
                "field_path": path,
                "kind": "equality" if eq is not None else ("range" if lo is not None or hi is not None else "scan"),
                "eq": eq,
                "lo": lo,
                "hi": hi,
                "offset": offset,
                "limit": limit,
                "scan": scan,
            }
            append_query_history(
                conn,
                collection_name=collection_name,
                index_path=path,
                query=query_payload,
                status="ok",
                result_digest=digest_json(page),
                filtered_count=filtered,
            )
            if scan == "index" and filtered_by_segment:
                increment_isolation_stats(
                    conn,
                    collection_name,
                    path,
                    filtered_by_segment,
                )
            return {
                "ok": True,
                "collection": collection_name,
                "field_path": path,
                "scan": scan,
                "collection_version": int(collection["collection_version"]),
                "index_version": int(index_row["index_version"]),
                "candidate_segments": candidates,
                "total_before_pagination": total,
                "offset": offset,
                "limit": limit,
                "next_offset": next_offset,
                "filtered_stale": filtered,
                "items": page,
                "result_digest": digest_json(page),
            }
    finally:
        conn.close()


def full_scan(
    session_dir: str | Path,
    collection_name: str,
    path: str,
    *,
    eq: Any = None,
    lo: Any = None,
    hi: Any = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    return query_index(
        session_dir,
        collection_name,
        path,
        eq=eq,
        lo=lo,
        hi=hi,
        offset=offset,
        limit=limit,
        scan="full",
    )


def manifest(session_dir: str | Path) -> dict[str, Any]:
    conn = open_session(session_dir)
    try:
        state = state_manifest(conn)
        return {
            "ok": True,
            "session": str(Path(session_dir).resolve()),
            **state,
        }
    finally:
        conn.close()


def diagnose(
    session_dir: str | Path,
    collection_name: str,
    path: str,
) -> dict[str, Any]:
    conn = open_session(session_dir)
    try:
        collection = get_collection(conn, collection_name)
        index_row = get_index(conn, collection_name, path)
        index_spec = IndexSpec(
            path=path,
            segment_capacity=int(index_row["segment_capacity"]),
            allow_null=bool(index_row["allow_null"]),
        )
        docs = load_documents(conn, collection_name)
        segments = load_segments(conn, collection_name, path, with_entries=True)
        filtered_counts = isolation_stats(conn, collection_name, path)
        report_segments: list[dict[str, Any]] = []
        sorted_segments = sorted(segments, key=lambda segment: segment.segment_id)
        for segment in sorted_segments:
            stale_keys: list[IndexEntry] = []
            for entry in segment.entries:
                present, current_value = _current_entry(docs.get(entry.pk_json), index_spec)
                if (
                    not present
                    or current_value is None
                    or not is_number(current_value)
                    or current_value != entry.value
                ):
                    stale_keys.append(entry)
            stale_keys.sort(key=lambda entry: entry.sort_key)
            report_segments.append(
                {
                    "segment_id": segment.segment_id,
                    "version": segment.version,
                    "entry_count": segment.entry_count,
                    "low_key": loads_strict(segment.low_value_json),
                    "low_pk": loads_strict(segment.low_pk_json),
                    "high_key": loads_strict(segment.high_value_json),
                    "high_pk": loads_strict(segment.high_pk_json),
                    "digest": segment.digest,
                    "stale_entry_count": len(stale_keys),
                    "filtered_count": int(filtered_counts.get(segment.segment_id, 0)),
                    "stale_keys": [
                        {
                            "value": entry.value,
                            "primary_key": entry.pk,
                        }
                        for entry in stale_keys
                    ],
                }
            )
        ledger = ledger_rows(
            conn,
            collection_name=collection_name,
            index_path=path,
        )
        return {
            "ok": True,
            "collection": collection_name,
            "field_path": path,
            "collection_version": int(collection["collection_version"]),
            "index_version": int(index_row["index_version"]),
            "consistent": int(collection["collection_version"]) == int(index_row["index_version"]),
            "segments": report_segments,
            "stale_ledger": ledger,
            "stale_ledger_count": len(ledger),
            "isolation_filtered_total": sum(item["filtered_count"] for item in report_segments),
        }
    finally:
        conn.close()

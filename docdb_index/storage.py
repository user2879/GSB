"""SQLite persistence primitives for segments, ledgers and audit state."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .canonical import (
    canonical_json,
    digest_bytes,
    digest_json,
    loads_strict,
    number_from_json_text,
)
from .errors import StateError
from .models import (
    CollectionSpec,
    IndexEntry,
    IndexSpec,
    SegmentRecord,
    segment_content_blob,
    segment_digest,
)

SCHEMA_VERSION = 1
SESSION_DB_NAME = "session.db"
LOG_RELATIVE_PATH = Path("logs") / "events.jsonl"


# The segment table intentionally does not use a foreign key to indexes: SQLite
# cannot express a foreign key into a partial composite key with a different
# column name in older versions. This table is recreated below.
CORE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collections (
    name TEXT PRIMARY KEY,
    primary_key TEXT NOT NULL,
    collection_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS indexes (
    collection_name TEXT NOT NULL,
    path TEXT NOT NULL,
    segment_capacity INTEGER NOT NULL,
    allow_null INTEGER NOT NULL,
    index_version INTEGER NOT NULL,
    next_segment_number INTEGER NOT NULL,
    PRIMARY KEY (collection_name, path),
    FOREIGN KEY (collection_name) REFERENCES collections(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS documents (
    collection_name TEXT NOT NULL,
    pk_json TEXT NOT NULL,
    doc_json TEXT NOT NULL,
    PRIMARY KEY (collection_name, pk_json),
    FOREIGN KEY (collection_name) REFERENCES collections(name) ON DELETE CASCADE
);
"""

SEGMENT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS segments (
    collection_name TEXT NOT NULL,
    index_path TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    entry_count INTEGER NOT NULL,
    low_value_json TEXT NOT NULL,
    low_pk_json TEXT NOT NULL,
    high_value_json TEXT NOT NULL,
    high_pk_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    content_blob BLOB NOT NULL,
    PRIMARY KEY (collection_name, index_path, segment_id),
    UNIQUE (collection_name, index_path, table_name)
);
"""

STALE_TABLE_SQL = """
CREATE TABLE {table} (
    collection_name TEXT NOT NULL,
    index_path TEXT NOT NULL,
    pk_json TEXT NOT NULL,
    stale_value_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    original_segment TEXT NOT NULL,
    discovered_collection_version INTEGER NOT NULL,
    last_seen_collection_version INTEGER NOT NULL,
    PRIMARY KEY (collection_name, index_path, pk_json, stale_value_json)
);
"""

OTHER_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stale_ledger (
    collection_name TEXT NOT NULL,
    index_path TEXT NOT NULL,
    pk_json TEXT NOT NULL,
    stale_value_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    original_segment TEXT NOT NULL,
    discovered_collection_version INTEGER NOT NULL,
    last_seen_collection_version INTEGER NOT NULL,
    PRIMARY KEY (collection_name, index_path, pk_json, stale_value_json)
);

CREATE TABLE IF NOT EXISTS stale_ledger_expected (
    collection_name TEXT NOT NULL,
    index_path TEXT NOT NULL,
    pk_json TEXT NOT NULL,
    stale_value_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    original_segment TEXT NOT NULL,
    discovered_collection_version INTEGER NOT NULL,
    last_seen_collection_version INTEGER NOT NULL,
    PRIMARY KEY (collection_name, index_path, pk_json, stale_value_json)
);

CREATE TABLE IF NOT EXISTS query_history (
    seq INTEGER PRIMARY KEY,
    collection_name TEXT NOT NULL,
    index_path TEXT NOT NULL,
    query_json TEXT NOT NULL,
    status TEXT NOT NULL,
    result_digest TEXT NOT NULL,
    filtered_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS isolation_stats (
    collection_name TEXT NOT NULL,
    index_path TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    filtered_count INTEGER NOT NULL,
    PRIMARY KEY (collection_name, index_path, segment_id)
);

CREATE TABLE IF NOT EXISTS audit_baseline (
    collection_name TEXT PRIMARY KEY,
    declaration_json TEXT NOT NULL,
    documents_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_batches (
    seq INTEGER PRIMARY KEY,
    batch_id TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    changes_json TEXT NOT NULL,
    options_json TEXT NOT NULL,
    version_before INTEGER NOT NULL,
    version_after INTEGER NOT NULL,
    before_json TEXT NOT NULL,
    after_json TEXT NOT NULL,
    affected_json TEXT NOT NULL,
    ledger_after_json TEXT NOT NULL
);
"""


def session_db_path(session_dir: str | Path) -> Path:
    return Path(session_dir) / SESSION_DB_NAME


def ensure_session_dirs(session_dir: str | Path) -> Path:
    path = Path(session_dir)
    (path / "logs").mkdir(parents=True, exist_ok=True)
    (path / "snapshots").mkdir(parents=True, exist_ok=True)
    return path


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def open_session(session_dir: str | Path) -> sqlite3.Connection:
    db_path = session_db_path(session_dir)
    if not db_path.is_file():
        raise StateError("session database does not exist", session=str(session_dir))
    conn = connect(db_path)
    version = read_meta(conn, "schema_version")
    if version != str(SCHEMA_VERSION):
        conn.close()
        raise StateError(
            "unsupported session schema version",
            expected=SCHEMA_VERSION,
            actual=version,
        )
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(CORE_SCHEMA_SQL)
    conn.executescript(SEGMENT_TABLE_SQL)
    conn.executescript(OTHER_SCHEMA_SQL)
    write_meta(conn, "schema_version", str(SCHEMA_VERSION))


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()


def write_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def read_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def collection_versions(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        str(row["name"]): int(row["collection_version"])
        for row in conn.execute("SELECT name, collection_version FROM collections ORDER BY name")
    }


def index_versions(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        f"{row['collection_name']}|{row['path']}": int(row["index_version"])
        for row in conn.execute(
            "SELECT collection_name, path, index_version FROM indexes "
            "ORDER BY collection_name, path"
        )
    }


def load_collection_specs(conn: sqlite3.Connection) -> list[CollectionSpec]:
    specs: list[CollectionSpec] = []
    for row in conn.execute(
        "SELECT name, primary_key, collection_version FROM collections ORDER BY name"
    ):
        name = str(row["name"])
        indexes = tuple(
            IndexSpec(
                path=str(index_row["path"]),
                segment_capacity=int(index_row["segment_capacity"]),
                allow_null=bool(index_row["allow_null"]),
            )
            for index_row in conn.execute(
                "SELECT path, segment_capacity, allow_null FROM indexes "
                "WHERE collection_name=? ORDER BY path",
                (name,),
            )
        )
        docs = tuple(
            loads_strict(str(doc_row["doc_json"]))
            for doc_row in conn.execute(
                "SELECT doc_json FROM documents WHERE collection_name=? ORDER BY pk_json",
                (name,),
            )
        )
        specs.append(
            CollectionSpec(
                name=name,
                version=int(row["collection_version"]),
                primary_key=str(row["primary_key"]),
                documents=docs,
                indexes=indexes,
            )
        )
    return specs


def get_collection(conn: sqlite3.Connection, collection_name: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT name, primary_key, collection_version FROM collections WHERE name=?",
        (collection_name,),
    ).fetchone()
    if row is None:
        raise StateError("collection does not exist", collection=collection_name)
    return row


def get_index(conn: sqlite3.Connection, collection_name: str, path: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT collection_name, path, segment_capacity, allow_null, index_version, "
        "next_segment_number FROM indexes WHERE collection_name=? AND path=?",
        (collection_name, path),
    ).fetchone()
    if row is None:
        raise StateError(
            "index does not exist",
            collection=collection_name,
            field_path=path,
        )
    return row


def load_documents(conn: sqlite3.Connection, collection_name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in conn.execute(
        "SELECT pk_json, doc_json FROM documents WHERE collection_name=? ORDER BY pk_json",
        (collection_name,),
    ):
        pk_json = str(row["pk_json"])
        result[pk_json] = loads_strict(str(row["doc_json"]))
    return result


def document_key(document: dict[str, Any], primary_key: str) -> tuple[Any, str]:
    from .validation import get_path

    present, pk = get_path(document, primary_key)
    if not present:
        raise StateError("document lost its primary key", field_path=primary_key)
    return pk, canonical_json(pk)


def segment_table_name(collection_name: str, path: str, segment_id: str) -> str:
    token = digest_bytes(f"{collection_name}\0{path}\0{segment_id}".encode("utf-8"))[:24]
    return f"seg_entries_{token}"


def read_segment_entries(conn: sqlite3.Connection, table_name: str) -> list[IndexEntry]:
    rows = conn.execute(
        f'SELECT value_json, pk_json FROM "{table_name}" ORDER BY seq'
    ).fetchall()
    entries: list[IndexEntry] = []
    for row in rows:
        value_text = str(row["value_json"])
        pk_text = str(row["pk_json"])
        entries.append(
            IndexEntry(
                value=number_from_json_text(value_text),
                pk=loads_strict(pk_text),
            )
        )
    return entries


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def load_segments(
    conn: sqlite3.Connection,
    collection_name: str,
    index_path: str,
    *,
    with_entries: bool = True,
) -> list[SegmentRecord]:
    rows = conn.execute(
        "SELECT segment_id, table_name, version, entry_count, low_value_json, "
        "low_pk_json, high_value_json, high_pk_json, digest, content_blob "
        "FROM segments WHERE collection_name=? AND index_path=?",
        (collection_name, index_path),
    ).fetchall()
    records: list[SegmentRecord] = []
    for row in rows:
        record = SegmentRecord(
            segment_id=str(row["segment_id"]),
            table_name=str(row["table_name"]),
            version=int(row["version"]),
            entry_count=int(row["entry_count"]),
            low_value_json=str(row["low_value_json"]),
            low_pk_json=str(row["low_pk_json"]),
            high_value_json=str(row["high_value_json"]),
            high_pk_json=str(row["high_pk_json"]),
            digest=str(row["digest"]),
            content_blob=bytes(row["content_blob"]),
        )
        if with_entries:
            record.entries = read_segment_entries(conn, record.table_name)
        records.append(record)
    records.sort(key=lambda item: (item.low_key, item.segment_id))
    return records


def write_segment(
    conn: sqlite3.Connection,
    collection_name: str,
    index_path: str,
    segment_id: str,
    version: int,
    entries: list[IndexEntry],
) -> SegmentRecord:
    if not entries:
        raise StateError("cannot persist an empty segment", segment_id=segment_id)
    ordered = sorted(entries, key=lambda item: item.sort_key)
    table_name = segment_table_name(collection_name, index_path, segment_id)
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.execute(
        f'CREATE TABLE "{table_name}" ('
        "seq INTEGER PRIMARY KEY, "
        "value_json TEXT NOT NULL, "
        "pk_json TEXT NOT NULL, "
        "UNIQUE(value_json, pk_json))"
    )
    for seq, entry in enumerate(ordered, start=1):
        conn.execute(
            f'INSERT INTO "{table_name}"(seq, value_json, pk_json) VALUES(?, ?, ?)',
            (seq, entry.value_json, entry.pk_json),
        )
    blob = segment_content_blob(segment_id, version, ordered)
    digest = segment_digest(blob)
    low = ordered[0]
    high = ordered[-1]
    conn.execute(
        "INSERT INTO segments("
        "collection_name, index_path, segment_id, table_name, version, entry_count, "
        "low_value_json, low_pk_json, high_value_json, high_pk_json, digest, content_blob"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(collection_name, index_path, segment_id) DO UPDATE SET "
        "table_name=excluded.table_name, version=excluded.version, "
        "entry_count=excluded.entry_count, low_value_json=excluded.low_value_json, "
        "low_pk_json=excluded.low_pk_json, high_value_json=excluded.high_value_json, "
        "high_pk_json=excluded.high_pk_json, digest=excluded.digest, "
        "content_blob=excluded.content_blob",
        (
            collection_name,
            index_path,
            segment_id,
            table_name,
            version,
            len(ordered),
            low.value_json,
            low.pk_json,
            high.value_json,
            high.pk_json,
            digest,
            sqlite3.Binary(blob),
        ),
    )
    return SegmentRecord(
        segment_id=segment_id,
        table_name=table_name,
        version=version,
        entry_count=len(ordered),
        low_value_json=low.value_json,
        low_pk_json=low.pk_json,
        high_value_json=high.value_json,
        high_pk_json=high.pk_json,
        digest=digest,
        content_blob=blob,
        entries=ordered,
    )


def delete_segment(
    conn: sqlite3.Connection,
    collection_name: str,
    index_path: str,
    segment_id: str,
    table_name: str,
) -> None:
    conn.execute(
        "DELETE FROM segments WHERE collection_name=? AND index_path=? AND segment_id=?",
        (collection_name, index_path, segment_id),
    )
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.execute(
        "DELETE FROM isolation_stats WHERE collection_name=? AND index_path=? AND segment_id=?",
        (collection_name, index_path, segment_id),
    )


def allocate_segment_ids(conn: sqlite3.Connection, collection_name: str, path: str, count: int) -> list[str]:
    if count <= 0:
        return []
    row = conn.execute(
        "SELECT next_segment_number FROM indexes WHERE collection_name=? AND path=?",
        (collection_name, path),
    ).fetchone()
    if row is None:
        raise StateError("index disappeared while allocating segment ids", field_path=path)
    start = int(row["next_segment_number"])
    ids = [f"seg-{number:08d}" for number in range(start, start + count)]
    conn.execute(
        "UPDATE indexes SET next_segment_number=? WHERE collection_name=? AND path=?",
        (start + count, collection_name, path),
    )
    return ids


def segment_manifest(
    conn: sqlite3.Connection,
    collection_name: str | None = None,
    index_path: str | None = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if collection_name is not None:
        clauses.append("collection_name=?")
        params.append(collection_name)
    if index_path is not None:
        clauses.append("index_path=?")
        params.append(index_path)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        "SELECT collection_name, index_path, segment_id, version, entry_count, "
        "low_value_json, low_pk_json, high_value_json, high_pk_json, digest "
        f"FROM segments {where}",
        params,
    ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "collection": str(row["collection_name"]),
                "index_path": str(row["index_path"]),
                "segment_id": str(row["segment_id"]),
                "version": int(row["version"]),
                "entry_count": int(row["entry_count"]),
                "low_key": loads_strict(str(row["low_value_json"])),
                "low_pk": loads_strict(str(row["low_pk_json"])),
                "high_key": loads_strict(str(row["high_value_json"])),
                "high_pk": loads_strict(str(row["high_pk_json"])),
                "digest": str(row["digest"]),
            }
        )
    result.sort(
        key=lambda item: (
            item["collection"],
            item["index_path"],
            item["segment_id"],
        )
    )
    return result


def segment_manifest_map(
    conn: sqlite3.Connection,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in segment_manifest(conn):
        key = f"{item['collection']}|{item['index_path']}|{item['segment_id']}"
        result[key] = item
    return result


def ledger_rows(
    conn: sqlite3.Connection,
    table: str = "stale_ledger",
    collection_name: str | None = None,
    index_path: str | None = None,
) -> list[dict[str, Any]]:
    if table not in {"stale_ledger", "stale_ledger_expected"}:
        raise ValueError("invalid ledger table")
    clauses: list[str] = []
    params: list[Any] = []
    if collection_name is not None:
        clauses.append("collection_name=?")
        params.append(collection_name)
    if index_path is not None:
        clauses.append("index_path=?")
        params.append(index_path)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        "SELECT collection_name, index_path, pk_json, stale_value_json, reason, "
        "original_segment, discovered_collection_version, last_seen_collection_version "
        f"FROM {table} {where}",
        params,
    ).fetchall()
    result = [
        {
            "collection": str(row["collection_name"]),
            "index_path": str(row["index_path"]),
            "primary_key": loads_strict(str(row["pk_json"])),
            "pk_json": str(row["pk_json"]),
            "stale_key": loads_strict(str(row["stale_value_json"])),
            "stale_key_json": str(row["stale_value_json"]),
            "reason": str(row["reason"]),
            "original_segment": str(row["original_segment"]),
            "discovered_collection_version": int(row["discovered_collection_version"]),
            "last_seen_collection_version": int(row["last_seen_collection_version"]),
        }
        for row in rows
    ]
    result.sort(
        key=lambda item: (
            item["collection"],
            item["index_path"],
            item["stale_key_json"],
            item["pk_json"],
            item["reason"],
            item["original_segment"],
        )
    )
    return result


def insert_stale_ledger(
    conn: sqlite3.Connection,
    *,
    collection_name: str,
    index_path: str,
    pk_json: str,
    stale_value_json: str,
    reason: str,
    original_segment: str,
    collection_version: int,
) -> None:
    conn.execute(
        "INSERT INTO stale_ledger("
        "collection_name, index_path, pk_json, stale_value_json, reason, "
        "original_segment, discovered_collection_version, last_seen_collection_version"
        ") VALUES(?,?,?,?,?,?,?,?) "
        "ON CONFLICT(collection_name, index_path, pk_json, stale_value_json) DO UPDATE SET "
        "last_seen_collection_version="
        "CASE WHEN excluded.last_seen_collection_version > stale_ledger.last_seen_collection_version "
        "THEN excluded.last_seen_collection_version "
        "ELSE stale_ledger.last_seen_collection_version END",
        (
            collection_name,
            index_path,
            pk_json,
            stale_value_json,
            reason,
            original_segment,
            collection_version,
            collection_version,
        ),
    )


def sync_expected_ledger(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM stale_ledger_expected")
    conn.execute(
        "INSERT INTO stale_ledger_expected("
        "collection_name, index_path, pk_json, stale_value_json, reason, "
        "original_segment, discovered_collection_version, last_seen_collection_version"
        ") SELECT collection_name, index_path, pk_json, stale_value_json, reason, "
        "original_segment, discovered_collection_version, last_seen_collection_version "
        "FROM stale_ledger"
    )
    grouped = conn.execute(
        "SELECT collection_name, index_path FROM stale_ledger "
        "GROUP BY collection_name, index_path ORDER BY collection_name, index_path"
    ).fetchall()
    for row in grouped:
        rows = ledger_rows(
            conn,
            collection_name=str(row["collection_name"]),
            index_path=str(row["index_path"]),
        )
        write_meta(
            conn,
            f"ledger_digest:{row['collection_name']}:{row['index_path']}",
            digest_json(rows),
        )


def ledger_digest(conn: sqlite3.Connection, table: str = "stale_ledger") -> str:
    return digest_json(ledger_rows(conn, table=table))


def query_history_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        {
            "seq": int(row["seq"]),
            "collection": str(row["collection_name"]),
            "index_path": str(row["index_path"]),
            "query": loads_strict(str(row["query_json"])),
            "status": str(row["status"]),
            "result_digest": str(row["result_digest"]),
            "filtered_count": int(row["filtered_count"]),
        }
        for row in conn.execute(
            "SELECT seq, collection_name, index_path, query_json, status, "
            "result_digest, filtered_count FROM query_history ORDER BY seq"
        )
    ]


def query_history_digest(conn: sqlite3.Connection) -> str:
    return digest_json(query_history_rows(conn))


def append_query_history(
    conn: sqlite3.Connection,
    *,
    collection_name: str,
    index_path: str,
    query: dict[str, Any],
    status: str,
    result_digest: str,
    filtered_count: int,
) -> None:
    row = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS seq FROM query_history").fetchone()
    seq = int(row["seq"])
    conn.execute(
        "INSERT INTO query_history("
        "seq, collection_name, index_path, query_json, status, result_digest, filtered_count"
        ") VALUES(?,?,?,?,?,?,?)",
        (
            seq,
            collection_name,
            index_path,
            canonical_json(query),
            status,
            result_digest,
            filtered_count,
        ),
    )


def isolation_stats(conn: sqlite3.Connection, collection_name: str, index_path: str) -> dict[str, int]:
    return {
        str(row["segment_id"]): int(row["filtered_count"])
        for row in conn.execute(
            "SELECT segment_id, filtered_count FROM isolation_stats "
            "WHERE collection_name=? AND index_path=? ORDER BY segment_id",
            (collection_name, index_path),
        )
    }


def increment_isolation_stats(
    conn: sqlite3.Connection,
    collection_name: str,
    index_path: str,
    counts: dict[str, int],
) -> None:
    for segment_id in sorted(counts):
        count = counts[segment_id]
        if count <= 0:
            continue
        conn.execute(
            "INSERT INTO isolation_stats("
            "collection_name, index_path, segment_id, filtered_count"
            ") VALUES(?,?,?,?) ON CONFLICT(collection_name, index_path, segment_id) "
            "DO UPDATE SET filtered_count=filtered_count + excluded.filtered_count",
            (collection_name, index_path, segment_id, count),
        )


def state_manifest(conn: sqlite3.Connection) -> dict[str, Any]:
    segments = segment_manifest(conn)
    return {
        "collection_versions": collection_versions(conn),
        "index_versions": index_versions(conn),
        "segments": segments,
        "segment_digests": {
            f"{item['collection']}|{item['index_path']}|{item['segment_id']}": item["digest"]
            for item in segments
        },
        "ledger_digest": ledger_digest(conn),
        "query_history_digest": query_history_digest(conn),
        "query_history_count": len(query_history_rows(conn)),
    }


def write_audit_baseline(
    conn: sqlite3.Connection,
    specs: Iterable[CollectionSpec],
) -> None:
    for spec in specs:
        declarations = {
            "name": spec.name,
            "version": spec.version,
            "primary_key": spec.primary_key,
            "indexes": [
                {
                    "path": index.path,
                    "segment_capacity": index.segment_capacity,
                    "allow_null": index.allow_null,
                }
                for index in spec.indexes
            ],
        }
        from .validation import get_path

        documents = sorted(
            spec.documents,
            key=lambda doc: canonical_json(get_path(doc, spec.primary_key)[1]),
        )
        conn.execute(
            "INSERT OR REPLACE INTO audit_baseline("
            "collection_name, declaration_json, documents_json) VALUES(?,?,?)",
            (spec.name, canonical_json(declarations), canonical_json(documents)),
        )


def append_audit_batch(
    conn: sqlite3.Connection,
    *,
    batch_id: str,
    collection_name: str,
    changes: list[dict[str, Any]],
    options: dict[str, Any],
    version_before: int,
    version_after: int,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    affected: dict[str, Any],
    ledger_after: list[dict[str, Any]],
) -> None:
    row = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS seq FROM audit_batches").fetchone()
    conn.execute(
        "INSERT INTO audit_batches("
        "seq, batch_id, collection_name, changes_json, options_json, version_before, "
        "version_after, before_json, after_json, affected_json, ledger_after_json"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            int(row["seq"]),
            batch_id,
            collection_name,
            canonical_json(changes),
            canonical_json(options),
            version_before,
            version_after,
            canonical_json(before_state),
            canonical_json(after_state),
            canonical_json(affected),
            canonical_json(ledger_after),
        ),
    )


def log_event(
    session_dir: str | Path,
    *,
    level: str,
    code: str,
    message: str,
    context: dict[str, Any] | None = None,
) -> None:
    path = Path(session_dir) / LOG_RELATIVE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level,
            "code": code,
            "message": message,
            "context": context or {},
        }
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(record) + "\n")
    except Exception:
        # Logging must never mask the business error being reported.
        pass

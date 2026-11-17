"""SQLite-backed state transition ledger with versioned full replacement."""

import json
import os
import shutil
import sqlite3

from .errors import BusError, E_LEDGER, E_PROGRESS, E_STATE

VALID_STATES = ("idle", "processing", "committed", "failed")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transitions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  subscriber_id TEXT NOT NULL,
  source_id TEXT NOT NULL,
  event_seq INTEGER NOT NULL,
  from_state TEXT NOT NULL,
  to_state TEXT NOT NULL,
  clock_ms INTEGER NOT NULL,
  ledger_version INTEGER NOT NULL,
  complete INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS progress (
  subscriber_id TEXT PRIMARY KEY,
  committed_seq INTEGER NOT NULL
);
"""

META_KEYS = ("format_version", "set_id", "set_version", "subscriber_snapshot")


class ReplacementInfo:
    def __init__(self, replaced, backup_path):
        self.replaced = replaced
        self.backup_path = backup_path


def _connect(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _read_meta(connection):
    try:
        rows = connection.execute("SELECT key, value FROM meta").fetchall()
    except sqlite3.DatabaseError as exc:
        raise BusError(E_LEDGER, "ledger meta table unreadable: %s" % exc, field="meta")
    return {row["key"]: row["value"] for row in rows}


def _validate_meta(meta):
    for key in META_KEYS:
        if key not in meta:
            raise BusError(E_LEDGER, "ledger meta missing key %r" % key, field="meta." + key)
    for key in ("format_version", "set_version"):
        try:
            int(meta[key])
        except (TypeError, ValueError):
            raise BusError(E_LEDGER, "ledger meta key %r is not an integer" % key,
                           field="meta." + key)


def _validate_rows(connection):
    try:
        rows = connection.execute(
            "SELECT id, subscriber_id, event_seq, from_state, to_state, clock_ms,"
            " ledger_version, complete FROM transitions"
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise BusError(E_LEDGER, "ledger transitions table unreadable: %s" % exc,
                       field="transitions")
    for row in rows:
        field = "transitions[%d]" % row["id"]
        for state_field in ("from_state", "to_state"):
            if row[state_field] not in VALID_STATES:
                raise BusError(
                    E_LEDGER,
                    "invalid %s value %r" % (state_field, row[state_field]),
                    field=field + "." + state_field,
                    subscriber_id=row["subscriber_id"],
                    event_seq=row["event_seq"],
                )
        for int_field in ("event_seq", "clock_ms", "ledger_version", "complete"):
            if not isinstance(row[int_field], int):
                raise BusError(E_LEDGER, "field %s is not an integer" % int_field,
                               field=field + "." + int_field,
                               subscriber_id=row["subscriber_id"])


def _write_fresh(connection, format_version, set_id, set_version, snapshot, replacements):
    connection.executescript(SCHEMA)
    meta = {
        "format_version": str(format_version),
        "set_id": set_id,
        "set_version": str(set_version),
        "subscriber_snapshot": snapshot,
        "replacements": json.dumps(replacements, ensure_ascii=False, sort_keys=True),
    }
    with connection:
        for key, value in meta.items():
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))


def open_ledger(path, format_version, set_id, set_version, snapshot, logger=None):
    """Open the ledger, performing a full replacement when versions diverge.

    Returns ``(ledger, replacement_info)``; ``replacement_info`` is None when
    the existing ledger matches the requested versions.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    existed = os.path.exists(path) and os.path.getsize(path) > 0
    replacement = None
    if existed:
        try:
            connection = _connect(path)
            meta = _read_meta(connection)
        except sqlite3.DatabaseError as exc:
            raise BusError(E_LEDGER, "ledger file is not a readable database: %s" % exc,
                           field="ledger")
        _validate_meta(meta)
        _validate_rows(connection)
        old_format = int(meta["format_version"])
        old_set_version = int(meta["set_version"])
        if old_format != format_version or old_set_version != set_version:
            connection.close()
            backup_path = "%s.backup-f%d-s%d" % (path, old_format, old_set_version)
            shutil.copyfile(path, backup_path)
            os.chmod(backup_path, 0o444)
            record = {
                "replaced_set_id": meta["set_id"],
                "format_version_old": old_format,
                "format_version_new": format_version,
                "set_version_old": old_set_version,
                "set_version_new": set_version,
                "backup_path": backup_path,
            }
            os.remove(path)
            for suffix in ("-wal", "-shm"):
                if os.path.exists(path + suffix):
                    os.remove(path + suffix)
            connection = _connect(path)
            _write_fresh(connection, format_version, set_id, set_version, snapshot, [record])
            replacement = ReplacementInfo(record, backup_path)
            if logger:
                logger.log("INFO", "ledger replaced wholesale",
                           replaced_set_id=record["replaced_set_id"],
                           format_version_old=old_format,
                           format_version_new=format_version,
                           set_version_old=old_set_version,
                           set_version_new=set_version)
        ledger = Ledger(connection, path, format_version)
        return ledger, replacement
    connection = _connect(path)
    _write_fresh(connection, format_version, set_id, set_version, snapshot, [])
    return Ledger(connection, path, format_version), None


class Ledger:
    def __init__(self, connection, path, format_version):
        self.connection = connection
        self.path = path
        self.format_version = format_version

    def close(self):
        self.connection.close()

    def meta(self):
        return _read_meta(self.connection)

    def record_transition(self, subscriber_id, source_id, event_seq, from_state,
                          to_state, clock_ms, complete=1):
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO transitions(subscriber_id, source_id, event_seq, from_state,"
                " to_state, clock_ms, ledger_version, complete)"
                " VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (subscriber_id, source_id, event_seq, from_state, to_state, clock_ms,
                 self.format_version, complete))
        return cursor.lastrowid

    def has_committed(self, subscriber_id, source_id, event_seq):
        row = self.connection.execute(
            "SELECT 1 FROM transitions WHERE subscriber_id=? AND source_id=?"
            " AND event_seq=? AND to_state='committed' AND complete=1 LIMIT 1",
            (subscriber_id, source_id, event_seq)).fetchone()
        return row is not None

    def commit_with_progress(self, subscriber_id, source_id, event_seq, from_state,
                             clock_ms):
        """Write the committed transition and the progress row atomically."""
        if self.has_committed(subscriber_id, source_id, event_seq):
            raise BusError(
                E_STATE,
                "duplicate committed transition rejected; original state preserved",
                subscriber_id=subscriber_id, event_seq=event_seq)
        with self.connection:
            self.connection.execute(
                "INSERT INTO transitions(subscriber_id, source_id, event_seq, from_state,"
                " to_state, clock_ms, ledger_version, complete)"
                " VALUES(?, ?, ?, ?, 'committed', ?, ?, 1)",
                (subscriber_id, source_id, event_seq, from_state, clock_ms,
                 self.format_version))
            self.connection.execute(
                "INSERT INTO progress(subscriber_id, committed_seq) VALUES(?, ?)"
                " ON CONFLICT(subscriber_id) DO UPDATE SET"
                " committed_seq = MAX(committed_seq, excluded.committed_seq)",
                (subscriber_id, event_seq))
            self._assert_progress_within_ledger(subscriber_id)

    def _assert_progress_within_ledger(self, subscriber_id):
        progress = self.get_progress(subscriber_id)
        max_committed = self.max_committed_seq(subscriber_id)
        if progress is not None and (max_committed is None or progress > max_committed):
            raise BusError(
                E_PROGRESS,
                "progress %s exceeds max committed event seq %s" % (progress, max_committed),
                subscriber_id=subscriber_id, event_seq=progress)

    def check_progress_consistency(self):
        rows = self.connection.execute(
            "SELECT subscriber_id, committed_seq FROM progress ORDER BY subscriber_id"
        ).fetchall()
        for row in rows:
            self._assert_progress_within_ledger(row["subscriber_id"])

    def get_progress(self, subscriber_id):
        row = self.connection.execute(
            "SELECT committed_seq FROM progress WHERE subscriber_id=?",
            (subscriber_id,)).fetchone()
        return row["committed_seq"] if row else None

    def max_committed_seq(self, subscriber_id):
        row = self.connection.execute(
            "SELECT MAX(event_seq) AS max_seq FROM transitions WHERE subscriber_id=?"
            " AND to_state='committed' AND complete=1",
            (subscriber_id,)).fetchone()
        return row["max_seq"] if row and row["max_seq"] is not None else None

    def committed_pairs(self):
        rows = self.connection.execute(
            "SELECT subscriber_id, source_id, event_seq FROM transitions"
            " WHERE to_state='committed' AND complete=1").fetchall()
        pairs = {}
        for row in rows:
            pairs.setdefault(row["subscriber_id"], set()).add(
                (row["source_id"], row["event_seq"]))
        return pairs

    def transitions_for(self, subscriber_id, *, effective_only=True):
        sql = ("SELECT id, subscriber_id, source_id, event_seq, from_state, to_state,"
               " clock_ms, ledger_version, complete FROM transitions"
               " WHERE subscriber_id=?")
        if effective_only:
            sql += " AND complete=1"
        sql += " ORDER BY id"
        return self.connection.execute(sql, (subscriber_id,)).fetchall()

    def all_transitions(self):
        return self.connection.execute(
            "SELECT id, subscriber_id, source_id, event_seq, from_state, to_state,"
            " clock_ms, ledger_version, complete FROM transitions ORDER BY id").fetchall()

    def incomplete_rows(self):
        return self.connection.execute(
            "SELECT id, subscriber_id, source_id, event_seq, from_state, to_state"
            " FROM transitions WHERE complete=0 ORDER BY id").fetchall()

    def dangling_processing_rows(self):
        """Complete processing rows that have no persisted outcome yet."""
        return self.connection.execute(
            "SELECT t.id, t.subscriber_id, t.source_id, t.event_seq FROM transitions t"
            " WHERE t.to_state='processing' AND t.complete=1 AND NOT EXISTS ("
            "   SELECT 1 FROM transitions o WHERE o.subscriber_id=t.subscriber_id"
            "   AND o.source_id=t.source_id AND o.event_seq=t.event_seq"
            "   AND o.to_state IN ('committed','failed') AND o.complete=1"
            "   AND o.id > t.id)"
            " ORDER BY t.id").fetchall()

    def mark_incomplete(self, row_id):
        with self.connection:
            self.connection.execute(
                "UPDATE transitions SET complete=0 WHERE id=?", (row_id,))

    def subscriber_ids(self):
        rows = self.connection.execute(
            "SELECT DISTINCT subscriber_id FROM transitions ORDER BY subscriber_id"
        ).fetchall()
        return [row["subscriber_id"] for row in rows]

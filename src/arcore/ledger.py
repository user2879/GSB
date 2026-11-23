"""SQLite ledger with format versioning, wholesale replacement and replay."""

import sqlite3
from pathlib import Path

from .canonical import canonical_json, digest_bytes
from .errors import LedgerError

FORMAT_VERSION = 1

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS ledger_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rounds (
        round_id INTEGER PRIMARY KEY,
        status TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        format_version INTEGER NOT NULL,
        committed_at_ms INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS round_stats (
        round_id INTEGER PRIMARY KEY,
        stats_json TEXT NOT NULL,
        FOREIGN KEY(round_id) REFERENCES rounds(round_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS replay_diffs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        round_id INTEGER NOT NULL,
        field_path TEXT NOT NULL,
        ledger_value TEXT,
        replayed_value TEXT
    )
    """,
)

# Timing fields depend on the local monotonic clock and are never expected to
# match across processes; replay compares only deterministic outputs.
_REPLAY_EXCLUDED_KEYS = (
    "wait_start_ms", "wait_end_ms", "wait_ms",
    "run_start_ms", "run_end_ms", "granted_ms", "expires_ms",
    "reaped_ms", "at_ms", "committed_at_ms", "average_queue_ms_q",
)

# Whole subtrees that are genuine runtime artifacts rather than reproducible
# algorithm outputs: thread interleaving decides lease ids, event ordering and
# occupancy peaks, so they never participate in replay equality.
_REPLAY_EXCLUDED_SUBTREES = ("lease_occupancy", "batch_observations")


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _meta(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM ledger_meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _initialize(connection: sqlite3.Connection) -> None:
    for statement in _DDL:
        connection.execute(statement)
    connection.execute(
        "INSERT OR REPLACE INTO ledger_meta(key, value) VALUES(?, ?)",
        ("format_version", str(FORMAT_VERSION)))
    connection.commit()


def open_ledger(path: str | Path) -> tuple[sqlite3.Connection, list[str]]:
    """Open the ledger, replacing it wholesale on version/digest mismatch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    if path.exists() and path.stat().st_size > 0:
        try:
            connection = _connect(path)
            stored_version = _meta(connection, "format_version")
            if stored_version != str(FORMAT_VERSION):
                raise LedgerError(
                    f"format version mismatch: ledger={stored_version}, "
                    f"runtime={FORMAT_VERSION}")
            rows = connection.execute(
                "SELECT round_id, content_digest, payload_json "
                "FROM rounds ORDER BY round_id").fetchall()
            for round_id, stored_digest, payload_json in rows:
                actual = digest_bytes(payload_json.encode("utf-8"))
                if actual != stored_digest:
                    raise LedgerError(
                        f"content digest mismatch at round {round_id}")
            notes.append(f"validated {len(rows)} committed round(s)")
            return connection, notes
        except (sqlite3.DatabaseError, LedgerError) as exc:
            connection.close()
            # Wholesale replacement, never incremental merging.
            path.unlink()
            notes.append(f"ledger replaced wholesale due to: {exc}")
    connection = _connect(path)
    _initialize(connection)
    notes.append("initialized new ledger")
    return connection, notes


def commit_round(connection: sqlite3.Connection, round_id: int,
                 payload: dict, stats: dict, committed_at_ms: int) -> str:
    payload_bytes = canonical_json(payload)
    content_digest = digest_bytes(payload_bytes)
    with connection:
        connection.execute("DELETE FROM replay_diffs WHERE round_id = ?",
                           (round_id,))
        connection.execute("DELETE FROM round_stats WHERE round_id = ?",
                           (round_id,))
        connection.execute("DELETE FROM rounds WHERE round_id = ?",
                           (round_id,))
        connection.execute(
            "INSERT INTO rounds(round_id, status, payload_json, "
            "content_digest, format_version, committed_at_ms) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (round_id, "committed", payload_bytes.decode("utf-8"),
             content_digest, FORMAT_VERSION, committed_at_ms))
        connection.execute(
            "INSERT INTO round_stats(round_id, stats_json) VALUES(?, ?)",
            (round_id, canonical_json(stats).decode("utf-8")))
    return content_digest


def mark_round_unknown(connection: sqlite3.Connection, round_id: int) -> None:
    """Record an unfinished round after a hard kill; no exit report is produced."""
    placeholder = canonical_json({
        "round_id": round_id,
        "status": "unknown",
        "note": "process terminated before commit; round must be rerun",
    }).decode("utf-8")
    with connection:
        connection.execute(
            "INSERT INTO rounds(round_id, status, payload_json, "
            "content_digest, format_version, committed_at_ms) "
            "VALUES(?, ?, ?, ?, ?, NULL)",
            (round_id, "unknown", placeholder,
             digest_bytes(placeholder.encode("utf-8")), FORMAT_VERSION))


def load_committed_rounds(connection: sqlite3.Connection) -> list[dict]:
    rows = connection.execute(
        "SELECT payload_json FROM rounds WHERE status = 'committed' "
        "ORDER BY round_id").fetchall()
    import json
    return [json.loads(row[0]) for row in rows]


def unknown_round_ids(connection: sqlite3.Connection) -> list[int]:
    rows = connection.execute(
        "SELECT round_id FROM rounds WHERE status = 'unknown' "
        "ORDER BY round_id").fetchall()
    return [row[0] for row in rows]


def last_committed_round_id(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT COALESCE(MAX(round_id), 0) FROM rounds "
        "WHERE status = 'committed'").fetchone()
    return int(row[0])


def _strip_excluded(value, key_name: str = ""):
    if isinstance(value, dict):
        return {
            key: _strip_excluded(item, key)
            for key, item in value.items()
            if key not in _REPLAY_EXCLUDED_KEYS
            and key not in _REPLAY_EXCLUDED_SUBTREES
        }
    if isinstance(value, list):
        return [_strip_excluded(item, key_name) for item in value]
    return value


def _diff_values(path: str, ledger_value, replayed_value, diffs: list) -> None:
    if isinstance(ledger_value, dict) and isinstance(replayed_value, dict):
        keys = sorted(set(ledger_value) | set(replayed_value))
        for key in keys:
            _diff_values(f"{path}.{key}" if path else key,
                         ledger_value.get(key), replayed_value.get(key), diffs)
    elif isinstance(ledger_value, list) and isinstance(replayed_value, list):
        if len(ledger_value) != len(replayed_value):
            diffs.append((path, ledger_value, replayed_value))
            return
        for index, (left, right) in enumerate(
                zip(ledger_value, replayed_value)):
            _diff_values(f"{path}[{index}]", left, right, diffs)
    elif ledger_value != replayed_value:
        diffs.append((path, ledger_value, replayed_value))


def replay_rounds(connection: sqlite3.Connection,
                  replayed_rounds: list[dict]) -> list[dict]:
    """Replay committed rounds and persist field-by-field comparison diffs."""
    import json
    stored_rows = connection.execute(
        "SELECT round_id, payload_json FROM rounds "
        "WHERE status = 'committed' ORDER BY round_id").fetchall()
    all_diffs: list[dict] = []
    replayed_by_id = {payload["round_id"]: payload
                      for payload in replayed_rounds}
    for round_id, payload_json in stored_rows:
        stored = json.loads(payload_json)
        replayed = replayed_by_id.get(round_id)
        if replayed is None:
            diffs = [("round", "present", "missing")]
        else:
            diffs = []
            _diff_values("", _strip_excluded(stored),
                         _strip_excluded(replayed), diffs)
        with connection:
            for field_path, left, right in diffs:
                connection.execute(
                    "INSERT INTO replay_diffs(round_id, field_path, "
                    "ledger_value, replayed_value) VALUES(?, ?, ?, ?)",
                    (round_id, field_path,
                     json.dumps(left, sort_keys=True),
                     json.dumps(right, sort_keys=True)))
                all_diffs.append({
                    "round_id": round_id,
                    "field_path": field_path,
                    "ledger_value": left,
                    "replayed_value": right,
                })
    return all_diffs

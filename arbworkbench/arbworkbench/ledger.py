"""SQLite arbitration ledger with transactional crash recovery.

Durability contract
-------------------
Each view conclusion is written inside one ``BEGIN IMMEDIATE`` transaction that
inserts the conclusion row and only then inserts the ``committed`` marker. A
hard process kill leaves either both rows or neither. On reopen, any conclusion
without a matching committed marker is reported as ``unknown`` and must be
re-run; the ledger itself deletes the orphan row during recovery.

The ledger header stores an integer format version and a rolling content
digest. Reopening a ledger whose format version or digest does not match raises
``E_VERSION`` rather than silently mixing incompatible records.
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path

from .canonical import canonical_json, content_digest
from .errors import VersionError
from .engine import KeyConclusion, ViewConclusion

LEDGER_FORMAT_VERSION = 1
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ledger_meta (
    meta_key TEXT PRIMARY KEY,
    meta_value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS view_conclusions (
    view_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    conclusion_digest TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_order INTEGER NOT NULL,
    PRIMARY KEY (view_id, version)
);
CREATE TABLE IF NOT EXISTS committed_views (
    view_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    conclusion_digest TEXT NOT NULL,
    committed_order INTEGER NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA_SQL)
        self._init_meta()
        self._recover_orphans()

    # ------------------------------------------------------------------
    def _init_meta(self) -> None:
        row = self.conn.execute(
            "SELECT meta_value FROM ledger_meta WHERE meta_key='format_version'"
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO ledger_meta(meta_key, meta_value) VALUES('format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
            self.conn.execute(
                "INSERT INTO ledger_meta(meta_key, meta_value) VALUES('content_digest', ?)",
                (self._compute_digest(),),
            )
            self.conn.commit()
        else:
            if int(row[0]) != LEDGER_FORMAT_VERSION:
                raise VersionError(
                    f"ledger format version {row[0]} != supported {LEDGER_FORMAT_VERSION}",
                    location="ledger_meta.format_version",
                )
            stored = self.conn.execute(
                "SELECT meta_value FROM ledger_meta WHERE meta_key='content_digest'"
            ).fetchone()[0]
            actual = self._compute_digest()
            if stored != actual:
                raise VersionError(
                    "ledger content digest mismatch; refusing to mix state",
                    location="ledger_meta.content_digest",
                    detail={"stored": stored, "actual": actual},
                )

    def _compute_digest(self) -> str:
        rows = self.conn.execute(
            "SELECT view_id, version, conclusion_digest, payload "
            "FROM view_conclusions ORDER BY view_id, version"
        ).fetchall()
        return content_digest([{"view_id": r[0], "version": r[1], "digest": r[2]} for r in rows])

    def _recover_orphans(self) -> list[str]:
        """Delete conclusion rows that never reached the committed marker."""
        orphans = self.conn.execute(
            "SELECT vc.view_id, vc.version FROM view_conclusions vc "
            "LEFT JOIN committed_views cv "
            "ON vc.view_id=cv.view_id AND vc.version=cv.version "
            "WHERE cv.view_id IS NULL"
        ).fetchall()
        for view_id, version in orphans:
            self.conn.execute(
                "DELETE FROM view_conclusions WHERE view_id=? AND version=?",
                (view_id, version),
            )
        if orphans:
            self.conn.commit()
        return [view_id for view_id, _ in orphans]

    # ------------------------------------------------------------------
    def _next_order(self, table: str) -> int:
        row = self.conn.execute(
            f"SELECT COALESCE(MAX(created_order), 0) FROM {table}"
        ).fetchone()
        return (row[0] or 0) + 1

    def write_view(self, conclusion: ViewConclusion) -> None:
        """Atomically persist one view conclusion (single transaction)."""
        payload = canonical_json(
            {
                "view_id": conclusion.view_id,
                "version": conclusion.version,
                "added_keys": conclusion.added_keys,
                "removed_keys": conclusion.removed_keys,
                "changed_keys": conclusion.changed_keys,
                "keys": {key: asdict(kc) for key, kc in sorted(conclusion.keys.items())},
            }
        ).decode("utf-8")
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            # Whole-view replacement: retire every older version of this view.
            self.conn.execute(
                "DELETE FROM view_conclusions WHERE view_id=?", (conclusion.view_id,)
            )
            self.conn.execute(
                "DELETE FROM committed_views WHERE view_id=?", (conclusion.view_id,)
            )
            order = self._next_order("view_conclusions")
            self.conn.execute(
                "INSERT INTO view_conclusions(view_id, version, conclusion_digest, "
                "payload, created_order) VALUES(?, ?, ?, ?, ?)",
                (conclusion.view_id, conclusion.version,
                 conclusion.conclusion_digest, payload, order),
            )
            # A crash between this point and commit leaves an orphan row only;
            # recovery deletes it and the view remains 'unknown'.
            self.conn.execute(
                "INSERT INTO committed_views(view_id, version, conclusion_digest, "
                "committed_order) VALUES(?, ?, ?, ?)",
                (conclusion.view_id, conclusion.version,
                 conclusion.conclusion_digest, self._next_order_committed()),
            )
            self.conn.execute(
                "UPDATE ledger_meta SET meta_value=? WHERE meta_key='content_digest'",
                (self._compute_digest(),),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _next_order_committed(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(committed_order), 0) FROM committed_views"
        ).fetchone()
        return (row[0] or 0) + 1

    # ------------------------------------------------------------------
    def load_committed(self) -> dict[str, dict]:
        rows = self.conn.execute(
            "SELECT cv.view_id, cv.version, vc.payload, cv.conclusion_digest "
            "FROM committed_views cv JOIN view_conclusions vc "
            "ON vc.view_id=cv.view_id AND vc.version=cv.version "
            "ORDER BY cv.committed_order"
        ).fetchall()
        import json
        return {
            view_id: {
                "version": version,
                "payload": json.loads(payload),
                "digest": digest,
            }
            for view_id, version, payload, digest in rows
        }

    def active_views(self) -> set[str]:
        return {
            row[0] for row in self.conn.execute(
                "SELECT view_id FROM committed_views ORDER BY view_id"
            ).fetchall()
        }

    def unknown_views(self, expected_view_ids: set[str]) -> list[str]:
        """Views the current spec expects but have no committed record."""
        return sorted(expected_view_ids - self.active_views())

    def close(self) -> None:
        self.conn.close()

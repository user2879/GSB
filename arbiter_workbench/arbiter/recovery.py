"""Crash recovery and post-recovery replay validation.

Only fully committed ledger rows are trusted. Views whose latest commit row is
UNKNOWN (an interrupted transaction) are surfaced as requiring a full rerun.
The workbench never fabricates an exit report for a forced termination.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .errors import ErrorContext, ReplayError, StateError
from .ledger import COMMITTED, FORMAT_VERSION, Ledger, UNKNOWN, verify_replay


@dataclass(frozen=True)
class RecoveryReport:
    recovered_views: tuple[str, ...]
    unknown_views: tuple[str, ...]
    replay_checked_views: tuple[str, ...]
    meta_digest: str
    note: str


def inspect_db(db_path: str | Path) -> dict[str, dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT format_version, session_scale, content_digest FROM meta WHERE id=1"
        ).fetchone()
        if row is None or row["format_version"] != FORMAT_VERSION:
            raise StateError(
                "ledger meta missing or wrong format; snapshot replacement required",
                ErrorContext(str(db_path), "meta"),
            )
        rows = conn.execute(
            "SELECT vc.* FROM view_commit vc JOIN ("
            " SELECT view_id, MAX(version) AS mv FROM view_commit GROUP BY view_id"
            ") m ON m.view_id=vc.view_id AND m.mv=vc.version"
        ).fetchall()
        return {r["view_id"]: dict(r) for r in rows}
    finally:
        conn.close()


def recover(
    db_path: str | Path,
    session_scale: int,
    replay_fn,
) -> RecoveryReport:
    """Open the ledger and replay every committed view via ``replay_fn``.

    ``replay_fn(view_id, version) -> list[KeyRecord]`` must rebuild records from
    the deterministic inputs. UNKNOWN views are not replayed; they are returned
    for the caller to rerun.
    """
    latest = inspect_db(db_path)
    ledger = Ledger(db_path, session_scale=session_scale)
    recovered: list[str] = []
    unknown: list[str] = []
    replayed: list[str] = []
    try:
        for view_id in sorted(latest):
            row = latest[view_id]
            if row["status"] == UNKNOWN or not row["content_digest"]:
                unknown.append(view_id)
                continue
            version = row["version"]
            stored = ledger.load_records(view_id, version)
            if not stored:
                unknown.append(view_id)
                continue
            rebuilt = replay_fn(view_id, version)
            verify_replay(view_id, stored, rebuilt)
            recovered.append(view_id)
            replayed.append(view_id)
        report = RecoveryReport(
            recovered_views=tuple(recovered),
            unknown_views=tuple(unknown),
            replay_checked_views=tuple(replayed),
            meta_digest=ledger.meta_digest(),
            note=(
                "recovery uses only fully committed rows; UNKNOWN views require "
                "a full rerun and no termination exit report is generated"
            ),
        )
    finally:
        ledger.close()
    return report


def assert_no_fabricated_exit_report(report_path: str | Path) -> None:
    """A forced termination must not leave behind a claimed exit report."""
    path = Path(report_path)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("termination") == "forced":
            raise StateError(
                "forced termination must not produce an exit report",
                ErrorContext(str(report_path), "termination"),
            )

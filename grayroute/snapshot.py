"""Versioned session snapshots and forced-termination recovery."""
from __future__ import annotations

import json
from dataclasses import dataclass
from .canonical import content_hash
from .constants import SNAPSHOT_FORMAT_VERSION, UNKNOWN_BATCH
from .errors import SnapshotError


@dataclass(frozen=True)
class SessionSnapshot:
    format_version: int
    rule_set_version: int
    worker_set_version: int
    param_version_seq: int
    ledger_summary: str
    cursor: dict
    snapshot_summary: str

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "rule_set_version": self.rule_set_version,
            "worker_set_version": self.worker_set_version,
            "param_version_seq": self.param_version_seq,
            "ledger_summary": self.ledger_summary,
            "cursor": self.cursor,
            "snapshot_summary": self.snapshot_summary,
        }


def build_snapshot(*, rule_set_version: int, worker_set_version: int,
                   param_version_seq: int, ledger_summary: str,
                   cursor: dict) -> SessionSnapshot:
    body = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "rule_set_version": rule_set_version,
        "worker_set_version": worker_set_version,
        "param_version_seq": param_version_seq,
        "ledger_summary": ledger_summary,
        "cursor": cursor,
    }
    return SessionSnapshot(
        format_version=SNAPSHOT_FORMAT_VERSION,
        rule_set_version=rule_set_version,
        worker_set_version=worker_set_version,
        param_version_seq=param_version_seq,
        ledger_summary=ledger_summary,
        cursor=cursor,
        snapshot_summary=content_hash(body),
    )


def save_snapshot(path: str, snapshot: SessionSnapshot) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(snapshot.to_dict(), handle, ensure_ascii=False,
                  sort_keys=True, indent=2)


def load_snapshot(path: str) -> SessionSnapshot:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise SnapshotError(f"snapshot not found: {path}",
                            object_id=path, field="file")
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"invalid snapshot JSON: {exc.msg}",
                            object_id=path, field="file")
    stored = raw.get("snapshot_summary")
    body = {k: v for k, v in raw.items() if k != "snapshot_summary"}
    if stored != content_hash(body):
        raise SnapshotError("snapshot content summary mismatch",
                            object_id=path, field="snapshot_summary")
    return SessionSnapshot(
        format_version=raw["format_version"],
        rule_set_version=raw["rule_set_version"],
        worker_set_version=raw["worker_set_version"],
        param_version_seq=raw["param_version_seq"],
        ledger_summary=raw["ledger_summary"],
        cursor=raw["cursor"],
        snapshot_summary=stored,
    )


def restore_or_replace(path: str, expected: SessionSnapshot,
                       ) -> tuple[SessionSnapshot, str]:
    """Version-aware restore.

    Returns ``(snapshot, action)`` where action is ``restored`` when versions
    and summaries match, otherwise ``replaced`` (wholesale replacement, never
    incremental merge).
    """
    try:
        stored = load_snapshot(path)
    except SnapshotError:
        save_snapshot(path, expected)
        return expected, "replaced"
    same_versions = (
        stored.format_version == expected.format_version
        and stored.rule_set_version == expected.rule_set_version
        and stored.worker_set_version == expected.worker_set_version
        and stored.param_version_seq == expected.param_version_seq
    )
    if same_versions and stored.ledger_summary == expected.ledger_summary:
        return stored, "restored"
    save_snapshot(path, expected)
    return expected, "replaced"

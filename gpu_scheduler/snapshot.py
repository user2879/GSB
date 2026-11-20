"""Session snapshots: versioned full-state replacement, never incremental merge.

A snapshot captures the resource-pool version, task version, fixed-point
scale, current quota usage, every shard outcome and the ledger digest.  On
reopen, snapshot versions are compared to the current declaration versions;
any mismatch causes a *whole-state replacement* (the old snapshot is moved to
a read-only backup directory) rather than an incremental merge.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional

from .errors import E_LEDGER, Problem, SchedulerError
from .ledger import Ledger
from .model import (
    Declaration,
    content_digest,
)

SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOT_FILE = "snapshot.json"


@dataclass(frozen=True)
class Snapshot:
    format_version: int
    pool_version: str
    task_version: str
    config_version: str
    fixed_point_scale: int
    ledger_path: str
    ledger_digest: str
    last_round: int
    last_clock: int
    quota_state: List[dict]
    shard_states: List[dict]
    digest: str

    def to_dict(self) -> dict:
        payload = {
            "format_version": self.format_version,
            "pool_version": self.pool_version,
            "task_version": self.task_version,
            "config_version": self.config_version,
            "fixed_point_scale": self.fixed_point_scale,
            "ledger_path": self.ledger_path,
            "ledger_digest": self.ledger_digest,
            "last_round": self.last_round,
            "last_clock": self.last_clock,
            "quota_state": self.quota_state,
            "shard_states": self.shard_states,
        }
        payload["digest"] = content_digest(payload)
        return payload


def build_snapshot(declaration: Declaration, ledger: Ledger,
                   digest: str) -> Snapshot:
    cursor = ledger.last_cursor()
    last_round = cursor["last_round"] if cursor else 0
    last_clock = cursor["last_clock"] if cursor else 0
    latest_usage: Dict[str, dict] = {}
    for record in ledger.load_rounds():
        for usage in record.pool_usage:
            latest_usage[usage.pool_id] = usage.to_dict()
    shard_states = [
        {
            "task_id": shard.task_id,
            "shard_index": shard.shard_index,
            "state": shard.state,
            "version": shard.version,
            "output_digest": shard.output_digest,
            "pool_id": shard.pool_id,
            "started_ms": shard.started_ms,
            "ended_ms": shard.ended_ms,
            "wait_duration_ms": shard.wait_duration_ms,
            "attempts": shard.attempts,
        }
        for shard in declaration.all_shards()
    ]
    return Snapshot(
        format_version=SNAPSHOT_FORMAT_VERSION,
        pool_version=declaration.pool_version,
        task_version=declaration.task_version,
        config_version=declaration.config_version,
        fixed_point_scale=declaration.config.fixed_point_scale,
        ledger_path=os.path.abspath(ledger.path),
        ledger_digest=digest,
        last_round=last_round,
        last_clock=last_clock,
        quota_state=[latest_usage[pid] for pid in sorted(latest_usage)],
        shard_states=shard_states,
        digest="",
    )


def write_snapshot(snapshot: Snapshot, snapshot_dir: str) -> str:
    os.makedirs(snapshot_dir, exist_ok=True)
    path = os.path.join(snapshot_dir, SNAPSHOT_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(snapshot.to_dict(), handle, indent=2, sort_keys=True,
                  ensure_ascii=False)
    return os.path.abspath(path)


def read_snapshot(snapshot_dir: str) -> Optional[dict]:
    path = os.path.join(snapshot_dir, SNAPSHOT_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    stored = payload.pop("digest", None)
    if stored != content_digest(payload):
        raise SchedulerError(
            E_LEDGER, "snapshot digest mismatch",
            [Problem(path, "matching content digest", stored, E_LEDGER)])
    payload["digest"] = stored
    return payload


def versions_match(declaration: Declaration, snapshot: dict) -> bool:
    return snapshot.get("pool_version") == declaration.pool_version and \
        snapshot.get("task_version") == declaration.task_version and \
        int(snapshot.get("fixed_point_scale", -1)) == \
        declaration.config.fixed_point_scale


def replace_stale_snapshot(snapshot_dir: str, backup_dir: str) -> str:
    """Move an incompatible snapshot/ledger aside as a read-only backup."""
    os.makedirs(backup_dir, exist_ok=True)
    stamp = _index(backup_dir)
    target = os.path.join(backup_dir, f"snapshot_backup_{stamp}")
    shutil.move(snapshot_dir, target)
    _make_readonly(target)
    return os.path.abspath(target)


def _index(backup_dir: str) -> int:
    existing = [name for name in os.listdir(backup_dir)
                if name.startswith("snapshot_backup_")]
    return len(existing) + 1


def _make_readonly(path: str) -> None:
    for root, _, files in os.walk(path):
        os.chmod(root, 0o555)
        for name in files:
            os.chmod(os.path.join(root, name), 0o444)


def restore_shard_states(declaration: Declaration, snapshot: dict) -> None:
    """Apply committed shard outcomes back onto a fresh declaration.

    Successful shards regain their exact version and output digest so recovery
    only recomputes lost work; any shard recorded as running in the snapshot is
    treated as UNKNOWN by the caller once the ledger is consulted.
    """
    index = declaration.shard_index()
    for item in snapshot["shard_states"]:
        key = (item["task_id"], item["shard_index"])
        shard = index.get(key)
        if shard is None:
            raise SchedulerError(
                E_LEDGER, "snapshot references unknown shard",
                [Problem("snapshot.shard_states", "existing shard",
                         f"{key[0]}#{key[1]}", E_LEDGER)])
        shard.state = item["state"]
        shard.version = int(item["version"])
        shard.output_digest = item["output_digest"]
        shard.pool_id = item["pool_id"]
        shard.started_ms = item["started_ms"]
        shard.ended_ms = item["ended_ms"]
        shard.wait_duration_ms = int(item["wait_duration_ms"])
        shard.attempts = int(item["attempts"])

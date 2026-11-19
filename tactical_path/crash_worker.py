"""Subprocess helper that leaves uncommitted state then terminates abruptly."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .canonical import atomic_write_json
from .persistence import StateStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    store = StateStore(args.workspace)
    committed = store.load_committed()
    if committed is None:
        raise SystemExit("no committed snapshot to crash against")
    sequence = store.next_snapshot_sequence()
    snapshot_id = f"snap-{sequence:06d}"
    uncommitted = dict(committed)
    uncommitted["snapshot_id"] = snapshot_id
    uncommitted["commit_label"] = "uncommitted-crash-cache"
    if uncommitted.get("path_cache"):
        uncommitted["path_cache"][0]["route_hash"] = "0" * 64
    atomic_write_json(
        store.state / "snapshots" / snapshot_id / "snapshot.json",
        uncommitted,
    )
    atomic_write_json(
        store.pending / "tx-demo-crash.json",
        {
            "transaction": "tx-demo-crash",
            "snapshot_id": snapshot_id,
            "status": "prepared-not-committed",
        },
    )
    os._exit(91)


if __name__ == "__main__":
    main()


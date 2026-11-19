"""Interruptible synthesis worker used for forced-termination drills.

The worker is launched as an independent subprocess by the demo command and
can also be invoked directly via ``python -m syncpipe.cli worker ...``.
After every durable version it atomically refreshes a progress file, so the
orchestrator can wait until a chosen version exists on disk and then kill
the process with SIGTERM/TerminateProcess (no graceful shutdown handler runs).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .engine import Engine
from .parser import parse_stream
from .state import adjudicate_tick
from .storage import Storage


def _write_progress(engine: Engine, storage: Storage) -> None:
    storage.write_progress(
        "worker_progress.json",
        {
            "version": engine.version,
            "tick": engine.tick,
            "state_hash": engine.hash_chain[-1]["state_hash"],
        },
    )


def run_worker(
    root: str,
    stream_path: str,
    seed: int,
    stop_after_version: Optional[int] = None,
    clean_shutdown: bool = True,
) -> Dict[str, Any]:
    storage = Storage(root)
    with open(stream_path, "rb") as handle:
        lines = [line.decode("utf-8") for line in handle.read().splitlines() if line.strip()]
    tick_groups, parser_errors, _quarantined = parse_stream(lines)

    engine = Engine(storage, seed)
    engine.initialize()
    for record in parser_errors:
        storage.append_error(record)
        engine.errors.append(record)

    worklist: Dict[Tuple[str, int], Dict[str, Any]] = {}
    conflict_by_tick: Dict[int, List[Dict[str, Any]]] = {}
    commit_order = lambda e: (e["tick"], e["peer_id"], e["local_seq"])

    for group in tick_groups:
        conflict_records = adjudicate_tick(group)
        engine.conflicts.extend(conflict_records)
        conflict_by_tick[group[0]["tick"]] = conflict_records
        for event in group:
            worklist[(event["peer_id"], event["local_seq"])] = event

        from syncpipe.errors import make_error as _make_error, E_BASE_VERSION as _EBV
        while worklist:
            ready = [
                event for event in worklist.values()
                if event["base_snapshot_version"] == engine.version
            ]
            if ready:
                event = min(ready, key=commit_order)
            else:
                lagging = [
                    e for e in worklist.values()
                    if e["base_snapshot_version"] < engine.version
                ]
                pool = lagging or list(worklist.values())
                event = min(pool, key=commit_order)
                record = _make_error(
                    _EBV,
                    f"base drift driver: base={event['base_snapshot_version']} "
                    f"head={engine.version}",
                    version=engine.version,
                    tick=event["tick"],
                    event_loc=(event["peer_id"], event["local_seq"]),
                )
                engine.errors.append(record)
                storage.append_error(record)
            del worklist[(event["peer_id"], event["local_seq"])]
            engine._handle_event(event, conflict_by_tick.get(event["tick"], []))
            _write_progress(engine, storage)
            if stop_after_version is not None and engine.version >= stop_after_version:
                engine.suspended = list(worklist.values())
                engine._suspended_keys = set(worklist)
                storage.write_manifest(engine.manifest)
                storage.close_log()
                return {
                    "stopped_early": True,
                    "version": engine.version,
                    "state_hash": engine.hash_chain[-1]["state_hash"],
                    "pid": os.getpid(),
                }
        storage.write_manifest(engine.manifest)

    engine.suspended = list(worklist.values())
    engine._suspended_keys = set(worklist)
    head_snapshot = engine._ensure_head_snapshot()
    storage.write_manifest(engine.manifest)
    storage.close_log()
    if clean_shutdown:
        engine.write_shutdown()
    return {
        "stopped_early": False,
        "version": engine.version,
        "state_hash": head_snapshot["state_hash"],
        "pid": os.getpid(),
    }

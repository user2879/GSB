from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .engine import Engine
from .errors import SpecError
from .ledger import Ledger
from .projection import build_projection, compare_projections
from .spec import Config, Event, group_batches, load_events


def copy_ledger(source_path: str | Path, destination_path: str | Path) -> None:
    source = sqlite3.connect(str(source_path))
    destination = sqlite3.connect(str(destination_path))
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()


def prefix_snapshot_bytes(ledger: Ledger, checkpoint_seq: int) -> list[tuple[int, str]]:
    rows = ledger.connection.execute(
        "SELECT batch_id, snapshot_json FROM state_snapshots WHERE last_seq<=? ORDER BY batch_id",
        (checkpoint_seq,),
    ).fetchall()
    return [(int(row["batch_id"]), str(row["snapshot_json"])) for row in rows]


def verify_incremental(
    config: Config,
    events: tuple[Event, ...],
    *,
    checkpoint_seq: int,
    work_dir: str | Path,
) -> dict[str, Any]:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    base_path = work / "base.sqlite"
    incremental_path = work / "incremental.sqlite"
    full_path = work / "full.sqlite"
    for path in (base_path, incremental_path, full_path):
        if path.exists():
            path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(path) + suffix)
            if sidecar.exists():
                sidecar.unlink()

    batches = group_batches(events)
    checkpoint_batches = [batch for batch in batches if batch.events[-1].seq <= checkpoint_seq]
    if not checkpoint_batches:
        raise SpecError("checkpoint_seq does not include any complete batch", "checkpoint_seq")
    if checkpoint_batches[-1].events[-1].seq != checkpoint_seq:
        raise SpecError("checkpoint_seq must be the last sequence of a batch", "checkpoint_seq")
    prefix_count = checkpoint_batches[-1].events[-1].ordinal
    suffix_count = len(events) - prefix_count

    base_ledger = Ledger(base_path, config)
    try:
        base_engine = Engine(config, events, base_ledger, work / "nodes_base")
        base_engine.run(until_seq=checkpoint_seq)
        base_prefix_hashes = prefix_snapshot_bytes(base_ledger, checkpoint_seq)
        base_progress = base_ledger.progress()
    finally:
        base_ledger.close()

    copy_ledger(base_path, incremental_path)
    incremental_ledger = Ledger(incremental_path, config)
    try:
        incremental_engine = Engine(config, events, incremental_ledger, work / "nodes_incremental")
        incremental_report = incremental_engine.run(resume=True)
        if incremental_engine.processed_events != suffix_count:
            raise SpecError(
                f"incremental path processed {incremental_engine.processed_events} events, expected {suffix_count}",
                "incremental",
            )
        incremental_prefix_hashes = prefix_snapshot_bytes(incremental_ledger, checkpoint_seq)
        if base_prefix_hashes != incremental_prefix_hashes:
            raise SpecError("incremental path modified committed prefix snapshots", "incremental")
        incremental_projection = build_projection(incremental_ledger, config)
        incremental_progress = incremental_ledger.progress()
        incremental_content_digest = incremental_ledger.meta("content_digest")
    finally:
        incremental_ledger.close()

    full_ledger = Ledger(full_path, config)
    try:
        full_engine = Engine(config, events, full_ledger, work / "nodes_full")
        full_report = full_engine.run()
        full_projection = build_projection(full_ledger, config)
        full_progress = full_ledger.progress()
        full_content_digest = full_ledger.meta("content_digest")
    finally:
        full_ledger.close()

    compare_projections(incremental_projection, full_projection, config)
    if incremental_content_digest != full_content_digest:
        from .errors import DivergenceError

        raise DivergenceError("*", "*", "content_digest", "incremental and full content digests differ")
    result = {
        "ok": True,
        "checkpoint_seq": checkpoint_seq,
        "prefix_events": prefix_count,
        "suffix_events": suffix_count,
        "incremental_processed_events": incremental_engine.processed_events,
        "full_processed_events": full_engine.processed_events,
        "prefix_snapshots_byte_identical": True,
        "progress_equal": {
            "last_ordinal": int(incremental_progress["last_ordinal"])
            == int(full_progress["last_ordinal"]),
            "last_seq": int(incremental_progress["last_seq"]) == int(full_progress["last_seq"]),
        },
        "projection_digest": _digest_projection(incremental_projection),
        "content_digest": incremental_content_digest,
        "content_digest_equal": True,
        "incremental": {
            "ledger": str(incremental_path),
            "report": incremental_report,
        },
        "full": {
            "ledger": str(full_path),
            "report": full_report,
        },
        "base_progress": {
            "last_ordinal": int(base_progress["last_ordinal"]),
            "last_seq": int(base_progress["last_seq"]),
        },
    }
    if result["progress_equal"]["last_ordinal"] is False or result["progress_equal"]["last_seq"] is False:
        from .errors import DivergenceError

        raise DivergenceError("*", "*", "progress", "incremental and full progress differ")
    return result


def _digest_projection(projection: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(projection).encode("utf-8")).hexdigest()

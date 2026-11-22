"""Input-only anomaly injection helpers.

No production source path is mutated by these functions; they create separate
input or snapshot files (or a copied ledger) used by the CLI/demo.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .content import snapshot_digest
from .sample_data import duplicate_position_payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def inject_corrupt_batch(workdir: str | os.PathLike[str]) -> str:
    path = Path(workdir) / "anomaly" / "corrupt-batch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('{"format_version": 1, "transactions": [{"transaction_id": "broken"\n')
    return str(path)


def inject_duplicate_position(workdir: str | os.PathLike[str]) -> str:
    path = Path(workdir) / "anomaly" / "duplicate-position.json"
    _write_json(path, duplicate_position_payload())
    return str(path)


def inject_snapshot_digest(
    snapshot_path: str | os.PathLike[str],
    workdir: str | os.PathLike[str],
) -> str:
    source = Path(snapshot_path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["ledger_digest"] = "sha256:" + "0" * 64
    payload["content_digest"] = snapshot_digest(payload)
    destination = Path(workdir) / "anomaly" / "snapshot-digest-mismatch.json"
    _write_json(destination, payload)
    return str(destination)


def inject_ledger_digest(
    ledger_path: str | os.PathLike[str],
    workdir: str | os.PathLike[str],
) -> str:
    """Create a copied ledger with a deliberately inconsistent content digest."""

    import shutil
    import sqlite3

    destination = Path(workdir) / "anomaly" / "ledger-digest-mismatch.sqlite"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(ledger_path), destination)
    connection = sqlite3.connect(str(destination))
    try:
        connection.execute(
            "UPDATE meta SET value=? WHERE key='ledger_digest'",
            ("sha256:" + "0" * 64,),
        )
        connection.commit()
    finally:
        connection.close()
    return str(destination)

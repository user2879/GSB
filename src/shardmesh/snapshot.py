"""Whole-view JSON snapshots with digest verification and read-only backup."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from .canonical import sha256_hex
from .ledger import LEDGER_FORMAT_VERSION, Ledger


def write_snapshot(path: str | Path, view: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(view)
    payload["format_version"] = LEDGER_FORMAT_VERSION
    payload.pop("content_digest", None)
    payload["content_digest"] = sha256_hex(payload)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with temporary.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return payload


def read_verified_snapshot(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("content_digest", None)
    expected = sha256_hex(payload)
    payload["content_digest"] = digest
    if digest != expected:
        raise ValueError("snapshot content_digest mismatch")
    if payload.get("format_version") != LEDGER_FORMAT_VERSION:
        raise ValueError("snapshot format_version mismatch")
    return payload


def backup_readonly(path: Path) -> Path | None:
    if not path.exists():
        return None
    index = 1
    while True:
        backup = path.with_name(f"{path.name}.readonly-backup-{index}")
        if not backup.exists():
            break
        index += 1
    shutil.copy2(path, backup)
    os.chmod(backup, 0o444)
    return backup


def load_or_rebuild_snapshot(
    path: str | Path,
    ledger: Ledger,
    build_view: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], Path | None]:
    path = Path(path)
    expected_version = int(ledger.meta_get("snapshot_version", "0") or "0")
    valid = False
    payload: dict[str, Any] | None = None
    try:
        payload = read_verified_snapshot(path)
        valid = int(payload.get("snapshot_version", -1)) == expected_version
    except (OSError, ValueError, json.JSONDecodeError):
        valid = False

    backup = None
    if not valid:
        backup = backup_readonly(path)
        version = ledger.increment_snapshot_version()
        view = build_view()
        view["snapshot_version"] = version
        payload = write_snapshot(path, view)
    assert payload is not None
    return payload, backup

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .ledger import LEDGER_FORMAT_VERSION, Ledger
from .specs import ChainSpec, DiagnosticConfig


SNAPSHOT_FORMAT_VERSION = 1


def build_snapshot(ledger: Ledger, chain: ChainSpec,
                   diagnostic: DiagnosticConfig) -> dict[str, Any]:
    return {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "ledger_format_version": LEDGER_FORMAT_VERSION,
        "hop_set_version": chain.hop_set_version,
        "scale_caliber_version": diagnostic.scale_caliber_version,
        "session_scale": diagnostic.session_scale,
        "content_digest": ledger.get_meta("content_digest"),
        "verified_digest": ledger.verify_content_digest(),
        "cursor": ledger.cursor(),
        "requests": ledger.all_results(),
    }


def write_snapshot_atomic(path: str | Path, snapshot: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json(snapshot))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _readonly_backup(path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    candidate = path.with_name(f"{path.name}.{timestamp}.readonly.bak")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{timestamp}-{counter}.readonly.bak")
        counter += 1
    shutil.copy2(path, candidate)
    os.chmod(candidate, 0o444)
    return candidate


def open_or_replace_snapshot(
    path: str | Path,
    ledger: Ledger,
    chain: ChainSpec,
    diagnostic: DiagnosticConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open a compatible snapshot or replace it wholesale from the ledger.

    Returns ``(snapshot, replacement_report)``. A mismatching file is copied
    to a mode-0444 backup before the atomic whole-file replacement.
    """

    path = Path(path)
    current = build_snapshot(ledger, chain, diagnostic)
    if not path.exists():
        write_snapshot_atomic(path, current)
        return current, {"replaced": False, "reason": "created", "backup": None}

    old = _read_snapshot(path)
    reasons: list[str] = []
    if old is None:
        reasons.append("snapshot_json_invalid")
    else:
        if old.get("hop_set_version") != chain.hop_set_version:
            reasons.append("hop_set_version_mismatch")
        if old.get("scale_caliber_version") != diagnostic.scale_caliber_version:
            reasons.append("scale_caliber_version_mismatch")
        if old.get("session_scale") != diagnostic.session_scale:
            reasons.append("session_scale_mismatch")
        if old.get("content_digest") != current["content_digest"]:
            reasons.append("content_digest_mismatch")
        if old.get("ledger_format_version") != LEDGER_FORMAT_VERSION:
            reasons.append("ledger_format_version_mismatch")

    if not reasons:
        return old, {"replaced": False, "reason": "compatible", "backup": None}

    backup = _readonly_backup(path)
    write_snapshot_atomic(path, current)
    return current, {
        "replaced": True,
        "reason": "+".join(reasons),
        "reasons": reasons,
        "backup": str(backup),
        "backup_mode": oct(backup.stat().st_mode & 0o777),
    }

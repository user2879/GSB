"""Atomic group publication for the report and result documents."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .errors import BusinessError


def _write_temp(target: Path, payload: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json_group(contents: dict[Path, dict[str, Any]]) -> None:
    """Stage all JSON files, then replace targets as one rollback-capable group."""

    temps: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []
    published = False
    try:
        for target, document in contents.items():
            payload = json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8") + b"\n"
            temps[target] = _write_temp(target, payload)
        for target in contents:
            if target.exists():
                backup = target.with_name(f".{target.name}.{os.getpid()}.bak")
                os.replace(target, backup)
                backups[target] = backup
        for target in contents:
            os.replace(temps[target], target)
            replaced.append(target)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
        published = True
    except Exception as exc:
        for target in contents:
            try:
                backup = backups.get(target)
                if backup is not None and backup.exists():
                    if target.exists():
                        target.unlink()
                    os.replace(backup, target)
                elif target in replaced and target.exists():
                    target.unlink()
            except OSError:
                pass
        import sys

        raise BusinessError(
            "E_WR",
            "unable to atomically publish output files; previous version retained where possible",
            {"reason": str(exc), "exception_type": type(exc).__name__},
        ) from sys.exc_info()[1]
    finally:
        for temporary in temps.values():
            temporary.unlink(missing_ok=True)
        if published:
            for backup in backups.values():
                backup.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, document: dict[str, Any]) -> None:
    atomic_write_json_group({Path(path): document})

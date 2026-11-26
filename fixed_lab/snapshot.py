"""Temporary-directory snapshots committed by content version hash."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .cache import write_json_atomic
from .hashing import canonical_sha256


def commit_snapshot(
    snapshot_root: str | Path,
    payloads: dict[str, Any],
    *,
    version_hash: str | None = None,
) -> dict[str, str]:
    root = Path(snapshot_root)
    root.mkdir(parents=True, exist_ok=True)
    version = version_hash or canonical_sha256(payloads)
    final_dir = root / version
    temp_dir = Path(tempfile.mkdtemp(prefix=f".tmp-{version[:12]}-", dir=root))
    try:
        for name, payload in payloads.items():
            target = temp_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        if final_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            os.replace(temp_dir, final_dir)
        pointer = root / "current.json"
        write_json_atomic(pointer, {"version_hash": version, "path": str(Path(version))})
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    return {"version_hash": version, "path": str(Path(snapshot_root.name) / version), "pointer": str(Path(snapshot_root.name) / "current.json")}

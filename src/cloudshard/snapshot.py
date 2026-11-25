from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from . import FORMAT_VERSION
from .canonical import canonical_text, content_digest
from .ledger import Ledger
from .spec import Spec


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_text(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return path


def build_snapshot(spec: Spec, ledger: Ledger, *, revision: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    view_rows = ledger.conn.execute(
        """
        SELECT shard_id, revision, attempt, outcome_key, summary_digest, outcome_canonical, content_digest
        FROM current_view ORDER BY shard_id
        """
    ).fetchall()
    view = [
        {
            "shard_id": row["shard_id"],
            "revision": int(row["revision"]),
            "attempt": int(row["attempt"]),
            "outcome_key": row["outcome_key"],
            "summary_digest": row["summary_digest"],
            "outcome_canonical": row["outcome_canonical"],
            "row_content_digest": row["content_digest"],
        }
        for row in view_rows
    ]
    snapshot: dict[str, Any] = {
        "snapshot_format_version": FORMAT_VERSION,
        "ledger_format_version": FORMAT_VERSION,
        "revision": revision,
        "spec_digest": content_digest(spec.raw),
        "shard_count": len(spec.shards),
        "committed_count": len(view),
        "view": view,
        "ledger_digest": ledger.ledger_digest(),
    }
    if extra:
        snapshot.update(extra)
    snapshot["snapshot_digest"] = content_digest({k: v for k, v in snapshot.items() if k != "snapshot_digest"})
    return snapshot


def read_snapshot_version(path: str | Path) -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(data["snapshot_format_version"])
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported snapshot format version {version}")
    return version

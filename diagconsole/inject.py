"""Command-line fault injection helpers used by replay/demo tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .errors import canonical_json


def inject_corrupt_spec(revisions_path: str | Path,
                        out_dir: str | Path | None = None) -> str:
    """Break monotonic ordering -> E_SPEC on reload."""
    path = Path(revisions_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["revisions"][2]["version"] = 1
    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / (path.stem + "_corrupt.json")
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(target)


def inject_dangling_derives(revisions_path: str | Path,
                            out_dir: str | Path | None = None) -> str:
    """Point a derived reference at a non-existent earlier revision -> E_REF."""
    path = Path(revisions_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["revisions"][3]["derives"] = [999]
    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / (path.stem + "_dangling.json")
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(target)


def inject_snapshot_digest(snapshot_path: str | Path,
                           out_dir: str | Path | None = None) -> str:
    """Tamper with a snapshot node digest while keeping the outer digest stale."""
    path = Path(snapshot_path)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["nodes"][0]["content_digest"] = \
        "0" * len(snapshot["nodes"][0]["content_digest"])
    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / (path.stem + "_tampered.json")
    target.write_text(canonical_json(snapshot), encoding="utf-8")
    return str(target)

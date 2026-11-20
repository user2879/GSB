from __future__ import annotations

import json
import os
from pathlib import Path

from .model import canonical_bytes, canonical_json, content_hash


class CrashAbort(Exception):
    """Simulated forced termination: uncommitted tail must be discarded."""


def atomic_write(path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class MatchStore:
    """Atomic, recoverable on-disk state for one match.

    Commit order per step: snapshot file -> ledger append -> manifest bump.
    The manifest is the commit point; anything written before it without a
    manifest update is an uncommitted tail and is discarded on recovery.
    """

    def __init__(self, root, match_id: str):
        self.dir = Path(root) / "matches" / match_id
        self.snap_dir = self.dir / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)

    def snap_path(self, step: int) -> Path:
        return self.snap_dir / f"step_{step:06d}.json"

    def read_manifest(self):
        path = self.dir / "manifest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def read_snapshot(self, step: int):
        return json.loads(self.snap_path(step).read_text(encoding="utf-8"))

    def committed_hashes(self):
        manifest = self.read_manifest()
        if not manifest:
            return []
        return [self.read_snapshot(i)["hash"] for i in range(manifest["committed"])]

    def commit(self, content, ledger_entry) -> str:
        step = content["step"]
        digest = content_hash(content)
        atomic_write(self.snap_path(step), canonical_bytes({"hash": digest, "content": content}))
        entry = dict(ledger_entry)
        entry["snapshot_hash"] = digest
        with open(self.dir / "ledger.jsonl", "a", encoding="utf-8") as fh:
            fh.write(canonical_json(entry) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        atomic_write(
            self.dir / "manifest.json",
            canonical_bytes({"committed": step + 1, "last_hash": digest}),
        )
        return digest

    def truncate_ledger(self, keep: int) -> None:
        path = self.dir / "ledger.jsonl"
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = lines[:keep]
        atomic_write(path, ("\n".join(kept) + ("\n" if kept else "")).encode("utf-8"))

    def leave_partial_artifact(self, step: int) -> None:
        (self.snap_dir / f"step_{step:06d}.json.tmp").write_bytes(b"PARTIAL-UNCOMMITTED\n")

    def close(self, result) -> None:
        atomic_write(self.dir / "result.json", canonical_bytes(result))
        atomic_write(self.dir / "closed.marker", b"CLOSED\n")

    def is_closed(self) -> bool:
        return (self.dir / "closed.marker").exists()

    def load_result(self):
        return json.loads((self.dir / "result.json").read_text(encoding="utf-8"))

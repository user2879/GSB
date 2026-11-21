"""Stage-wise atomic snapshots and resumable cache entries."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, sha256_json

SNAPSHOT_SCHEMA = "qcal-snapshot/1.0"


@dataclass
class Snapshot:
    manifest: dict[str, Any] = field(default_factory=dict)
    entries: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    @property
    def global_version(self) -> int:
        return int(self.manifest.get("global_version", 0))

    def copy(self) -> "Snapshot":
        return Snapshot(manifest=json.loads(json.dumps(self.manifest)), entries=json.loads(json.dumps(self.entries)))


class SnapshotStore:
    """Commits whole snapshot directories and swaps a single atomic pointer.

    Snapshot directories are never modified after publication. This is the
    directory-level analogue of an atomic file replace and also avoids partial
    per-file updates if the process is terminated.
    """

    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.snapshots_dir = self.state_dir / "snapshots"
        self.current_path = self.state_dir / "current.json"

    def load_current(self) -> Snapshot:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        if not self.current_path.exists():
            return Snapshot()
        try:
            pointer = json.loads(self.current_path.read_text(encoding="utf-8"))
            snapshot_dir = self.snapshots_dir / pointer["snapshot"]
            manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
            entries = json.loads((snapshot_dir / "entries.json").read_text(encoding="utf-8"))
            expected = manifest.get("entries_digest")
            actual = sha256_json(entries)
            if expected and expected != actual:
                raise ValueError("snapshot entries digest mismatch")
            if manifest.get("schema_version") != SNAPSHOT_SCHEMA:
                raise ValueError("unsupported snapshot schema")
            return Snapshot(manifest=manifest, entries=entries)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            # A corrupt/unpublished pointer must be treated as no resumable
            # state. The caller will start from a clean in-memory snapshot.
            return Snapshot()

    def commit_stage(
        self,
        snapshot: Snapshot,
        stage: str,
        *,
        config_hash: str,
        exchange_fingerprint: str,
        sidecar_hash: str,
        changed_stages: list[str],
    ) -> Snapshot:
        global_version = snapshot.global_version + 1
        completed: dict[str, list[str]] = {}
        for stage_name in sorted(snapshot.entries):
            completed[stage_name] = sorted(
                layer
                for layer, entry in snapshot.entries[stage_name].items()
                if entry.get("status") == "ok"
            )
        entries_payload = snapshot.entries
        entries_digest = sha256_json(entries_payload)
        manifest = {
            "schema_version": SNAPSHOT_SCHEMA,
            "global_version": global_version,
            "current_stage": stage,
            "config_hash": config_hash,
            "exchange_version_fingerprint": exchange_fingerprint,
            "sidecar_hash": sidecar_hash,
            "completed_layers": completed,
            "changed_stages": sorted(set(changed_stages)),
            "entries_digest": entries_digest,
        }
        manifest["manifest_digest"] = sha256_json(manifest)
        snapshot_id_base = f"s{global_version:06d}-{manifest['manifest_digest'][:16]}"
        final_dir = self.snapshots_dir / snapshot_id_base
        suffix = 0
        while final_dir.exists():
            suffix += 1
            final_dir = self.snapshots_dir / f"{snapshot_id_base}-{suffix}"
        temp_dir = Path(
            tempfile.mkdtemp(prefix=f".{final_dir.name}.", suffix=".tmp", dir=self.snapshots_dir)
        )
        try:
            atomic_write_json(temp_dir / "entries.json", entries_payload)
            atomic_write_json(temp_dir / "manifest.json", manifest)
            os.replace(temp_dir, final_dir)
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        pointer = {
            "schema_version": SNAPSHOT_SCHEMA,
            "snapshot": final_dir.name,
            "global_version": global_version,
            "manifest_digest": manifest["manifest_digest"],
        }
        atomic_write_json(self.current_path, pointer)
        snapshot.manifest = manifest
        return snapshot

    def snapshot_names(self) -> list[str]:
        if not self.snapshots_dir.exists():
            return []
        return sorted(
            path.name for path in self.snapshots_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
        )


def entries_digest(entries: dict[str, dict[str, dict[str, Any]]]) -> str:
    return sha256_json(entries)

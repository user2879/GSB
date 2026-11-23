from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, atomic_write_text


class SnapshotStore:
    """Immutable snapshot directories plus an atomically replaced latest pointer."""

    def __init__(self, root: str | Path, *, clean: bool = True):
        self.root = Path(root)
        if clean and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots = self.root / "snapshots"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict[str, Any]] = []
        self.latest: dict[str, Any] | None = None

    def commit(self, step: int, report: dict[str, Any]) -> Path:
        temp_dir = Path(tempfile.mkdtemp(prefix=".tmp-step-", dir=str(self.snapshots)))
        try:
            atomic_write_json(temp_dir / "report.json", report)
            pointwise_text = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in report.get("pointwise", [])
            )
            atomic_write_text(temp_dir / "pointwise.jsonl", pointwise_text)
            atomic_write_json(temp_dir / "failures.json", report.get("failures", []))
            atomic_write_json(
                temp_dir / "stage_manifest.json",
                {
                    "stage_keys": report.get("cache", {}).get("stage_keys", {}),
                    "ledger_signature": report.get("cache", {}).get("ledger_signature"),
                    "reused_count": report.get("cache", {}).get("reused_count"),
                    "recomputed_count": report.get("cache", {}).get("recomputed_count"),
                },
            )
            checkpoint_dir = temp_dir / "checkpoints"
            checkpoints = report.get("training", {}).get("checkpoints", [])
            if checkpoints:
                checkpoint_dir.mkdir()
                for index, checkpoint in enumerate(checkpoints, start=1):
                    atomic_write_json(
                        checkpoint_dir / f"checkpoint_{index:04d}.json",
                        checkpoint,
                    )
            digest = report["run"]["core_digest"].split(":", 1)[-1][:12]
            destination = self.snapshots / f"step_{step:04d}_{digest}"
            collision = 1
            while destination.exists():
                destination = self.snapshots / f"step_{step:04d}_{digest}_r{collision}"
                collision += 1
            os.replace(temp_dir, destination)
        except BaseException:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        entry = {
            "step": step,
            "snapshot": destination.name,
            "core_digest": report["run"]["core_digest"],
            "state_id": report["run"]["state_id"],
            "command_index": report["run"]["command_index"],
            "command_type": report["run"]["command_type"],
        }
        self.entries.append(entry)
        self.latest = entry
        atomic_write_json(self.root / "latest.json", entry)
        return destination

    def finalize(self, extra: dict[str, Any] | None = None) -> None:
        index = {
            "schema_version": "1.0",
            "entries": self.entries,
            "latest": self.latest,
        }
        if extra:
            index.update(extra)
        atomic_write_json(self.root / "index.json", index)

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RecoveryDecision:
    resume: bool
    reason: str
    manifest: dict[str, Any] | None = None
    snapshot_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SnapshotStore:
    """Versioned snapshots with staging directory and atomic current symlink."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current_link = self.root / "current"

    def cleanup_staging(self) -> list[str]:
        removed: list[str] = []
        for path in self.root.glob(".staging-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed.append(str(path))
        return removed

    def begin_staging(self) -> Path:
        self.cleanup_staging()
        return Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))

    def commit_report(self, report: dict[str, Any], *, config_hash: str, exchange_version: str,
                      pending: list[dict[str, Any]] | None = None,
                      consumed_budget: int | None = None) -> dict[str, Any]:
        staging = self.begin_staging()
        try:
            for name, key in (
                ("token_results.json", "token_results"),
                ("cache_records.json", "cache_records"),
                ("stage_summaries.json", "stage_summaries"),
                ("reuse_ledger.json", "reuse_ledger"),
                ("execution_order.json", "execution_order"),
                ("diagnostics.json", "diagnostics"),
                ("fault_report.json", "fault_report"),
            ):
                (staging / name).write_text(
                    json.dumps(report.get(key, []), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            scheduler = report.get("scheduler", {})
            (staging / "pending_queue.json").write_text(
                json.dumps(pending if pending is not None else scheduler.get("pending", []), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (staging / "consumed_budget.json").write_text(
                json.dumps(
                    {"consumed_budget": consumed_budget if consumed_budget is not None else scheduler.get("consumed_budget", 0),
                     "consumed_by_op": scheduler.get("consumed_by_op", {})},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (staging / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            previous = self.read_manifest()
            version = int((previous or {}).get("version", 0)) + 1
            manifest = {
                "schema_version": "snapshot-v1",
                "version": version,
                "config_hash": config_hash,
                "exchange_version": exchange_version,
                "status": report.get("status"),
                "pending_count": len(pending if pending is not None else scheduler.get("pending", [])),
                "consumed_budget": consumed_budget if consumed_budget is not None else scheduler.get("consumed_budget", 0),
                "files": [
                    "report.json",
                    "token_results.json",
                    "cache_records.json",
                    "stage_summaries.json",
                    "reuse_ledger.json",
                    "pending_queue.json",
                    "consumed_budget.json",
                ],
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            final_dir = self.root / f"v{version:05d}"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(staging, final_dir)

            tmp_link = self.root / ".current.tmp"
            if tmp_link.exists() or tmp_link.is_symlink():
                tmp_link.unlink()
            tmp_link.symlink_to(final_dir.name, target_is_directory=True)
            os.replace(tmp_link, self.current_link)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def read_manifest(self) -> dict[str, Any] | None:
        manifest_path = self.current_link / "manifest.json"
        if not manifest_path.exists():
            versions = sorted(self.root.glob("v*/manifest.json"))
            if not versions:
                return None
            return json.loads(versions[-1].read_text(encoding="utf-8"))
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def recover(self, *, config_hash: str, exchange_version: str) -> RecoveryDecision:
        removed = self.cleanup_staging()
        manifest = self.read_manifest()
        if manifest is None:
            return RecoveryDecision(False, "no_committed_snapshot", None, None)
        snapshot_dir = self.current_link.resolve() if self.current_link.exists() else None
        if manifest.get("config_hash") != config_hash:
            return RecoveryDecision(
                False,
                "config_hash_mismatch_rebuild",
                {**manifest, "cleaned_staging": removed},
                str(snapshot_dir) if snapshot_dir else None,
            )
        if manifest.get("exchange_version") != exchange_version:
            return RecoveryDecision(
                False,
                "exchange_version_mismatch_rebuild",
                {**manifest, "cleaned_staging": removed},
                str(snapshot_dir) if snapshot_dir else None,
            )
        return RecoveryDecision(
            True,
            "committed_snapshot_usable",
            {**manifest, "cleaned_staging": removed},
            str(snapshot_dir) if snapshot_dir else None,
        )


def simulate_hard_kill(staging_root: str | Path) -> None:
    """Create a partial staging directory and kill the calling process.

    Tests invoke this through a subprocess so ``SIGKILL`` is real rather than a
    simulated exception.
    """

    store = SnapshotStore(staging_root)
    staging = store.begin_staging()
    (staging / "partial_stage.json").write_text('{"stage":"embedding","partial":', encoding="utf-8")
    (staging / "pending_queue.json").write_text("[]\n", encoding="utf-8")
    os.kill(os.getpid(), 9)

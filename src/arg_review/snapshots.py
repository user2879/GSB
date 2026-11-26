from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .canonical import fingerprint


class SnapshotStore:
    STAGE_FILES = {
        "validation": "validation.json",
        "triggers": "triggers.json",
        "arguments": "arguments.json",
        "adjudication": "adjudication.json",
        "references": "references.json",
        "scheduler": "scheduler.json",
        "ledger": "ledger.json",
        "report": "diagnostic_report.json",
    }

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current_link = self.root / "current"

    def recover(self) -> dict[str, Any]:
        discarded: list[str] = []
        for path in self.root.glob("tmp-*"):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
                discarded.append(path.name)

        if self.current_link.exists() or self.current_link.is_symlink():
            commit_path = self.current_link / "commit.json"
            if commit_path.exists():
                try:
                    commit = json.loads(commit_path.read_text(encoding="utf-8"))
                    return {
                        "decision": "use_current_snapshot",
                        "current_snapshot": os.readlink(self.current_link)
                        if self.current_link.is_symlink()
                        else str(self.current_link),
                        "commit": commit,
                        "discarded_temp_dirs": sorted(discarded),
                    }
                except (json.JSONDecodeError, OSError):
                    pass
        return {
            "decision": "full_rebuild_required",
            "current_snapshot": None,
            "commit": None,
            "discarded_temp_dirs": sorted(discarded),
        }

    def commit(
        self,
        stage_outputs: dict[str, Any],
        global_version: int,
        config_hash: str,
        extra_commit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous = None
        previous_name = None
        if self.current_link.is_symlink():
            previous_name = os.readlink(self.current_link)
            previous_commit_path = self.current_link / "commit.json"
            if previous_commit_path.exists():
                previous = json.loads(previous_commit_path.read_text(encoding="utf-8"))

        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f"tmp-v{global_version}-", dir=self.root)
        )
        try:
            file_manifest: dict[str, str] = {}
            for stage, filename in self.STAGE_FILES.items():
                if stage not in stage_outputs:
                    continue
                payload = stage_outputs[stage]
                path = temporary_dir / filename
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                file_manifest[filename] = fingerprint(payload, 16)

            snapshot_fingerprint = fingerprint(
                {"version": global_version, "files": file_manifest, "config_hash": config_hash},
                16,
            )
            commit = {
                "snapshot_id": f"snap-v{global_version}-{snapshot_fingerprint}",
                "global_version": global_version,
                "config_hash": config_hash,
                "snapshot_fingerprint": snapshot_fingerprint,
                "files": file_manifest,
                "overall_replace": True,
            }
            if extra_commit:
                commit.update(extra_commit)
            (temporary_dir / "commit.json").write_text(
                json.dumps(commit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

            final_dir = self.root / commit["snapshot_id"]
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(temporary_dir, final_dir)

            temporary_link = self.root / f"current.new-{os.getpid()}"
            if temporary_link.exists() or temporary_link.is_symlink():
                temporary_link.unlink()
            os.symlink(final_dir.name, temporary_link)
            os.replace(temporary_link, self.current_link)

            return {
                "committed": True,
                "snapshot_id": commit["snapshot_id"],
                "path": str(final_dir),
                "current_link": str(self.current_link),
                "previous_snapshot": previous_name,
                "version_changed": not previous
                or previous.get("global_version") != global_version
                or previous.get("config_hash") != config_hash,
                "previous_global_version": (previous or {}).get("global_version"),
                "global_version": global_version,
                "config_hash": config_hash,
                "commit": commit,
            }
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)

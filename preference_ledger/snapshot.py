from __future__ import annotations

import json
import os
import signal
import tempfile
from pathlib import Path
from typing import Any

from .util import fingerprint


REQUIRED_FILES = (
    "manifest.json",
    "01_validate.json",
    "02_features.json",
    "03_scores.json",
    "04_windows.json",
    "05_partitions.json",
    "scheduler.json",
    "last_report.json",
)


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snapshots_dir = self.root / "snapshots"
        self.current_path = self.root / "current.json"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def load_committed(self) -> dict[str, Any] | None:
        if not self.current_path.exists():
            return None
        try:
            pointer = json.loads(self.current_path.read_text(encoding="utf-8"))
            snapshot_dir = Path(pointer["path"])
            if not snapshot_dir.is_absolute():
                snapshot_dir = self.root / snapshot_dir
            manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
            for name in REQUIRED_FILES:
                path = snapshot_dir / name
                if not path.exists():
                    return None
            state_doc = json.loads((snapshot_dir / "01_validate.json").read_text(encoding="utf-8"))
            features = json.loads((snapshot_dir / "02_features.json").read_text(encoding="utf-8"))
            scores = json.loads((snapshot_dir / "03_scores.json").read_text(encoding="utf-8"))
            windows = json.loads((snapshot_dir / "04_windows.json").read_text(encoding="utf-8"))
            partitions = json.loads((snapshot_dir / "05_partitions.json").read_text(encoding="utf-8"))
            scheduler = json.loads((snapshot_dir / "scheduler.json").read_text(encoding="utf-8"))
            report = json.loads((snapshot_dir / "last_report.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, KeyError):
            return None
        state_doc["cache"] = {
            "features": features,
            "scores": scores,
            "windows": windows,
            "partitions": partitions,
        }
        state_doc["scheduler_state"] = scheduler
        state_doc["last_report"] = report
        state_doc["manifest"] = manifest
        state_doc["snapshot_path"] = str(snapshot_dir)
        return state_doc

    def commit(
        self,
        raw_records: list[dict[str, Any]],
        raw_weights: dict[str, Any],
        raw_config: dict[str, Any],
        cache: dict[str, Any],
        scheduler_state: dict[str, Any],
        report: dict[str, Any],
        previous_version: int = 0,
        crash_after_stage: str | None = None,
    ) -> dict[str, Any]:
        config_hash = fingerprint(raw_config)
        input_hash = fingerprint({"records": raw_records, "weights": raw_weights})
        version = int(previous_version) + 1
        temp_dir = Path(tempfile.mkdtemp(prefix=f"txn-v{version:06d}-", dir=self.snapshots_dir))

        def write_json(name: str, value: Any) -> None:
            target = temp_dir / name
            with target.open("w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            if crash_after_stage == name:
                os.kill(os.getpid(), signal.SIGKILL)

        validate_doc = {
            "version": version,
            "raw_records": raw_records,
            "raw_weights": raw_weights,
            "raw_config": raw_config,
            "config_hash": config_hash,
            "input_hash": input_hash,
        }
        write_json("01_validate.json", validate_doc)
        if crash_after_stage == "01_validate":
            os.kill(os.getpid(), signal.SIGKILL)
        write_json("02_features.json", cache.get("features", {}))
        write_json("03_scores.json", cache.get("scores", {}))
        write_json("04_windows.json", cache.get("windows", {}))
        write_json("05_partitions.json", cache.get("partitions", {}))
        write_json("scheduler.json", scheduler_state)
        write_json("last_report.json", report)
        manifest = {
            "manifest_version": "snapshot-manifest/v1",
            "version": version,
            "config_version": raw_config.get("snapshot_config_version", raw_config.get("schema_version", "1.0")),
            "config_hash": config_hash,
            "input_hash": input_hash,
            "files": list(REQUIRED_FILES),
            "previous_version": previous_version,
        }
        write_json("manifest.json", manifest)

        final_dir = self.snapshots_dir / f"snap-v{version:06d}-{config_hash[:12]}"
        os.rename(temp_dir, final_dir)
        pointer = {"version": version, "config_hash": config_hash, "path": str(final_dir.resolve())}
        pointer_tmp = self.root / "current.json.tmp"
        pointer_tmp.write_text(json.dumps(pointer, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(pointer_tmp, self.current_path)
        return pointer

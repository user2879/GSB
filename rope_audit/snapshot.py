from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .hashing import content_hash


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.current_pointer = self.root / "current.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp-write")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp-write")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _read_pointer(self) -> dict[str, Any] | None:
        if not self.current_pointer.exists():
            return None
        try:
            with self.current_pointer.open("r", encoding="utf-8") as handle:
                pointer = json.load(handle)
            target = self.root / pointer["path"]
            manifest_path = target / "manifest.json"
            if not manifest_path.exists() or not (target / "COMMITTED").exists():
                return None
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            pointer["manifest"] = manifest
            pointer["target"] = str(target)
            return pointer
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def latest_committed(self) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for manifest_path in self.versions_dir.glob("*/manifest.json"):
            version_dir = manifest_path.parent
            if not (version_dir / "COMMITTED").exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            manifest["target"] = str(version_dir)
            candidates.append(manifest)
        if not candidates:
            return None
        return max(candidates, key=lambda item: int(item["global_version"]))

    def prepare_transaction(self, engine: Any, report: dict[str, Any] | None = None) -> Path:
        pointer = self._read_pointer()
        latest = self.latest_committed()
        global_version = int((pointer or {}).get("manifest", {}).get("global_version", 0) or 0)
        if latest:
            global_version = max(global_version, int(latest["global_version"]))
        global_version += 1

        temporary = Path(tempfile.mkdtemp(prefix="tmp-stage-", dir=self.root))
        state = engine.export_state()
        budget = {
            "pending_sample_ids": engine.pending,
            "consumed_total_points": engine.consumed_total_points,
            "last_batch_consumed_points": engine.last_batch_consumed_points,
        }
        manifest = {
            "global_version": global_version,
            "config_hash": engine.config_hash,
            "status": "prepared",
            "state_file": "state.json",
            "report_file": "report.json",
            "budget_file": "budget.json",
            "artifact_files": [
                "artifacts/position.jsonl",
                "artifacts/rotation.jsonl",
                "artifacts/dot.jsonl",
                "artifacts/normalization.jsonl",
                "artifacts/summary.jsonl",
            ],
        }
        self._write_json(temporary / "manifest.json", manifest)
        self._write_json(temporary / "state.json", state)
        self._write_json(temporary / "budget.json", budget)
        self._write_json(temporary / "report.json", report or engine.last_report or {})

        artifacts: dict[str, list[dict[str, Any]]] = {
            "position": [],
            "rotation": [],
            "dot": [],
            "normalization": [],
            "summary": [],
        }
        for envelope in engine.cache.values():
            stage = envelope.get("stage")
            if stage in artifacts:
                artifacts[stage].append(
                    {key: value for key, value in envelope.items() if key != "value"}
                    | {"value": envelope["value"]}
                )
        for stage, rows in artifacts.items():
            rows.sort(key=lambda item: (item.get("scheme_id") or "", item.get("sample_id") or "", item["cache_key"]))
            self._write_jsonl(temporary / "artifacts" / f"{stage}.jsonl", rows)

        ledger_rows = list(engine.ledger)
        self._write_jsonl(temporary / "ledger.jsonl", ledger_rows)
        return temporary

    def commit_prepared(self, temporary: Path) -> dict[str, Any]:
        manifest_path = temporary / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "committed"
        self._write_json(manifest_path, manifest)
        committed_marker = temporary / "COMMITTED"
        committed_marker.write_text("committed\n", encoding="utf-8")
        with committed_marker.open("r", encoding="utf-8") as handle:
            os.fsync(handle.fileno())

        version = int(manifest["global_version"])
        config_hash = str(manifest["config_hash"])
        target = self.versions_dir / f"v{version:06d}-{config_hash[:12]}"
        if target.exists():
            shutil.rmtree(target)
        os.rename(temporary, target)

        pointer = {"global_version": version, "config_hash": config_hash, "path": str(target.relative_to(self.root))}
        pointer_tmp = self.current_pointer.with_name("current.json.tmp-write")
        self._write_json(pointer_tmp, pointer)
        os.replace(pointer_tmp, self.current_pointer)
        return pointer | {"target": str(target), "manifest": manifest}

    def commit(self, engine: Any, report: dict[str, Any] | None = None) -> dict[str, Any]:
        temporary = self.prepare_transaction(engine, report)
        return self.commit_prepared(temporary)

    def recover(self, expected_config_hash: str | None = None) -> dict[str, Any]:
        orphaned: list[str] = []
        for temporary in self.root.glob("tmp-stage-*"):
            if temporary.is_dir():
                shutil.rmtree(temporary)
                orphaned.append(str(temporary))

        pointer = self._read_pointer()
        latest = self.latest_committed()
        selected = (pointer or {}).get("manifest") or latest
        if pointer is None and latest is not None:
            target = Path(latest["target"])
            replacement = {
                "global_version": latest["global_version"],
                "config_hash": latest["config_hash"],
                "path": str(target.relative_to(self.root)),
            }
            self._write_json(self.current_pointer, replacement)
            pointer = replacement | {"manifest": latest, "target": str(target)}
        elif pointer is not None and "target" in pointer:
            selected["target"] = pointer["target"]

        if selected is None:
            status = "empty"
        elif expected_config_hash and selected.get("config_hash") != expected_config_hash:
            status = "config_hash_mismatch_cold_rebuild"
        elif pointer is not None:
            status = "current_snapshot_ready"
        else:
            status = "latest_committed_ready"

        return {
            "status": status,
            "global_version": selected.get("global_version") if selected else None,
            "config_hash": selected.get("config_hash") if selected else None,
            "snapshot_dir": (pointer or {}).get("target") or (selected.get("target") if selected else None),
            "orphaned_temp_dirs_removed": orphaned,
            "current_pointer": str(self.current_pointer),
        }

    def load_state(self, expected_config_hash: str) -> dict[str, Any] | None:
        recovery = self.recover(expected_config_hash)
        if recovery["status"] != "current_snapshot_ready":
            return None
        state_path = Path(recovery["snapshot_dir"]) / "state.json"
        with state_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def hash_config(config: dict[str, Any]) -> str:
    return content_hash(config, 20)

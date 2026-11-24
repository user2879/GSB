"""Phase-partitioned, atomically replaced workbench snapshots."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .codes import BusinessFault, E_SNAPSHOT_CORRUPT, E_SNAPSHOT_IO


STAGE_FILES = {
    "load": "load.jsonl",
    "decay": "decay.jsonl",
    "relations": "relations.jsonl",
    "aggregation": "aggregation.jsonl",
}


class SnapshotStore:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.snapshots_dir = state_dir / "snapshots"
        self.current_path = state_dir / "current.json"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.last_version = self._discover_last_version()

    def _discover_last_version(self) -> int:
        versions: list[int] = []
        for path in self.snapshots_dir.iterdir():
            if path.is_dir() and path.name.startswith("v") and path.name[1:].isdigit():
                versions.append(int(path.name[1:]))
        return max(versions, default=0)

    def load_current(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]] | None:
        if not self.current_path.exists():
            return None
        try:
            pointer = json.loads(self.current_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BusinessFault(E_SNAPSHOT_CORRUPT, "current snapshot pointer is unreadable", "snapshot") from exc
        if not isinstance(pointer, dict) or not isinstance(pointer.get("snapshot"), str):
            raise BusinessFault(E_SNAPSHOT_CORRUPT, "current snapshot pointer is invalid", "snapshot")
        snapshot_name = pointer["snapshot"]
        if Path(snapshot_name).name != snapshot_name:
            raise BusinessFault(E_SNAPSHOT_CORRUPT, "snapshot pointer escapes snapshot directory", "snapshot")
        snapshot_dir = self.snapshots_dir / snapshot_name
        if not snapshot_dir.is_dir():
            raise BusinessFault(E_SNAPSHOT_CORRUPT, "snapshot directory is missing", "snapshot")
        try:
            manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BusinessFault(E_SNAPSHOT_CORRUPT, "snapshot manifest is unreadable", "snapshot") from exc
        if not isinstance(manifest, dict):
            raise BusinessFault(E_SNAPSHOT_CORRUPT, "snapshot manifest is invalid", "snapshot")
        for filename in (*STAGE_FILES.values(), "records.jsonl"):
            if not (snapshot_dir / filename).is_file():
                raise BusinessFault(E_SNAPSHOT_CORRUPT, f"snapshot stage file {filename} is missing", "snapshot")
        records: dict[str, dict[str, Any]] = {}
        try:
            lines = (snapshot_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise BusinessFault(E_SNAPSHOT_CORRUPT, "snapshot records are unreadable", "snapshot") from exc
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BusinessFault(E_SNAPSHOT_CORRUPT, "snapshot contains malformed record JSON", "snapshot") from exc
            workorder_id = record.get("workorder_id") if isinstance(record, dict) else None
            if not isinstance(workorder_id, str) or not workorder_id:
                raise BusinessFault(E_SNAPSHOT_CORRUPT, "snapshot record has no workorder_id", "snapshot")
            records[workorder_id] = record
        version = manifest.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
            raise BusinessFault(E_SNAPSHOT_CORRUPT, "snapshot manifest version is invalid", "snapshot")
        self.last_version = max(self.last_version, version)
        manifest["snapshot_name"] = snapshot_name
        return manifest, records

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in sorted(rows, key=lambda item: str(item.get("workorder_id", ""))):
                handle.write(canonical_json(row))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def commit(
        self,
        records: dict[str, dict[str, Any]],
        *,
        config_hash: str,
        rules_fingerprint: str,
        input_version: int,
        completed_workorders: list[str],
        pending_workorders: list[str],
        consumed_budget_this_run: int,
        consumed_budget_total: int,
        stage_reuse_counts: dict[str, int],
        commit_reason: str,
    ) -> dict[str, Any]:
        version = self.last_version + 1
        final_name = f"v{version:08d}"
        final_dir = self.snapshots_dir / final_name
        if final_dir.exists():
            raise BusinessFault(E_SNAPSHOT_IO, f"snapshot {final_name} already exists", "snapshot")
        try:
            temp_name = tempfile.mkdtemp(prefix=".tmp-snapshot-", dir=self.state_dir)
            temp_dir = Path(temp_name)
        except OSError as exc:
            raise BusinessFault(E_SNAPSHOT_IO, "cannot create snapshot temporary directory", "snapshot") from exc

        try:
            sorted_records = [records[key] for key in sorted(records)]
            load_rows = [
                {
                    "workorder_id": record["workorder_id"],
                    "evidence_fingerprint": record.get("evidence_fingerprint"),
                    "failure": record.get("failure"),
                }
                for record in sorted_records
            ]
            decay_rows = [
                {
                    "workorder_id": record["workorder_id"],
                    "cache_key": record.get("decay_key"),
                    "payload": record.get("decay_payload"),
                }
                for record in sorted_records
            ]
            relation_rows = [
                {
                    "workorder_id": record["workorder_id"],
                    "cache_key": record.get("relation_key"),
                    "payload": record.get("relation_payload"),
                }
                for record in sorted_records
            ]
            aggregation_rows = [
                {
                    "workorder_id": record["workorder_id"],
                    "cache_key": record.get("aggregate_key"),
                    "payload": record.get("aggregate_payload"),
                }
                for record in sorted_records
            ]
            self._write_jsonl(temp_dir / STAGE_FILES["load"], load_rows)
            self._write_jsonl(temp_dir / STAGE_FILES["decay"], decay_rows)
            self._write_jsonl(temp_dir / STAGE_FILES["relations"], relation_rows)
            self._write_jsonl(temp_dir / STAGE_FILES["aggregation"], aggregation_rows)
            self._write_jsonl(temp_dir / "records.jsonl", sorted_records)

            manifest = {
                "schema_version": 1,
                "version": version,
                "config_hash": config_hash,
                "rules_fingerprint": rules_fingerprint,
                "input_version": input_version,
                "completed_workorders": sorted(completed_workorders),
                "pending_workorders": sorted(pending_workorders),
                "consumed_budget_this_run": consumed_budget_this_run,
                "consumed_budget_total": consumed_budget_total,
                "stage_reuse_counts": dict(sorted(stage_reuse_counts.items())),
                "commit_reason": commit_reason,
                "stage_files": dict(STAGE_FILES),
                "atomic_commit": True,
            }
            with (temp_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(temp_dir, final_dir)
            except OSError as exc:
                raise BusinessFault(E_SNAPSHOT_IO, "atomic snapshot directory replace failed", "snapshot") from exc

            pointer = {
                "snapshot": final_name,
                "version": version,
                "config_hash": config_hash,
                "rules_fingerprint": rules_fingerprint,
                "input_version": input_version,
            }
            pointer_temp = self.state_dir / f".current-{version:08d}.tmp"
            try:
                pointer_temp.write_text(
                    json.dumps(pointer, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(pointer_temp, self.current_path)
            except OSError as exc:
                raise BusinessFault(E_SNAPSHOT_IO, "atomic current pointer replace failed", "snapshot") from exc
            self.last_version = version
            manifest["snapshot_name"] = final_name
            return manifest
        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

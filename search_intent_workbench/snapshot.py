"""Versioned stage snapshots with atomic publication and recovery checks."""

from __future__ import annotations

import copy
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, atomic_write_jsonl, file_digest, read_json
from .errors import E_SNAPSHOT_STALE, issue
from .pipeline import STAGES


SNAPSHOT_FORMAT_VERSION = "snapshot-v1"


class SimulatedCrash(RuntimeError):
    pass


class SnapshotVersionExists(RuntimeError):
    pass


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def _version_name(self, version: int) -> str:
        return f"v{version:06d}"

    def _stage_rows(self, run: dict[str, Any], stage: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for query in run.get("queries", []):
            record = query["stages"][stage]
            rows.append(
                {
                    "query_id": query["query_id"],
                    "sort_key": record["sort_key"],
                    "cache_key": record["cache_key"],
                    "output_fingerprint": record["output_fingerprint"],
                    "references": record["references"],
                    "data": record["data"],
                }
            )
        rows.sort(key=lambda item: item["query_id"])
        return rows

    def commit(
        self,
        *,
        version: int,
        state: Any,
        run: dict[str, Any],
        report: dict[str, Any] | None = None,
        crash_after_stage: str | None = None,
        hard_exit: bool = False,
    ) -> dict[str, Any]:
        version_name = self._version_name(version)
        final_dir = self.versions_dir / version_name
        if final_dir.exists():
            raise SnapshotVersionExists(f"snapshot version already exists: {version_name}")

        staging_dir = self.root / f".staging-{version_name}-{uuid.uuid4().hex[:12]}"
        staging_dir.mkdir(parents=True, exist_ok=False)
        manifest: dict[str, Any] = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "global_version": version,
            "status": "in_progress",
            "config_hash": run["metadata"]["config_hash"],
            "rule_fingerprint": run["metadata"]["rule_fingerprint"],
            "dictionary_hash": run["metadata"]["dictionary_hash"],
            "completed_stage": None,
            "completed_query_cursor": 0,
            "command_index": run.get("command_index", 0),
            "stage_files": {},
            "state_file": "state.json",
            "report_file": "report.json" if report is not None else None,
        }
        atomic_write_json(staging_dir / "state.json", {
            "queries": state.sorted_queries(),
            "config": state.config,
            "rules": state.rules,
            "failures": state.failures,
        })
        if report is not None:
            atomic_write_json(staging_dir / "report.json", report)
        atomic_write_json(staging_dir / "manifest.json", manifest)

        for stage in STAGES:
            rows = self._stage_rows(run, stage)
            stage_file = f"stage_{stage}.jsonl"
            atomic_write_jsonl(staging_dir / stage_file, rows)
            manifest["completed_stage"] = stage
            manifest["completed_query_cursor"] = len(rows)
            manifest["stage_files"][stage] = {
                "file": stage_file,
                "count": len(rows),
                "sha256": file_digest(staging_dir / stage_file),
            }
            atomic_write_json(staging_dir / "manifest.json", manifest)
            if crash_after_stage == stage:
                manifest["status"] = "crashed_in_progress"
                atomic_write_json(staging_dir / "manifest.json", manifest)
                if hard_exit:
                    os._exit(87)
                raise SimulatedCrash(
                    f"simulated termination after stage {stage}; staging={staging_dir}"
                )

        manifest["status"] = "committed"
        manifest["completed_stage"] = "rewrite"
        manifest["completed_query_cursor"] = len(run.get("queries", []))
        final_manifest = copy.deepcopy(manifest)
        atomic_write_json(staging_dir / "manifest.json", final_manifest)
        os.replace(staging_dir, final_dir)
        atomic_write_json(
            self.root / "CURRENT",
            {
                "version": version,
                "version_name": version_name,
                "path": str(final_dir.relative_to(self.root)),
                "config_hash": run["metadata"]["config_hash"],
            },
        )
        return final_manifest

    def _manifest_for(self, directory: Path) -> dict[str, Any] | None:
        try:
            value = read_json(directory / "manifest.json")
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def inspect(
        self, *, expected_version: int, expected_config_hash: str
    ) -> dict[str, Any]:
        """Decide RESUME vs FRESH vs E_SNAPSHOT_STALE without mixing artifacts."""

        committed: list[tuple[Path, dict[str, Any]]] = []
        staged: list[tuple[Path, dict[str, Any]]] = []
        for directory in sorted(self.versions_dir.iterdir() if self.versions_dir.exists() else []):
            if directory.is_dir() and directory.name.startswith("v"):
                manifest = self._manifest_for(directory)
                if manifest is not None:
                    committed.append((directory, manifest))
        for directory in sorted(self.root.glob(".staging-*")):
            if directory.is_dir():
                manifest = self._manifest_for(directory)
                if manifest is not None:
                    staged.append((directory, manifest))

        for directory, manifest in sorted(
            committed, key=lambda item: int(item[1].get("global_version", -1)), reverse=True
        ):
            if (
                manifest.get("global_version") == expected_version
                and manifest.get("config_hash") == expected_config_hash
                and manifest.get("status") == "committed"
            ):
                return {
                    "status": "RESUME",
                    "reusable": True,
                    "reusable_artifacts": [
                        str(directory / item["file"])
                        for item in manifest.get("stage_files", {}).values()
                    ],
                    "manifest": manifest,
                    "path": str(directory),
                    "completed_stage": manifest.get("completed_stage"),
                    "completed_query_cursor": manifest.get("completed_query_cursor"),
                    "errors": [],
                }

        for directory, manifest in sorted(staged, key=lambda item: str(item[0]), reverse=True):
            if (
                manifest.get("global_version") == expected_version
                and manifest.get("config_hash") == expected_config_hash
                and manifest.get("status") in {"in_progress", "crashed_in_progress"}
            ):
                completed_stages = list(manifest.get("stage_files", {}))
                return {
                    "status": "RESUME_STAGING",
                    "reusable": True,
                    "reusable_artifacts": [
                        str(directory / manifest["stage_files"][stage]["file"])
                        for stage in completed_stages
                    ],
                    "manifest": manifest,
                    "path": str(directory),
                    "completed_stage": manifest.get("completed_stage"),
                    "completed_query_cursor": manifest.get("completed_query_cursor", 0),
                    "errors": [],
                }

        if committed or staged:
            current = None
            try:
                current = read_json(self.root / "CURRENT")
            except (OSError, json.JSONDecodeError):
                pass
            return {
                "status": E_SNAPSHOT_STALE,
                "reusable": False,
                "reusable_artifacts": [],
                "manifest": None,
                "path": None,
                "current": current,
                "expected_version": expected_version,
                "expected_config_hash": expected_config_hash,
                "errors": [
                    issue(
                        E_SNAPSHOT_STALE,
                        "snapshot version or config hash does not match the requested state",
                        detail={
                            "expected_version": expected_version,
                            "expected_config_hash": expected_config_hash,
                            "current": current,
                        },
                    )
                ],
            }

        return {
            "status": "FRESH",
            "reusable": False,
            "reusable_artifacts": [],
            "manifest": None,
            "path": None,
            "errors": [],
        }

    def load_stage_cache(
        self, inspection: dict[str, Any]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Load stage data from a resumable snapshot, never from stale artifacts."""

        if not inspection.get("reusable"):
            return {}
        directory = Path(inspection["path"])
        manifest = inspection["manifest"]
        cache: dict[tuple[str, str], dict[str, Any]] = {}
        for stage, descriptor in manifest.get("stage_files", {}).items():
            path = directory / descriptor["file"]
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    cache[(row["query_id"], stage)] = {
                        "cache_key": row["cache_key"],
                        "output_fingerprint": row["output_fingerprint"],
                        "data": row["data"],
                    }
        return cache

    @staticmethod
    def remove(root: str | Path) -> None:
        path = Path(root)
        if path.exists():
            shutil.rmtree(path)

"""Stage-artifact snapshots with atomic whole-directory replacement."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .canonical import canonical_data, digest
from .errors import E_SNAPSHOT_INCOMPATIBLE, Failure, sorted_failures
from .models import FeatureCatalog, LabConfig, RawCandidate


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            canonical_data(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def _write_json(path: Path, value: Any) -> str:
    text = _json_text(value)
    path.write_text(text, encoding="utf-8", newline="\n")
    return digest(value)


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


class SnapshotManager:
    def __init__(
        self,
        output_dir: Path,
        config_hash: str,
        groups_hash: str,
        candidates_hash: str,
        config: LabConfig | None,
        catalog: FeatureCatalog,
    ) -> None:
        self.output_dir = output_dir
        self.snapshot_dir = output_dir / "snapshot"
        self.temp_dir = output_dir / ".tmp-snapshot"
        self.backup_dir = output_dir / ".snapshot-old"
        self.manifest_path = self.snapshot_dir / "manifest.json"
        self.config_hash = config_hash
        self.groups_hash = groups_hash
        self.candidates_hash = candidates_hash
        self.config = config
        self.catalog = catalog
        self.commits: list[dict[str, Any]] = []

    def read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.is_file():
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def prepare(self, clean: bool) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _remove(self.temp_dir)
        _remove(self.backup_dir)
        if clean:
            # Cleaning happens before recovery inspection so repeated clean runs
            # produce a byte-identical report rather than leaking old version data.
            _remove(self.snapshot_dir)
            return {
                "action": "clean",
                "previous_version": None,
                "temporary_directories_removed": True,
            }
        previous = self.read_manifest()
        previous_version = previous.get("global_version") if previous else None
        if previous is None:
            return {
                "action": "fresh",
                "previous_version": None,
                "temporary_directories_removed": True,
            }
    # A metadata-only recovery decision is safe: command replay always rebuilds
    # derived state and the runner never assumes an old cursor is current.
        hash_match = (
            previous.get("config_hash") == self.config_hash
            and previous.get("feature_groups_hash") == self.groups_hash
        )
        return {
            "action": "resume_metadata" if hash_match else "resume_incompatible",
            "previous_version": previous_version,
            "previous_config_hash": previous.get("config_hash"),
            "previous_groups_hash": previous.get("feature_groups_hash"),
            "hash_match": hash_match,
            "error_code": None if hash_match else E_SNAPSHOT_INCOMPATIBLE,
            "temporary_directories_removed": True,
        }

    def commit(
        self,
        version: int,
        command_index: int | None,
        candidates: list[RawCandidate],
        evaluation: Any,
        failures: list[Failure],
    ) -> dict[str, Any]:
        request_results = evaluation.requests if evaluation is not None else {}
        completed = sorted(request_results)
        cursor = completed[-1] if completed else None
        validation_artifact = {
            "version": version,
            "feature_catalog": self.catalog.to_dict(),
            "candidates": [
                {
                    "request_id": candidate.request_id,
                    "item_id": candidate.item_id,
                    "features": {
                        name: str(value)
                        for name, value in sorted(candidate.features.items())
                    },
                    "line": candidate.line,
                }
                for candidate in sorted(
                    candidates, key=lambda item: (item.request_id, item.item_id)
                )
            ],
            "failures": [failure.to_dict() for failure in sorted_failures(failures)],
        }
        normalization_artifact = {
            "version": version,
            "requests": {
                request_id: request_results[request_id]["normalization"]
                for request_id in completed
            },
        }
        fusion_artifact = {
            "version": version,
            "requests": {
                request_id: {
                    "feature_dimension": request_results[request_id]["feature_dimension"],
                    "cache_key": request_results[request_id]["cache_keys"]["fusion"],
                    "fusion": request_results[request_id]["fusion"],
                }
                for request_id in completed
            },
        }
        rerank_artifact = {
            "version": version,
            "requests": {
                request_id: {
                    "cache_key": request_results[request_id]["cache_keys"]["rerank"],
                    "final_sort_cache_key": request_results[request_id]["cache_keys"][
                        "final_sort"
                    ],
                    "rerank": request_results[request_id]["rerank"],
                }
                for request_id in completed
            },
        }

        stage_payloads = {
            "validation": validation_artifact,
            "normalization": normalization_artifact,
            "fusion": fusion_artifact,
            "rerank": rerank_artifact,
        }
        stage_hashes = {name: digest(payload) for name, payload in stage_payloads.items()}
        manifest = {
            "snapshot_schema_version": 1,
            "global_version": version,
            "command_index": command_index,
            "config_hash": self.config_hash,
            "feature_groups_hash": self.groups_hash,
            "candidates_hash": self.candidates_hash,
            "completed_request_cursor": cursor,
            "completed_requests": completed,
            "stage_artifacts": {
                name: {"version": version, "hash": stage_hashes[name]}
                for name in sorted(stage_hashes)
            },
            "request_result_keys": {
                request_id: request_results[request_id]["cache_keys"]["final_sort"]
                for request_id in completed
            },
        }
        manifest_hash = digest(manifest)
        manifest["manifest_hash"] = manifest_hash

        decision = "replace"
        reason = "version_or_content_changed"
        previous = self.read_manifest()
        if previous is not None:
            same_content = (
                previous.get("config_hash") == self.config_hash
                and previous.get("feature_groups_hash") == self.groups_hash
                and previous.get("candidates_hash") == self.candidates_hash
                and previous.get("completed_request_cursor") == cursor
            )
            if previous.get("global_version") == version and same_content:
                decision = "skip_unchanged"
                reason = "same_version_and_hashes"
                result = {
                    "decision": decision,
                    "reason": reason,
                    "global_version": version,
                    "config_hash": self.config_hash,
                    "feature_groups_hash": self.groups_hash,
                    "candidates_hash": self.candidates_hash,
                    "completed_request_cursor": cursor,
                    "manifest_hash": previous.get("manifest_hash"),
                    "path": "snapshot",
                }
                self.commits.append(result)
                return result

        _remove(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            for name in ("validation", "normalization", "fusion", "rerank"):
                _write_json(self.temp_dir / f"{name}.json", stage_payloads[name])
            _write_json(self.temp_dir / "manifest.json", manifest)
            _remove(self.backup_dir)
            if self.snapshot_dir.exists():
                os.replace(self.snapshot_dir, self.backup_dir)
            os.replace(self.temp_dir, self.snapshot_dir)
            _remove(self.backup_dir)
        except Exception:
            if self.snapshot_dir.exists() is False and self.backup_dir.exists():
                os.replace(self.backup_dir, self.snapshot_dir)
            _remove(self.temp_dir)
            raise

        result = {
            "decision": decision,
            "reason": reason,
            "global_version": version,
            "config_hash": self.config_hash,
            "feature_groups_hash": self.groups_hash,
            "candidates_hash": self.candidates_hash,
            "completed_request_cursor": cursor,
            "manifest_hash": manifest_hash,
            "stage_hashes": stage_hashes,
            "path": "snapshot",
        }
        self.commits.append(result)
        return result

    def latest_manifest(self) -> dict[str, Any] | None:
        return self.read_manifest()

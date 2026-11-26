"""Versioned stage snapshots with atomic pointer replacement and recovery."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gsb.util import canonical_json, json_dump_atomic, sha256_bytes


def _stage_payload(report: dict[str, Any]) -> dict[str, Any]:
    stages: dict[str, list[Any]] = {
        "candidates": [],
        "components": [],
        "keypoints": [],
        "recognitions": [],
        "reading": [],
        "conflicts": [],
        "invalidation": [],
        "failures": list(report.get("failures", [])),
        "pending": list(report.get("pending", [])),
    }
    for sample in report.get("samples", []):
        sample_id = sample.get("sample_id", "")
        for name in ("candidates", "components", "keypoints", "recognitions"):
            for item in sample.get(name, []):
                stages[name].append({"sample_id": sample_id, **item})
        for item in sample.get("reading_order", []):
            stages["reading"].append({"sample_id": sample_id, **item})
        for item in sample.get("conflicts", []):
            stages["conflicts"].append({"sample_id": sample_id, **item})
        for item in sample.get("invalidation", []):
            stages["invalidation"].append({"sample_id": sample_id, **item})
    return stages


@dataclass(frozen=True)
class SnapshotCommit:
    global_version: str
    snapshot_dir: Path
    manifest: dict[str, Any]
    committed: bool


@dataclass(frozen=True)
class SnapshotRecovery:
    report: dict[str, Any] | None
    manifest: dict[str, Any] | None
    recovered: bool
    current_valid: bool
    issues: tuple[str, ...]


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.current_path = self.root / "current.json"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def _validate_version(self, version_dir: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        issues: list[str] = []
        files = manifest.get("files")
        if not isinstance(files, dict):
            return None, ["manifest files mapping is missing"]
        for relative, expected_hash in sorted(files.items()):
            path = version_dir / relative
            if not path.is_file():
                issues.append(f"missing snapshot file: {relative}")
                continue
            actual_hash = sha256_bytes(path.read_bytes())
            if actual_hash != expected_hash:
                issues.append(f"snapshot file hash mismatch: {relative}")
        report_path = version_dir / "report.json"
        report: dict[str, Any] | None = None
        if report_path.is_file():
            try:
                loaded = json.loads(report_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    report = loaded
                else:
                    issues.append("report.json is not an object")
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"report.json cannot be read: {exc}")
        else:
            issues.append("report.json is missing")
        return report, issues

    def commit(
        self,
        report: dict[str, Any],
        *,
        completed_command_cursor: int,
        consumed_budget: int,
    ) -> SnapshotCommit:
        stages = _stage_payload(report)
        state_payload = {
            "config_hash": report.get("config_hash"),
            "template_version": report.get("template_version"),
            "template_fingerprint": report.get("template_fingerprint"),
            "completed_command_cursor": completed_command_cursor,
            "consumed_budget": consumed_budget,
            "pending": report.get("pending", []),
        }
        global_version = sha256_bytes(
            canonical_json({"report": report, "state": state_payload}).encode("utf-8")
        )
        final_dir = self.versions_dir / global_version
        if final_dir.is_dir():
            existing_manifest_path = final_dir / "manifest.json"
            if existing_manifest_path.is_file():
                manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
                self._write_current(global_version, manifest)
                return SnapshotCommit(global_version, final_dir, manifest, False)

        temp_dir = Path(tempfile.mkdtemp(prefix=".stage-", dir=self.root))
        try:
            files: dict[str, str] = {}
            payloads: dict[str, Any] = {
                "report.json": report,
                "stages/candidates.json": stages["candidates"],
                "stages/components.json": stages["components"],
                "stages/keypoints.json": stages["keypoints"],
                "stages/recognitions.json": stages["recognitions"],
                "stages/reading.json": stages["reading"],
                "stages/conflicts.json": stages["conflicts"],
                "stages/invalidation.json": stages["invalidation"],
                "stages/failures.json": stages["failures"],
                "stages/pending.json": stages["pending"],
                "state.json": state_payload,
            }
            for relative, payload in payloads.items():
                path = temp_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
                files[relative] = sha256_bytes(path.read_bytes())
            previous = None
            if self.current_path.is_file():
                try:
                    previous = json.loads(self.current_path.read_text(encoding="utf-8")).get("global_version")
                except (OSError, json.JSONDecodeError):
                    previous = None
            manifest = {
                "schema_version": 1,
                "global_version": global_version,
                "config_hash": report.get("config_hash"),
                "template_version": report.get("template_version"),
                "template_fingerprint": report.get("template_fingerprint"),
                "completed_command_cursor": completed_command_cursor,
                "consumed_budget": consumed_budget,
                "pending_count": len(report.get("pending", [])),
                "previous_version": previous,
                "files": files,
            }
            manifest_path = temp_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            if final_dir.exists():
                shutil.rmtree(temp_dir)
            else:
                os.replace(temp_dir, final_dir)
            self._write_current(global_version, manifest)
            return SnapshotCommit(global_version, final_dir, manifest, True)
        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _write_current(self, global_version: str, manifest: dict[str, Any]) -> None:
        pointer = {
            "schema_version": 1,
            "global_version": global_version,
            "manifest": "manifest.json",
            "manifest_sha256": sha256_bytes(
                canonical_json(manifest).encode("utf-8")
            ),
        }
        json_dump_atomic(self.current_path, pointer)

    def load_current(self) -> SnapshotRecovery:
        issues: list[str] = []
        if not self.current_path.is_file():
            return SnapshotRecovery(None, None, False, False, ("current pointer is missing",))
        pointer: dict[str, Any] = {}
        version: str | None = None
        version_dir: Path | None = None
        manifest: dict[str, Any] | None = None
        try:
            pointer = json.loads(self.current_path.read_text(encoding="utf-8"))
            version = pointer["global_version"]
            version_dir = self.versions_dir / version
            manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            issues.append(f"current pointer is invalid: {exc}")
        if version is not None and version_dir is not None and manifest is not None:
            expected_manifest_hash = pointer.get("manifest_sha256")
            actual_manifest_hash = sha256_bytes(canonical_json(manifest).encode("utf-8"))
            if isinstance(expected_manifest_hash, str) and expected_manifest_hash != actual_manifest_hash:
                issues.append("current manifest hash mismatch")
            report, current_issues = self._validate_version(version_dir, manifest)
            if not current_issues and report is not None and not issues:
                return SnapshotRecovery(report, manifest, False, True, ())
            issues.extend(current_issues)
        candidates = sorted(
            (path for path in self.versions_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        for candidate in candidates:
            if candidate.name == version:
                continue
            try:
                candidate_manifest = json.loads(
                    (candidate / "manifest.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            candidate_report, candidate_issues = self._validate_version(
                candidate,
                candidate_manifest,
            )
            if candidate_report is not None and not candidate_issues:
                return SnapshotRecovery(
                    candidate_report,
                    candidate_manifest,
                    True,
                    False,
                    tuple(issues),
                )
        return SnapshotRecovery(None, None, False, False, tuple(issues))

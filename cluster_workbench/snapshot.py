from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import E_SNAPSHOT_CORRUPT, E_SNAPSHOT_VERSION_MISMATCH, Issue
from .util import atomic_write_text, canonical_json, sha256_file, stable_hash


SNAPSHOT_SCHEMA_VERSION = 1


class SnapshotStore:
    """Versioned snapshots published through an atomically replaced pointer.

    A snapshot directory is fully written and fsynced before it becomes visible.
    The ``CURRENT`` pointer is then atomically replaced, giving readers either
    the old complete snapshot or the new complete snapshot, never a partial mix.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions_root = self.root / "versions"
        self.versions_root.mkdir(parents=True, exist_ok=True)
        self.pointer_path = self.root / "CURRENT"

    def commit(self, state: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
        version = self._next_version()
        temporary = Path(
            tempfile.mkdtemp(prefix=f".tmp-{version:06d}-", dir=str(self.versions_root))
        )
        try:
            state_text = canonical_json(state)
            report_text = canonical_json(report)
            atomic_write_text(temporary / "state.json", state_text)
            atomic_write_text(temporary / "report.json", report_text)
            snapshot_id = stable_hash(
                {"version": version, "state": state_text, "report": report_text}
            )
            manifest = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "version": version,
                "snapshot_id": snapshot_id,
                "state_fingerprint": report.get("state_fingerprint"),
                "command_seq": state.get("command_seq", report.get("command_seq", 0)),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "files": {
                    "state.json": sha256_file(temporary / "state.json"),
                    "report.json": sha256_file(temporary / "report.json"),
                },
                "stages": {
                    "preprocess": "complete",
                    "cluster": "complete",
                    "evaluation": "complete",
                    "diagnostics": report.get("diagnostics", {}).get("content_fingerprint"),
                },
                "counts": report.get("counts", {}),
            }
            atomic_write_text(temporary / "manifest.json", canonical_json(manifest))
            destination = self.versions_root / f"v{version:06d}-{snapshot_id[:12]}"
            os.replace(temporary, destination)
            pointer = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "version": version,
                "snapshot_id": snapshot_id,
                "directory": destination.name,
                "manifest_sha256": sha256_file(destination / "manifest.json"),
            }
            atomic_write_text(self.pointer_path, canonical_json(pointer))
            return pointer
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _next_version(self) -> int:
        maximum = 0
        for item in self.versions_root.iterdir():
            if not item.is_dir() or not item.name.startswith("v"):
                continue
            prefix = item.name[1:7]
            if prefix.isdigit():
                maximum = max(maximum, int(prefix))
        return maximum + 1

    def recover(self) -> tuple[dict[str, Any] | None, list[Issue]]:
        issues: list[Issue] = []
        if not self.pointer_path.exists():
            candidates = sorted(
                [item for item in self.versions_root.iterdir() if item.is_dir() and item.name.startswith("v")],
                reverse=True,
            )
            for candidate in candidates:
                payload, candidate_issues = self._load_directory(candidate)
                if payload is not None:
                    return payload, issues + candidate_issues
            return None, issues
        try:
            with self.pointer_path.open("r", encoding="utf-8") as handle:
                pointer = json.load(handle)
        except Exception as exc:
            return None, [
                Issue(
                    E_SNAPSHOT_CORRUPT,
                    f"cannot read snapshot pointer: {exc}",
                    severity="warning",
                    context={"scope": "snapshot"},
                )
            ]
        if not isinstance(pointer, dict) or pointer.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            return None, [
                Issue(
                    E_SNAPSHOT_VERSION_MISMATCH,
                    "snapshot pointer schema version is incompatible",
                    context={
                        "scope": "snapshot",
                        "expected": SNAPSHOT_SCHEMA_VERSION,
                        "actual": pointer.get("schema_version") if isinstance(pointer, dict) else None,
                    },
                )
            ]
        directory_name = pointer.get("directory")
        if not isinstance(directory_name, str):
            return None, [
                Issue(E_SNAPSHOT_CORRUPT, "snapshot pointer has no directory", context={"scope": "snapshot"})
            ]
        directory = self.versions_root / directory_name
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists() or sha256_file(manifest_path) != pointer.get("manifest_sha256"):
            issues.append(
                Issue(
                    E_SNAPSHOT_CORRUPT,
                    "pointed snapshot manifest is missing or corrupted",
                    severity="warning",
                    context={"scope": "snapshot", "directory": directory_name},
                )
            )
            fallback, fallback_issues = self._recover_older(exclude=directory_name)
            return fallback, issues + fallback_issues
        payload, load_issues = self._load_directory(directory)
        issues.extend(load_issues)
        return payload, issues

    def _recover_older(self, *, exclude: str | None = None) -> tuple[dict[str, Any] | None, list[Issue]]:
        issues: list[Issue] = []
        candidates = sorted(
            [
                item
                for item in self.versions_root.iterdir()
                if item.is_dir() and item.name.startswith("v") and item.name != exclude
            ],
            reverse=True,
        )
        for candidate in candidates:
            payload, candidate_issues = self._load_directory(candidate)
            issues.extend(candidate_issues)
            if payload is not None:
                return payload, issues
        return None, issues

    def _load_directory(self, directory: Path) -> tuple[dict[str, Any] | None, list[Issue]]:
        issues: list[Issue] = []
        manifest_path = directory / "manifest.json"
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception as exc:
            return None, [
                Issue(
                    E_SNAPSHOT_CORRUPT,
                    f"cannot read snapshot manifest: {exc}",
                    severity="warning",
                    context={"scope": "snapshot", "directory": directory.name},
                )
            ]
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            return None, [
                Issue(
                    E_SNAPSHOT_VERSION_MISMATCH,
                    "snapshot schema version is incompatible",
                    context={
                        "scope": "snapshot",
                        "directory": directory.name,
                        "expected": SNAPSHOT_SCHEMA_VERSION,
                        "actual": manifest.get("schema_version"),
                    },
                )
            ]
        files = manifest.get("files", {})
        if not isinstance(files, dict):
            return None, [
                Issue(E_SNAPSHOT_CORRUPT, "snapshot manifest files field is invalid", severity="warning")
            ]
        for name, expected_hash in files.items():
            path = directory / str(name)
            if not path.exists() or sha256_file(path) != expected_hash:
                return None, [
                    Issue(
                        E_SNAPSHOT_CORRUPT,
                        f"snapshot file {name!r} is missing or corrupted",
                        severity="warning",
                        context={"scope": "snapshot", "directory": directory.name},
                    )
                ]
        try:
            with (directory / "state.json").open("r", encoding="utf-8") as handle:
                state = json.load(handle)
            with (directory / "report.json").open("r", encoding="utf-8") as handle:
                report = json.load(handle)
        except Exception as exc:
            return None, [
                Issue(
                    E_SNAPSHOT_CORRUPT,
                    f"cannot read snapshot payload: {exc}",
                    severity="warning",
                    context={"scope": "snapshot", "directory": directory.name},
                )
            ]
        return {"state": state, "report": report, "manifest": manifest}, issues

    def versions(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for directory in sorted(self.versions_root.iterdir()):
            if not directory.is_dir() or not directory.name.startswith("v"):
                continue
            manifest_path = directory / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except Exception:
                continue
            result.append(
                {
                    "directory": directory.name,
                    "version": manifest.get("version"),
                    "schema_version": manifest.get("schema_version"),
                    "snapshot_id": manifest.get("snapshot_id"),
                }
            )
        return result

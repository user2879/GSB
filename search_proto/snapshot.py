from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .engine import SearchEngine
from .report import engine_artifacts
from .util import read_json, sha256_bytes, sha256_json, write_json, write_jsonl


SNAPSHOT_SCHEMA = "search-snapshot-v1"


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_dir(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class SnapshotStore:
    def __init__(self, root: Path, *, keep_versions: int = 3):
        self.root = root
        self.versions_dir = root / "versions"
        self.temp_dir = root / ".tmp"
        self.current_path = root / "CURRENT"
        self.last_good_path = root / "LAST_GOOD"
        self.keep_versions = max(1, keep_versions)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _next_version_name(self) -> str:
        numbers = []
        for path in self.versions_dir.iterdir():
            if path.is_dir() and path.name.startswith("v") and path.name[1:].isdigit():
                numbers.append(int(path.name[1:]))
        return f"v{max(numbers, default=0) + 1:08d}"

    def _write_staged_snapshot(self, engine: SearchEngine) -> tuple[Path, dict[str, Any]]:
        artifacts = engine_artifacts(engine)
        staging = self.temp_dir / f"stage-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        written: dict[str, Path] = {}

        def write(name: str, value: Any) -> None:
            path = staging / name
            write_json(path, value)
            written[name] = path

        def write_lines(name: str, rows: Any) -> None:
            path = staging / name
            write_jsonl(path, rows)
            written[name] = path

        write("index.json", artifacts["index"])
        write("term_stats.json", artifacts["term_stats"])
        write("scores.json", artifacts["scores"])
        write("ranking.json", artifacts["ranking"])
        write("failures.json", artifacts["failures"])
        write_lines("cache_ledger.jsonl", artifacts["cache_ledger"])
        write("report.json", artifacts["report"])
        state = engine.to_state_dict()
        write("state.json", state)

        checksums = {
            name: sha256_bytes(path.read_bytes()) for name, path in sorted(written.items())
        }
        manifest = {
            "schema": SNAPSHOT_SCHEMA,
            "global_version": engine.global_version,
            "config_hash": engine.config.full_fingerprint,
            "semantic_config_hash": engine.config.semantic_fingerprint,
            "stopword_version": engine.stopwords.effective_version,
            "completed_query_cursor": engine.completed_query_cursor,
            "completed_query_ids": list(engine.completed_query_ids),
            "pending_query_ids": list(engine.pending_query_ids),
            "batch_index": engine.batch_index,
            "total_consumed_points": engine.total_consumed_points,
            "batch_consumed_points": {
                str(key): engine.batch_consumed_points[key]
                for key in sorted(engine.batch_consumed_points)
            },
            "artifact_sha256": checksums,
            "state_fingerprint": checksums["state.json"],
            "state_json_fingerprint": sha256_json(state),
        }
        manifest_path = staging / "manifest.json"
        write_json(manifest_path, manifest)
        _fsync_file(manifest_path)
        for path in written.values():
            _fsync_file(path)
        _fsync_dir(staging)
        return staging, manifest

    def commit(self, engine: SearchEngine) -> dict[str, Any]:
        staging, manifest = self._write_staged_snapshot(engine)
        version_name = self._next_version_name()
        version_path = self.versions_dir / version_name
        os.replace(staging, version_path)
        _fsync_dir(self.versions_dir)

        previous_current = self._read_pointer(self.current_path)
        if previous_current:
            _atomic_write_text(self.last_good_path, previous_current + "\n")
        _atomic_write_text(self.current_path, version_name + "\n")
        self._cleanup_versions(keep={version_name})
        return {
            "version_name": version_name,
            "manifest": manifest,
            "path": str(version_path),
        }

    def _read_pointer(self, path: Path) -> str | None:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not value or not value.startswith("v"):
            return None
        return value

    def _validate_version(self, version_name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
        version_path = self.versions_dir / version_name
        manifest_path = version_path / "manifest.json"
        try:
            manifest = read_json(manifest_path)
            if manifest.get("schema") != SNAPSHOT_SCHEMA:
                return None, None, "manifest_schema_mismatch"
            checksums = manifest.get("artifact_sha256", {})
            for name, expected in checksums.items():
                path = version_path / name
                if not path.exists() or sha256_bytes(path.read_bytes()) != expected:
                    return None, manifest, f"checksum_mismatch:{name}"
            state = read_json(version_path / "state.json")
            if sha256_json(state) != manifest.get("state_json_fingerprint", sha256_json(state)):
                return None, manifest, "state_fingerprint_mismatch"
            return manifest, state, "ok"
        except (OSError, ValueError, TypeError) as exc:
            return None, None, f"read_error:{exc}"

    def load_best(self) -> dict[str, Any]:
        attempts: list[tuple[str, str | None]] = []
        current = self._read_pointer(self.current_path)
        last_good = self._read_pointer(self.last_good_path)
        if current:
            attempts.append(("current", current))
        if last_good and last_good != current:
            attempts.append(("last_good", last_good))
        for role, version_name in attempts:
            assert version_name is not None
            manifest, state, status = self._validate_version(version_name)
            if manifest is not None and state is not None:
                return {
                    "status": "ok" if role == "current" else "recovered_previous",
                    "role": role,
                    "version_name": version_name,
                    "manifest": manifest,
                    "state": state,
                    "validation": status,
                }
        return {
            "status": "missing" if not attempts else "corrupt",
            "role": None,
            "version_name": None,
            "attempts": [
                {"role": role, "version": version_name} for role, version_name in attempts
            ],
        }

    def load_for_resume(
        self,
        *,
        source_fingerprint: str,
        target_config_hash: str,
        target_stopword_version: str | None = None,
        expected_global_version: int | None = None,
    ) -> dict[str, Any]:
        best = self.load_best()
        if best["status"] not in {"ok", "recovered_previous"}:
            return {**best, "decision": "rebuild", "reason": best["status"]}
        state = best["state"]
        manifest = best["manifest"]
        if state.get("input_fingerprint") not in {None, source_fingerprint}:
            return {
                **best,
                "decision": "rebuild",
                "reason": "source_fingerprint_changed",
            }
        if manifest.get("config_hash") != target_config_hash:
            return {
                **best,
                "decision": "rebuild",
                "reason": "config_hash_changed",
            }
        if (
            target_stopword_version is not None
            and manifest.get("stopword_version") != target_stopword_version
        ):
            return {
                **best,
                "decision": "rebuild",
                "reason": "stopword_version_changed",
            }
        if (
            expected_global_version is not None
            and manifest.get("global_version") != expected_global_version
        ):
            return {
                **best,
                "decision": "rebuild",
                "reason": "global_version_mismatch",
            }
        return {
            **best,
            "decision": "resume",
            "reason": "version_and_config_match",
        }

    def _cleanup_versions(self, keep: set[str]) -> None:
        candidates = sorted(
            [
                path
                for path in self.versions_dir.iterdir()
                if path.is_dir() and path.name.startswith("v")
            ],
            key=lambda path: path.name,
            reverse=True,
        )
        retained = set(keep)
        for path in candidates:
            if len(retained) >= self.keep_versions:
                if path.name not in retained:
                    shutil.rmtree(path, ignore_errors=True)
            else:
                retained.add(path.name)

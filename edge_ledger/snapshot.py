from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import fcntl

from .canonical import digest
from .errors import SimulatedCrash


SNAPSHOT_SCHEMA_VERSION = 1
STAGE_NAMES = (
    "load_validation",
    "template_resolution",
    "schedule",
    "render",
    "score",
    "dedupe",
    "stats",
)


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    reason: str
    previous_generation: int | None = None
    previous_global_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "previous_generation": self.previous_generation,
            "previous_global_version": self.previous_global_version,
        }


@dataclass
class SnapshotView:
    generation: int
    manifest: dict[str, Any]
    stages: dict[str, Any]
    render_cache: dict[str, Any]
    score_cache: dict[str, Any]
    template_cache: dict[str, Any]
    pending_queue: list[dict[str, Any]]
    completed_template_cursor: dict[str, Any]
    consumed_budget: int
    result: dict[str, Any]
    input_fingerprints: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


@dataclass
class LoadSnapshotResult:
    snapshot: SnapshotView | None
    decision: RecoveryDecision


class SnapshotStore:
    def __init__(self, state_root: Path) -> None:
        self.state_root = Path(state_root)
        self.snapshots_root = self.state_root / "snapshots"
        self.current_path = self.state_root / "CURRENT"
        self.snapshots_root.mkdir(parents=True, exist_ok=True)

    def _generation_dirs(self) -> list[tuple[int, Path]]:
        result: list[tuple[int, Path]] = []
        for path in self.snapshots_root.iterdir():
            if not path.is_dir() or not path.name.isdigit():
                continue
            result.append((int(path.name), path))
        result.sort()
        return result

    def _next_generation(self) -> int:
        existing = self._generation_dirs()
        return (existing[-1][0] + 1) if existing else 1

    def _read_current_generation(self) -> int | None:
        if not self.current_path.exists() and not self.current_path.is_symlink():
            return None
        try:
            if self.current_path.is_symlink():
                target = os.readlink(self.current_path)
                return int(Path(target).name)
            text = self.current_path.read_text(encoding="utf-8").strip()
            return int(Path(text).name)
        except (OSError, ValueError):
            return None

    def load_current(
        self,
        *,
        expected_global_version: str | None = None,
        expected_config_hash: str | None = None,
    ) -> LoadSnapshotResult:
        wanted = self._read_current_generation()
        candidates: list[tuple[int, Path]] = []
        if wanted is not None:
            current_dir = self.snapshots_root / f"{wanted:012d}"
            if current_dir.exists():
                candidates.append((wanted, current_dir))
        for generation, path in reversed(self._generation_dirs()):
            if wanted is not None and generation == wanted:
                continue
            candidates.append((generation, path))
        warnings: list[str] = []
        for _, path in candidates:
            try:
                view = self._load_generation(path)
            except Exception as exc:
                warnings.append(f"ignored snapshot {path.name}: {exc}")
                continue
            if warnings:
                view.warnings.extend(warnings)
            previous_global = view.manifest.get("global_version")
            previous_config = view.manifest.get("config_hash")
            if expected_global_version is None:
                action, reason = "resume", "no expected version supplied"
            elif (
                previous_global == expected_global_version
                and previous_config == expected_config_hash
            ):
                action, reason = "resume", "version and config hash match"
            else:
                action, reason = (
                    "continue_incremental",
                    "input version or config hash changed; content-addressed caches remain usable",
                )
            return LoadSnapshotResult(
                snapshot=view,
                decision=RecoveryDecision(
                    action=action,
                    reason=reason,
                    previous_generation=view.generation,
                    previous_global_version=previous_global,
                ),
            )
        return LoadSnapshotResult(
            snapshot=None,
            decision=RecoveryDecision(
                action="cold_build",
                reason="no valid snapshot exists",
            ),
        )

    def _load_generation(self, path: Path) -> SnapshotView:
        manifest_path = path / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported snapshot schema")
        if manifest.get("generation") != int(path.name):
            raise ValueError("manifest generation does not match directory")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("manifest has no file map")
        for relative, expected_hash in files.items():
            file_path = path / relative
            if not file_path.is_file():
                raise ValueError(f"missing snapshot file: {relative}")
            raw = file_path.read_bytes()
            actual = digest(json.loads(raw.decode("utf-8")))
            if actual != expected_hash:
                raise ValueError(f"hash mismatch: {relative}")
        payload = {}
        for relative in files:
            with (path / relative).open("r", encoding="utf-8") as handle:
                payload[relative] = json.load(handle)
        stages = {
            name: payload[f"payload/stages/{name}.json"]
            for name in STAGE_NAMES
            if f"payload/stages/{name}.json" in payload
        }
        return SnapshotView(
            generation=int(manifest["generation"]),
            manifest=manifest,
            stages=stages,
            render_cache=payload.get("payload/render_cache.json", {}),
            score_cache=payload.get("payload/score_cache.json", {}),
            template_cache=payload.get("payload/template_cache.json", {}),
            pending_queue=payload.get("payload/pending_queue.json", []),
            completed_template_cursor=payload.get("payload/completed_template_cursor.json", {}),
            consumed_budget=int(payload.get("payload/consumed_budget.json", {}).get("value", 0)),
            result=payload.get("payload/result.json", {}),
            input_fingerprints=payload.get("payload/input_fingerprints.json", {}),
        )

    def commit(
        self,
        *,
        global_version: str,
        config_hash: str,
        template_chain_fingerprint: str,
        stages: Mapping[str, Any],
        render_cache: Mapping[str, Any],
        score_cache: Mapping[str, Any],
        template_cache: Mapping[str, Any],
        pending_queue: list[dict[str, Any]],
        completed_template_cursor: Mapping[str, Any],
        consumed_budget: int,
        result: Mapping[str, Any],
        input_fingerprints: Mapping[str, Any],
        previous_generation: int | None = None,
        crash_after_stage: str | None = None,
        crash_mode: str = "raise",
    ) -> int:
        with self._commit_lock():
            generation = self._next_generation()
            transaction = (
                self.state_root
                / f".txn-{generation:012d}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            )
            transaction.mkdir(parents=True, exist_ok=False)
            payloads: list[tuple[str, Any]] = []
            for stage_name in STAGE_NAMES:
                if stage_name in stages:
                    payloads.append(
                        (f"payload/stages/{stage_name}.json", stages[stage_name])
                    )
            payloads.extend(
                [
                    ("payload/render_cache.json", dict(render_cache)),
                    ("payload/score_cache.json", dict(score_cache)),
                    ("payload/template_cache.json", dict(template_cache)),
                    ("payload/pending_queue.json", list(pending_queue)),
                    ("payload/completed_template_cursor.json", dict(completed_template_cursor)),
                    ("payload/consumed_budget.json", {"value": consumed_budget}),
                    ("payload/result.json", dict(result)),
                    ("payload/input_fingerprints.json", dict(input_fingerprints)),
                ]
            )
            file_hashes: dict[str, str] = {}
            for relative, value in payloads:
                path = transaction / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                self._write_json(path, value)
                file_hashes[relative] = digest(value)
                if crash_after_stage and relative == f"payload/stages/{crash_after_stage}.json":
                    self._crash(transaction, crash_mode)
            manifest = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "generation": generation,
                "committed_at_unix_ns": time.time_ns(),
                "global_version": global_version,
                "config_hash": config_hash,
                "template_chain_fingerprint": template_chain_fingerprint,
                "previous_generation": previous_generation,
                "completed_template_revision_cursor": dict(completed_template_cursor),
                "files": dict(sorted(file_hashes.items())),
                "stages": {
                    name: {
                        "version": digest(stages[name]),
                        "file": f"payload/stages/{name}.json",
                    }
                    for name in STAGE_NAMES
                    if name in stages
                },
            }
            self._write_json(transaction / "manifest.json", manifest)
            self._fsync_directory(transaction)
            final_path = self.snapshots_root / f"{generation:012d}"
            os.replace(transaction, final_path)
            self._fsync_directory(self.snapshots_root)
            self._switch_current(final_path)
            self._prune_old_snapshots(keep=20)
            return generation

    @contextmanager
    def _commit_lock(self):
        lock_path = self.state_root / ".commit.lock"
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _crash(self, transaction: Path, crash_mode: str) -> None:
        self._fsync_directory(transaction)
        if crash_mode == "exit":
            os._exit(91)
        raise SimulatedCrash(f"simulated crash after writing {transaction}")

    def _write_json(self, path: Path, value: Any) -> None:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    def _switch_current(self, final_path: Path) -> None:
        relative_target = os.path.relpath(final_path, self.state_root)
        if self.current_path.is_dir() and not self.current_path.is_symlink():
            shutil.rmtree(self.current_path, ignore_errors=True)
        temporary = self.state_root / f".CURRENT.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            os.symlink(relative_target, temporary)
            os.replace(temporary, self.current_path)
        except OSError:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
            temp_file = self.state_root / f".CURRENT.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            with temp_file.open("w", encoding="utf-8") as handle:
                handle.write(relative_target)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_file, self.current_path)
        self._fsync_directory(self.state_root)

    def _fsync_directory(self, path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _prune_old_snapshots(self, keep: int) -> None:
        generations = self._generation_dirs()
        if len(generations) <= keep:
            return
        current = self._read_current_generation()
        for generation, path in generations[:-keep]:
            if generation == current:
                continue
            shutil.rmtree(path, ignore_errors=True)
        for path in self.state_root.glob(".txn-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


def stage_versions(stages: Mapping[str, Any]) -> dict[str, str]:
    return {name: digest(value) for name, value in sorted(stages.items())}

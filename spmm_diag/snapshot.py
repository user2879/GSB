from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any

import numpy as np

from .config import SpMMConfig
from .csr import CSRMatrix
from .diagnostics import Diagnostic, E_SNAPSHOT_CORRUPT, E_SNAPSHOT_STALE
from .engine import IncrementalEngine
from .fingerprint import digest_json
from .hot import HotBlockTable
from .kernels import BlockResult, dense_fingerprint


SCHEMA_VERSION = 1


class SnapshotError(RuntimeError):
    def __init__(self, diagnostic: Diagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


@dataclass(frozen=True)
class LoadedSnapshot:
    path: Path
    manifest: dict[str, Any]
    state: dict[str, Any]

    @property
    def global_version(self) -> int:
        return int(self.manifest["global_version"])


def input_fingerprint(
    csr: CSRMatrix,
    b_matrix: np.ndarray,
    c_matrix: np.ndarray,
    config: SpMMConfig,
) -> str:
    return digest_json(
        {
            "csr": csr.fingerprint(),
            "b": dense_fingerprint(b_matrix),
            "c": dense_fingerprint(c_matrix),
            "config": config.fingerprint(),
        }
    )


def engine_input_fingerprint(engine: IncrementalEngine) -> str:
    return input_fingerprint(engine.csr, engine.b_matrix, engine.c_matrix, engine.config)


def serialize_engine(engine: IncrementalEngine, global_version: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "global_version": int(global_version),
        "config_fingerprint": engine.config.fingerprint(),
        "input_fingerprint": engine_input_fingerprint(engine),
        "config": engine.config.to_dict(),
        "csr_fingerprint": engine.csr.fingerprint(),
        "b_fingerprint": dense_fingerprint(engine.b_matrix),
        "c_fingerprint": dense_fingerprint(engine.c_matrix),
        "run_id": engine.run_id,
        "edit_version": engine.edit_version,
        "completed_blocks": sorted(engine.output_blocks),
        "pending_blocks": sorted(engine.pending),
        "blocks": {
            str(block_id): block.to_dict()
            for block_id, block in sorted(engine.output_blocks.items())
        },
        "dependencies": {
            str(block_id): dependency.to_dict()
            for block_id, dependency in sorted(engine.dependencies.items())
        },
        "hot_table": engine.hot.to_dict(),
        "hot_event_log": [dict(item) for item in engine.hot_event_log],
        "budget_ledger": [dict(item) for item in engine.budget_ledger],
        "invalidation_events": [dict(item) for item in engine.invalidation_events],
        "recompute_events": [dict(item) for item in engine.recompute_events],
    }


class SnapshotStore:
    """All-or-nothing snapshot directory writer and validator."""

    def __init__(self, root: str | Path, *, retain_versions: int = 5) -> None:
        self.root = Path(root)
        self.current = self.root / "current"
        self.versions = self.root / "versions"
        self.retain_versions = max(1, int(retain_versions))
        self.root.mkdir(parents=True, exist_ok=True)
        self.versions.mkdir(parents=True, exist_ok=True)

    def _next_version(self) -> int:
        values: list[int] = []
        manifest = self.current / "manifest.json"
        if manifest.is_file():
            try:
                values.append(int(json.loads(manifest.read_text(encoding="utf-8"))["global_version"]))
            except Exception:
                pass
        for path in self.versions.glob("v*/manifest.json"):
            try:
                values.append(int(json.loads(path.read_text(encoding="utf-8"))["global_version"]))
            except Exception:
                pass
        return max(values, default=0) + 1

    @staticmethod
    def _sha256(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def save(self, engine: IncrementalEngine) -> LoadedSnapshot:
        version = self._next_version()
        state = serialize_engine(engine, version)
        stage = self.root / f".stage-{uuid.uuid4().hex}"
        stage.mkdir(parents=True, exist_ok=False)
        try:
            state_path = stage / "state.json"
            state_path.write_text(
                json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
                encoding="utf-8",
            )
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "global_version": version,
                "config_fingerprint": engine.config.fingerprint(),
                "input_fingerprint": state["input_fingerprint"],
                "completed_blocks": state["completed_blocks"],
                "pending_blocks": state["pending_blocks"],
                "files": {"state.json": self._sha256(state_path)},
            }
            (stage / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
                encoding="utf-8",
            )
            self._fsync_directory(stage)

            archived: Path | None = None
            if self.current.exists():
                archived = self.versions / f"v{version - 1:08d}"
                suffix = 2
                while archived.exists():
                    archived = self.versions / f"v{version - 1:08d}-{suffix}"
                    suffix += 1
                os.replace(self.current, archived)
            try:
                os.replace(stage, self.current)
            except Exception:
                if archived is not None and not self.current.exists():
                    os.replace(archived, self.current)
                raise
            self._trim_versions()
            return self._load_dir(self.current, verify_files=True)
        except Exception:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
            raise

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _trim_versions(self) -> None:
        candidates = sorted(
            (path for path in self.versions.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        for path in candidates[: max(0, len(candidates) - self.retain_versions)]:
            shutil.rmtree(path, ignore_errors=True)

    def _load_dir(self, directory: Path, *, verify_files: bool) -> LoadedSnapshot:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise SnapshotError(
                Diagnostic(
                    code=E_SNAPSHOT_CORRUPT,
                    phase="snapshot_load",
                    message="snapshot manifest is missing",
                    details={"path": str(directory)},
                )
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise SnapshotError(
                Diagnostic(
                    code=E_SNAPSHOT_CORRUPT,
                    phase="snapshot_load",
                    message=f"snapshot manifest is unreadable: {error}",
                    details={"path": str(directory)},
                )
            ) from error
        if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
            raise SnapshotError(
                Diagnostic(
                    code=E_SNAPSHOT_CORRUPT,
                    phase="snapshot_load",
                    message="snapshot schema version is unsupported",
                    details={"schema_version": manifest.get("schema_version")},
                )
            )
        files = manifest.get("files")
        if not isinstance(files, dict) or "state.json" not in files:
            raise SnapshotError(
                Diagnostic(
                    code=E_SNAPSHOT_CORRUPT,
                    phase="snapshot_load",
                    message="snapshot manifest has no state checksum",
                    details={"path": str(directory)},
                )
            )
        expected_files = set(files) | {"manifest.json"}
        actual_files = {
            path.name for path in directory.iterdir() if path.is_file()
        }
        if actual_files != expected_files:
            raise SnapshotError(
                Diagnostic(
                    code=E_SNAPSHOT_CORRUPT,
                    phase="snapshot_load",
                    message="snapshot contains missing or unexpected files",
                    details={
                        "missing": sorted(expected_files - actual_files),
                        "unexpected": sorted(actual_files - expected_files),
                    },
                )
            )
        if verify_files:
            for name, expected_hash in files.items():
                file_path = directory / name
                if not file_path.is_file() or self._sha256(file_path) != expected_hash:
                    raise SnapshotError(
                        Diagnostic(
                            code=E_SNAPSHOT_CORRUPT,
                            phase="snapshot_load",
                            message=f"snapshot file checksum mismatch: {name}",
                            details={"file": name, "path": str(file_path)},
                        )
                    )
        try:
            state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        except Exception as error:
            raise SnapshotError(
                Diagnostic(
                    code=E_SNAPSHOT_CORRUPT,
                    phase="snapshot_load",
                    message=f"snapshot state is unreadable: {error}",
                    details={"path": str(directory)},
                )
            ) from error
        if int(state.get("global_version", -1)) != int(manifest.get("global_version", -2)):
            raise SnapshotError(
                Diagnostic(
                    code=E_SNAPSHOT_CORRUPT,
                    phase="snapshot_load",
                    message="manifest and state global versions disagree",
                    details={
                        "manifest_version": manifest.get("global_version"),
                        "state_version": state.get("global_version"),
                    },
                )
            )
        return LoadedSnapshot(path=directory, manifest=manifest, state=state)

    def load_current(
        self,
        *,
        expected_config_fingerprint: str | None = None,
        expected_input_fingerprint: str | None = None,
        expected_version: int | None = None,
    ) -> LoadedSnapshot:
        loaded = self._load_dir(self.current, verify_files=True)
        reasons: list[str] = []
        if expected_version is not None and loaded.global_version != expected_version:
            reasons.append("version")
        if (
            expected_config_fingerprint is not None
            and loaded.manifest.get("config_fingerprint") != expected_config_fingerprint
        ):
            reasons.append("config_fingerprint")
        if (
            expected_input_fingerprint is not None
            and loaded.manifest.get("input_fingerprint") != expected_input_fingerprint
        ):
            reasons.append("input_fingerprint")
        if reasons:
            raise SnapshotError(
                Diagnostic(
                    code=E_SNAPSHOT_STALE,
                    phase="snapshot_validation",
                    message="snapshot is stale for the requested inputs",
                    details={
                        "reasons": reasons,
                        "snapshot_version": loaded.global_version,
                        "snapshot_config_fingerprint": loaded.manifest.get("config_fingerprint"),
                        "expected_config_fingerprint": expected_config_fingerprint,
                        "snapshot_input_fingerprint": loaded.manifest.get("input_fingerprint"),
                        "expected_input_fingerprint": expected_input_fingerprint,
                    },
                )
            )
        return loaded

    def load_latest_previous_valid(self, *, exclude: Path | None = None) -> LoadedSnapshot | None:
        candidates = sorted(
            (path for path in self.versions.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        for path in candidates:
            if exclude is not None and path.resolve() == exclude.resolve():
                continue
            try:
                return self._load_dir(path, verify_files=True)
            except SnapshotError:
                continue
        return None


def restore_engine_from_state(
    loaded: LoadedSnapshot,
    csr: CSRMatrix,
    b_matrix: np.ndarray,
    c_matrix: np.ndarray,
    config: SpMMConfig,
    *,
    bootstrap_diagnostics: list[Diagnostic] | None = None,
) -> tuple[IncrementalEngine, dict[str, Any]]:
    engine = IncrementalEngine(
        csr,
        b_matrix,
        c_matrix,
        config,
        bootstrap_diagnostics=bootstrap_diagnostics or (),
    )
    state = loaded.state
    reusable: list[int] = []
    rejected: list[int] = []
    raw_blocks = state.get("blocks", {})
    for block_id, (row_start, row_end) in enumerate(engine.block_ranges):
        raw = raw_blocks.get(str(block_id))
        if raw is None:
            continue
        block = BlockResult.from_dict(raw)
        dependency = engine.dependencies[block_id]
        if (
            block.row_start == row_start
            and block.row_end == row_end
            and block.dependency_fingerprint == dependency.input_fingerprint
            and block.precision_fingerprint == config.precision_fingerprint()
        ):
            engine.output_blocks[block_id] = block
            engine.dependencies[block_id] = dependency.with_output(block.output_fingerprint)
            reusable.append(block_id)
        else:
            rejected.append(block_id)

    state_config = SpMMConfig.from_dict(state["config"])
    if state_config.row_block_size == config.row_block_size:
        try:
            hot = HotBlockTable.from_dict(state.get("hot_table", {}))
            engine.hot = HotBlockTable(config.hot_capacity, hot.entries())
        except Exception:
            engine.hot = HotBlockTable(config.hot_capacity)
    else:
        engine.hot = HotBlockTable(config.hot_capacity)

    engine.budget_ledger = [dict(item) for item in state.get("budget_ledger", [])]
    engine.invalidation_events = [dict(item) for item in state.get("invalidation_events", [])]
    engine.recompute_events = [dict(item) for item in state.get("recompute_events", [])]
    if "hot_event_log" in state:
        engine.hot_event_log = [dict(item) for item in state.get("hot_event_log", [])]
    if state_config.row_block_size != config.row_block_size:
        engine.hot_event_log = [{"type": "clear", "reason": "restored_topology_changed"}]
    elif state_config.hot_capacity != config.hot_capacity:
        engine.hot_event_log.append(
            {
                "type": "resize",
                "capacity": config.hot_capacity,
                "entries": engine.hot.to_dict()["entries"],
            }
        )
    engine.run_id = int(state.get("run_id", 0))
    engine.edit_version = int(state.get("edit_version", 0))
    engine.pending = set(engine.current_block_ids) - set(reusable)
    engine.invalid_blocks = set(engine.pending)
    return engine, {
        "snapshot_version": loaded.global_version,
        "completed_blocks": sorted(reusable),
        "rejected_blocks": sorted(rejected),
        "pending_blocks": sorted(engine.pending),
    }


def bootstrap_engine(
    store: SnapshotStore,
    csr: CSRMatrix,
    b_matrix: np.ndarray,
    c_matrix: np.ndarray,
    config: SpMMConfig,
    *,
    expected_version: int | None = None,
) -> tuple[IncrementalEngine, dict[str, Any]]:
    expected_config = config.fingerprint()
    expected_input = input_fingerprint(csr, b_matrix, c_matrix, config)
    diagnostics: list[Diagnostic] = []
    current_error: SnapshotError | None = None
    if store.current.exists():
        try:
            loaded = store.load_current(
                expected_config_fingerprint=expected_config,
                expected_input_fingerprint=expected_input,
                expected_version=expected_version,
            )
            engine, info = restore_engine_from_state(loaded, csr, b_matrix, c_matrix, config)
            info["source"] = "current"
            return engine, info
        except SnapshotError as error:
            current_error = error
            diagnostics.append(error.diagnostic)

    previous = store.load_latest_previous_valid()
    if previous is not None:
        engine, info = restore_engine_from_state(
            previous,
            csr,
            b_matrix,
            c_matrix,
            config,
            bootstrap_diagnostics=diagnostics,
        )
        info["source"] = "previous_version"
        info["current_error"] = current_error.diagnostic.to_dict() if current_error else None
        return engine, info

    engine = IncrementalEngine(
        csr,
        b_matrix,
        c_matrix,
        config,
        bootstrap_diagnostics=diagnostics,
    )
    return engine, {
        "source": "empty_rebuild",
        "snapshot_version": None,
        "completed_blocks": [],
        "rejected_blocks": [],
        "pending_blocks": sorted(engine.pending),
        "current_error": current_error.diagnostic.to_dict() if current_error else None,
    }

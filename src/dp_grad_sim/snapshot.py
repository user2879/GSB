from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from .errors import SimError
from .retention import FidelityEntry
from .util import atomic_write_json, read_json, sha256_file


FORMAT_VERSION = 1


@dataclass
class SavedState:
    version: int = 0
    phase: str = ""
    completed_steps: int = 0
    window_index: int = -1
    window_start: int = 0
    window_end: int = 0
    params: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    grads: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    packets: dict[int, dict[str, bytes]] = field(default_factory=dict)
    received_q: dict[tuple[int, int, str], np.ndarray] = field(default_factory=dict)
    aggregate_fixed: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    selected: dict[str, Any] | None = None
    quant_params: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] | None = None
    send_records: list[dict[str, Any]] = field(default_factory=list)
    current_errors: list[dict[str, Any]] = field(default_factory=list)
    budget_snapshot: dict[str, Any] = field(default_factory=dict)
    window_records: list[dict[str, Any]] = field(default_factory=list)
    retention_events: list[dict[str, Any]] = field(default_factory=list)
    fidelity_entries: list[FidelityEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict, compare=False)


def _safe_name(name: str) -> str:
    return quote(name, safe="")


def _fsync_path(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass


class SnapshotManager:
    def __init__(
        self,
        root: str | Path,
        *,
        config_hash: str,
        topology_hash: str,
        data_hash: str,
        snapshot_interval_steps: int,
        fidelity_generation_limit: int,
    ) -> None:
        self.root = Path(root)
        self.commits_dir = self.root / "commits"
        self.manifest_path = self.root / "manifest.json"
        self.config_hash = config_hash
        self.topology_hash = topology_hash
        self.data_hash = data_hash
        self.snapshot_interval_steps = snapshot_interval_steps
        self.fidelity_generation_limit = fidelity_generation_limit
        self.root.mkdir(parents=True, exist_ok=True)
        self.commits_dir.mkdir(parents=True, exist_ok=True)
        self._next_version = 1

    def _validate_manifest_files(self, commit_dir: Path, manifest: dict[str, Any]) -> None:
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise SimError("E_SNAPSHOT_CORRUPT", "snapshot manifest has no file inventory")
        commit_root = commit_dir.resolve()
        for relative, expected_hash in files.items():
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise SimError("E_SNAPSHOT_CORRUPT", "snapshot file inventory is malformed")
            candidate = (commit_dir / relative).resolve()
            if commit_root not in candidate.parents and candidate != commit_root:
                raise SimError("E_SNAPSHOT_CORRUPT", f"unsafe snapshot path {relative!r}")
            if not candidate.is_file():
                raise SimError("E_SNAPSHOT_CORRUPT", f"snapshot file is missing: {relative}")
            if sha256_file(candidate) != expected_hash:
                raise SimError("E_SNAPSHOT_CORRUPT", f"snapshot file checksum mismatch: {relative}")

    def _load_commit_dir(self, commit_dir: Path) -> SavedState:
        manifest_path = commit_dir / "manifest.json"
        if not manifest_path.is_file():
            raise SimError("E_SNAPSHOT_CORRUPT", "snapshot commit is missing manifest.json")
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT_VERSION:
            raise SimError("E_SNAPSHOT_CORRUPT", "invalid snapshot manifest format")
        self._validate_manifest_files(commit_dir, manifest)
        state_file = manifest.get("state_file")
        if not isinstance(state_file, str):
            raise SimError("E_SNAPSHOT_CORRUPT", "snapshot manifest has no state file")
        state_path = commit_dir / state_file
        if sha256_file(state_path) != manifest.get("state_sha256"):
            raise SimError("E_SNAPSHOT_CORRUPT", "state checksum mismatch")
        state = read_json(state_path)
        if not isinstance(state, dict):
            raise SimError("E_SNAPSHOT_CORRUPT", "snapshot state must be an object")

        def load_array(relative: str) -> np.ndarray:
            path = commit_dir / relative
            try:
                array = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise SimError("E_SNAPSHOT_CORRUPT", f"cannot load snapshot array {relative}") from exc
            return np.asarray(array)

        params: dict[int, dict[str, np.ndarray]] = {}
        for worker_text, tensors in state.get("params", {}).items():
            params[int(worker_text)] = {
                name: load_array(relative) for name, relative in tensors.items()
            }
        grads: dict[int, dict[str, np.ndarray]] = {}
        for worker_text, tensors in state.get("grads", {}).items():
            grads[int(worker_text)] = {
                name: load_array(relative) for name, relative in tensors.items()
            }
        packets: dict[int, dict[str, bytes]] = {}
        for worker_text, tensors in state.get("packets", {}).items():
            packets[int(worker_text)] = {
                name: (commit_dir / relative).read_bytes() for name, relative in tensors.items()
            }
        received_q: dict[tuple[int, int, str], np.ndarray] = {}
        for dst_text, origins in state.get("received_q", {}).items():
            for origin_text, tensors in origins.items():
                for name, relative in tensors.items():
                    received_q[(int(dst_text), int(origin_text), name)] = load_array(relative)
        aggregate_fixed: dict[int, dict[str, np.ndarray]] = {}
        for worker_text, tensors in state.get("aggregate_fixed", {}).items():
            aggregate_fixed[int(worker_text)] = {
                name: load_array(relative) for name, relative in tensors.items()
            }
        fidelity_entries: list[FidelityEntry] = []
        for entry in state.get("fidelity", []):
            fidelity_entries.append(
                FidelityEntry(
                    window=int(entry["window"]),
                    step=int(entry["step"]),
                    gradient_id=str(entry["gradient_id"]),
                    tensors={
                        name: load_array(relative)
                        for name, relative in entry.get("tensors", {}).items()
                    },
                )
            )
        return SavedState(
            version=int(manifest.get("version", 0)),
            phase=str(state.get("phase", "")),
            completed_steps=int(state.get("completed_steps", 0)),
            window_index=int(state.get("window_index", -1)),
            window_start=int(state.get("window_start", 0)),
            window_end=int(state.get("window_end", 0)),
            params=params,
            grads=grads,
            packets=packets,
            received_q=received_q,
            aggregate_fixed=aggregate_fixed,
            selected=state.get("selected"),
            quant_params=state.get("quant_params", {}),
            decision=state.get("decision"),
            send_records=state.get("send_records", []),
            current_errors=state.get("current_errors", []),
            budget_snapshot=state.get("budget_snapshot", {}),
            window_records=state.get("window_records", []),
            retention_events=state.get("retention_events", []),
            fidelity_entries=fidelity_entries,
            warnings=list(state.get("warnings", [])),
            manifest=manifest,
        )

    def _load_pointer(self) -> tuple[SavedState | None, list[str]]:
        warnings: list[str] = []
        if self.manifest_path.is_file():
            try:
                pointer = read_json(self.manifest_path)
                commit_name = pointer["commit"]
                commit_dir = self.root / commit_name
                manifest_path = commit_dir / "manifest.json"
                if not manifest_path.is_file():
                    raise SimError("E_SNAPSHOT_CORRUPT", "pointer targets a missing commit")
                if sha256_file(manifest_path) != pointer.get("manifest_sha256"):
                    raise SimError("E_SNAPSHOT_CORRUPT", "commit manifest checksum mismatch")
                state = self._load_commit_dir(commit_dir)
                self._next_version = max(self._next_version, state.version + 1)
                return state, warnings
            except (OSError, KeyError, json.JSONDecodeError, SimError) as exc:
                warnings.append(f"E_SNAPSHOT_POINTER_FALLBACK: {exc}")

        candidates: list[Path] = []
        for path in self.commits_dir.iterdir():
            if path.is_dir() and (path / "manifest.json").is_file():
                candidates.append(path)
        candidates.sort(key=lambda path: path.name, reverse=True)
        for commit_dir in candidates:
            try:
                state = self._load_commit_dir(commit_dir)
                warnings.append(
                    f"E_SNAPSHOT_RECOVERED_FALLBACK: recovered complete commit {commit_dir.name}"
                )
                self._next_version = max(self._next_version, state.version + 1)
                return state, warnings
            except (OSError, KeyError, json.JSONDecodeError, SimError) as exc:
                warnings.append(f"E_SNAPSHOT_SKIP_CORRUPT: {commit_dir.name}: {exc}")
        return None, warnings

    def load_latest(self) -> SavedState | None:
        state, warnings = self._load_pointer()
        if state is not None:
            state.warnings = sorted(set(state.warnings + warnings))
        return state

    def validate_compatibility(self, state: SavedState | None) -> None:
        if state is None:
            return
        manifest = state.manifest
        if manifest.get("config_hash") != self.config_hash:
            raise SimError(
                "E_CONFIG_MISMATCH",
                "configuration hash differs from the latest snapshot; refusing to resume",
                expected=manifest.get("config_hash"),
                actual=self.config_hash,
            )
        if manifest.get("topology_hash") != self.topology_hash:
            raise SimError(
                "E_CONFIG_MISMATCH",
                "topology hash differs from the latest snapshot; refusing to resume",
                expected=manifest.get("topology_hash"),
                actual=self.topology_hash,
            )
        if manifest.get("data_hash") != self.data_hash:
            raise SimError(
                "E_CONFIG_MISMATCH",
                "data shard hash differs from the latest snapshot; refusing to resume",
                expected=manifest.get("data_hash"),
                actual=self.data_hash,
            )

    def _write_array(self, staging: Path, relative: str, array: np.ndarray) -> str:
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(array), allow_pickle=False)
        _fsync_path(path)
        return relative

    def commit(self, state: SavedState) -> SavedState:
        version = self._next_version
        self._next_version += 1
        commit_name = f"v{version:08d}-{uuid.uuid4().hex[:12]}"
        staging = self.root / f".staging-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        state_payload: dict[str, Any] = {
            "format": FORMAT_VERSION,
            "phase": state.phase,
            "completed_steps": state.completed_steps,
            "window_index": state.window_index,
            "window_start": state.window_start,
            "window_end": state.window_end,
            "selected": state.selected,
            "quant_params": state.quant_params,
            "decision": state.decision,
            "send_records": state.send_records,
            "current_errors": state.current_errors,
            "budget_snapshot": state.budget_snapshot,
            "window_records": state.window_records,
            "retention_events": state.retention_events,
            "warnings": state.warnings,
            "params": {},
            "grads": {},
            "packets": {},
            "received_q": {},
            "aggregate_fixed": {},
            "fidelity": [],
        }
        for worker_id in sorted(state.params):
            state_payload["params"][str(worker_id)] = {}
            for name in sorted(state.params[worker_id]):
                relative = f"arrays/params/w{worker_id}/{_safe_name(name)}.npy"
                state_payload["params"][str(worker_id)][name] = self._write_array(
                    staging, relative, state.params[worker_id][name]
                )
        for worker_id in sorted(state.grads):
            state_payload["grads"][str(worker_id)] = {}
            for name in sorted(state.grads[worker_id]):
                relative = f"arrays/grads/w{worker_id}/{_safe_name(name)}.npy"
                state_payload["grads"][str(worker_id)][name] = self._write_array(
                    staging, relative, state.grads[worker_id][name]
                )
        for worker_id in sorted(state.packets):
            state_payload["packets"][str(worker_id)] = {}
            for name in sorted(state.packets[worker_id]):
                relative = f"packets/w{worker_id}/{_safe_name(name)}.bin"
                path = staging / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(state.packets[worker_id][name])
                _fsync_path(path)
                state_payload["packets"][str(worker_id)][name] = relative
        for (dst, origin, name), array in sorted(state.received_q.items()):
            dst_payload = state_payload["received_q"].setdefault(str(dst), {})
            origin_payload = dst_payload.setdefault(str(origin), {})
            relative = f"arrays/received_q/w{dst}/o{origin}/{_safe_name(name)}.npy"
            origin_payload[name] = self._write_array(staging, relative, array)
        for worker_id in sorted(state.aggregate_fixed):
            state_payload["aggregate_fixed"][str(worker_id)] = {}
            for name in sorted(state.aggregate_fixed[worker_id]):
                relative = f"arrays/aggregate_fixed/w{worker_id}/{_safe_name(name)}.npy"
                state_payload["aggregate_fixed"][str(worker_id)][name] = self._write_array(
                    staging, relative, state.aggregate_fixed[worker_id][name]
                )
        for entry in state.fidelity_entries:
            entry_payload = {
                "window": entry.window,
                "step": entry.step,
                "gradient_id": entry.gradient_id,
                "tensors": {},
            }
            for name in sorted(entry.tensors):
                relative = f"arrays/fidelity/{entry.gradient_id}/{_safe_name(name)}.npy"
                entry_payload["tensors"][name] = self._write_array(
                    staging, relative, entry.tensors[name]
                )
            state_payload["fidelity"].append(entry_payload)

        state_path = staging / "state.json"
        state_path.write_text(
            json.dumps(state_payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _fsync_path(state_path)
        files: dict[str, str] = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                relative = path.relative_to(staging).as_posix()
                files[relative] = sha256_file(path)
        manifest = {
            "format": FORMAT_VERSION,
            "version": version,
            "phase": state.phase,
            "completed_steps": state.completed_steps,
            "window_index": state.window_index,
            "window_start": state.window_start,
            "window_end": state.window_end,
            "config_hash": self.config_hash,
            "topology_hash": self.topology_hash,
            "data_hash": self.data_hash,
            "snapshot_interval_steps": self.snapshot_interval_steps,
            "fidelity_generation_limit": self.fidelity_generation_limit,
            "retained_fidelity_generations": len(state.fidelity_entries),
            "state_file": "state.json",
            "state_sha256": sha256_file(state_path),
            "files": files,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _fsync_path(manifest_path)
        final_dir = self.commits_dir / commit_name
        os.replace(staging, final_dir)
        pointer = {
            "format": FORMAT_VERSION,
            "version": version,
            "phase": state.phase,
            "completed_steps": state.completed_steps,
            "commit": f"commits/{commit_name}",
            "manifest_sha256": sha256_file(final_dir / "manifest.json"),
        }
        atomic_write_json(self.manifest_path, pointer)
        state.version = version
        state.manifest = manifest
        return state

    def write_report(self, report: dict[str, Any]) -> Path:
        report_dir = self.root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        latest = report_dir / "latest.json"
        atomic_write_json(latest, report)
        return latest

    def remove_staging_dirs(self) -> None:
        for path in self.root.glob(".staging-*"):
            try:
                shutil.rmtree(path)
            except OSError:
                pass

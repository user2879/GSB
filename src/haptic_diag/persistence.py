"""Atomic workspace snapshots and crash recovery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_json_bytes, canonical_json_text, sha256_bytes, sha256_file, stable_hash
from .engine import EngineState

GENERATION_PATTERN = re.compile(r"^gen-(\d{6})-[0-9a-f]{12}$")


class PersistenceError(RuntimeError):
    pass


class InjectedCommitCrash(PersistenceError):
    pass


@dataclass(frozen=True)
class OrphanRecord:
    path: str
    classification: str
    reason: str
    generation: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "classification": self.classification,
            "reason": self.reason,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class CommitResult:
    generation: str
    state_hash: str
    protocol_version: int
    file_hashes: dict[str, str]
    current_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "state_hash": self.state_hash,
            "protocol_version": self.protocol_version,
            "file_hashes": dict(self.file_hashes),
            "current_path": self.current_path,
        }


@dataclass(frozen=True)
class RecoveryResult:
    state: EngineState | None
    current_generation: str | None
    state_hash: str | None
    orphans: tuple[OrphanRecord, ...]
    superseded: tuple[OrphanRecord, ...]
    recovery_log_path: str | None
    reason: str

    @property
    def recovered(self) -> bool:
        return self.state is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recovered": self.recovered,
            "current_generation": self.current_generation,
            "state_hash": self.state_hash,
            "orphans": [item.to_dict() for item in self.orphans],
            "superseded": [item.to_dict() for item in self.superseded],
            "recovery_log_path": self.recovery_log_path,
            "reason": self.reason,
            "protocol_version": None if self.state is None else self.state.protocol_version,
        }


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_bytes_fsync(path: Path, data: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


class AtomicWorkspaceStore:
    """Atomic generation store with a single committed-state pointer.

    A generation directory becomes committed only after every file is fsynced,
    its manifest is written, and the CURRENT pointer is atomically replaced.
    Directories left behind by a crash are never restored implicitly.
    """

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.state_dir = self.workspace / "state"
        self.commits_dir = self.state_dir / "commits"
        self.orphans_dir = self.state_dir / "orphans"
        self.history_dir = self.state_dir / "history"
        self.current_path = self.state_dir / "CURRENT"
        self.recovery_log_path = self.state_dir / "recovery_log.json"

    def _ensure_dirs(self) -> None:
        self.commits_dir.mkdir(parents=True, exist_ok=True)
        self.orphans_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _next_generation_prefix(self) -> int:
        maximum = 0
        if self.commits_dir.exists():
            for candidate in self.commits_dir.iterdir():
                match = GENERATION_PATTERN.match(candidate.name)
                if match:
                    maximum = max(maximum, int(match.group(1)))
        return maximum + 1

    @staticmethod
    def _generation_sequence(generation: str | None) -> int | None:
        if generation is None:
            return None
        match = GENERATION_PATTERN.match(generation)
        return None if match is None else int(match.group(1))

    def _snapshot_files(self, state: EngineState) -> dict[str, bytes]:
        state_dict = state.to_dict()
        state_dict["reuse_source"] = "stable-commit"
        graph = state_dict["graph"]
        frames = state_dict["frames"]
        digests = state_dict["digests"]
        ledger = {
            "active": state_dict["ledger"],
            "history": state_dict["ledger_history"],
        }
        meta = {
            "schema_version": 1,
            "protocol_version": state.protocol_version,
            "protocol_version_stamp": f"v{state.protocol_version}",
            "profile": state_dict["profile"],
            "report": state_dict["report"],
            "reuse_source": state_dict["reuse_source"],
            "full_state_hash": stable_hash(state_dict),
        }
        frame_hashes = {str(frame["seq"]): frame["content_hash"] for frame in frames}
        return {
            "state_meta.json": canonical_json_text(meta).encode("utf-8"),
            "graph.json": canonical_json_text(graph).encode("utf-8"),
            "frames.json": canonical_json_text(frames).encode("utf-8"),
            "frame_hashes.json": canonical_json_text(frame_hashes).encode("utf-8"),
            "digests.json": canonical_json_text(digests).encode("utf-8"),
            "dependency_graph.json": canonical_json_text(state_dict["dependency_graph"]).encode("utf-8"),
            "ledger.json": canonical_json_text(ledger).encode("utf-8"),
            "last_update.json": canonical_json_text(state_dict["last_update"]).encode("utf-8"),
        }

    def commit(
        self,
        state: EngineState,
        crash_after_files: int | None = None,
        hard_crash: bool = False,
    ) -> CommitResult:
        self._ensure_dirs()
        files = self._snapshot_files(state)
        state_dict = state.to_dict()
        state_dict["reuse_source"] = "stable-commit"
        state_hash = stable_hash(state_dict)
        expected_meta_hash = stable_hash(state_dict)
        meta = json.loads(files["state_meta.json"])
        if meta["full_state_hash"] != expected_meta_hash:
            raise AssertionError("snapshot state hash mismatch")
        sequence = self._next_generation_prefix()
        generation = f"gen-{sequence:06d}-{state_hash[:12]}"
        temporary = self.commits_dir / f".tmp-{uuid.uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        written_hashes: dict[str, str] = {}
        try:
            for index, (name, data) in enumerate(sorted(files.items()), start=1):
                _write_bytes_fsync(temporary / name, data)
                written_hashes[name] = sha256_bytes(data)
                if crash_after_files is not None and index >= crash_after_files:
                    if hard_crash:
                        os._exit(137)
                    raise InjectedCommitCrash(
                        f"injected crash after {index} snapshot file(s)"
                    )
            manifest = {
                "schema_version": 1,
                "complete": True,
                "generation": generation,
                "state_hash": state_hash,
                "protocol_version": state.protocol_version,
                "protocol_version_stamp": f"v{state.protocol_version}",
                "files": written_hashes,
            }
            manifest_bytes = canonical_json_text(manifest).encode("utf-8")
            _write_bytes_fsync(temporary / "COMMITTED.json", manifest_bytes)
            written_hashes["COMMITTED.json"] = sha256_bytes(manifest_bytes)
            _fsync_directory(temporary)
            final = self.commits_dir / generation
            if final.exists():
                raise PersistenceError(f"generation already exists: {generation}")
            os.replace(temporary, final)
            _fsync_directory(self.commits_dir)
            pointer_tmp = self.state_dir / f".CURRENT-{uuid.uuid4().hex}.tmp"
            _write_bytes_fsync(pointer_tmp, (generation + "\n").encode("ascii"))
            os.replace(pointer_tmp, self.current_path)
            _fsync_directory(self.state_dir)
        except BaseException:
            if temporary.exists():
                # Leave the orphan for the explicit recovery pass; never point at it.
                pass
            raise
        return CommitResult(
            generation=generation,
            state_hash=state_hash,
            protocol_version=state.protocol_version,
            file_hashes=written_hashes,
            current_path=str(self.current_path),
        )

    def _read_current_generation(self) -> str | None:
        if not self.current_path.exists():
            return None
        raw = self.current_path.read_text(encoding="ascii").strip()
        if not raw or "/" in raw or "\\" in raw or raw in {".", ".."}:
            return None
        return raw

    def _validate_manifest(self, directory: Path) -> dict[str, Any]:
        manifest_path = directory / "COMMITTED.json"
        if not manifest_path.exists():
            raise PersistenceError("COMMITTED.json is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise PersistenceError("manifest is not marked complete")
        file_hashes = manifest.get("files")
        if not isinstance(file_hashes, dict):
            raise PersistenceError("manifest files mapping is missing")
        for name, expected_hash in file_hashes.items():
            path = directory / name
            if not path.exists():
                raise PersistenceError(f"snapshot file is missing: {name}")
            actual = sha256_file(path)
            if actual != expected_hash:
                raise PersistenceError(f"snapshot file hash mismatch: {name}")
        return manifest

    def _state_from_directory(self, directory: Path, manifest: Mapping[str, Any]) -> EngineState:
        def load(name: str) -> Any:
            return json.loads((directory / name).read_text(encoding="utf-8"))

        meta = load("state_meta.json")
        graph = load("graph.json")
        frames = load("frames.json")
        frame_hashes = load("frame_hashes.json")
        for frame in frames:
            seq_key = str(frame["seq"])
            if frame_hashes.get(seq_key) != frame.get("content_hash"):
                raise PersistenceError(f"frame hash ledger mismatch at sequence {seq_key}")
        digests = load("digests.json")
        dependency_graph = load("dependency_graph.json")
        ledger_bundle = load("ledger.json")
        last_update = load("last_update.json")
        state_dict = {
            "schema_version": 1,
            "graph": graph,
            "protocol_version": int(meta["protocol_version"]),
            "profile": meta["profile"],
            "frames": frames,
            "digests": digests,
            "report": meta["report"],
            "dependency_graph": dependency_graph,
            "ledger": ledger_bundle["active"],
            "ledger_history": ledger_bundle["history"],
            "last_update": last_update,
            "reuse_source": "stable-commit",
        }
        actual_state_hash = stable_hash(state_dict)
        if actual_state_hash != manifest.get("state_hash"):
            raise PersistenceError("full state hash does not match manifest")
        if int(meta["protocol_version"]) != int(manifest["protocol_version"]):
            raise PersistenceError("protocol version stamp mismatch")
        state = EngineState.from_dict(state_dict)
        if state.ledger.protocol_version != state.protocol_version:
            raise PersistenceError("active ledger protocol version mismatch")
        return state

    def load_committed(self) -> tuple[EngineState, str, str] | None:
        generation = self._read_current_generation()
        if generation is None:
            return None
        directory = self.commits_dir / generation
        if not directory.is_dir():
            return None
        try:
            manifest = self._validate_manifest(directory)
            state = self._state_from_directory(directory, manifest)
        except Exception:
            return None
        if manifest.get("generation") != generation:
            return None
        return state, generation, str(manifest["state_hash"])

    def _append_recovery_log(self, records: Iterable[OrphanRecord]) -> str | None:
        records = tuple(records)
        if not records:
            return None
        existing: list[dict[str, Any]] = []
        if self.recovery_log_path.exists():
            try:
                value = json.loads(self.recovery_log_path.read_text(encoding="utf-8"))
                if isinstance(value, list):
                    existing = value
            except Exception:
                existing = []
        now = datetime.now(timezone.utc).isoformat()
        existing.extend({**record.to_dict(), "detected_at": now} for record in records)
        temporary = self.recovery_log_path.with_suffix(".json.tmp")
        _write_bytes_fsync(temporary, canonical_json_text(existing).encode("utf-8"))
        os.replace(temporary, self.recovery_log_path)
        return str(self.recovery_log_path)

    def recover(self) -> RecoveryResult:
        self._ensure_dirs()
        current_generation = self._read_current_generation()
        loaded = self.load_committed()
        current_sequence = self._generation_sequence(current_generation)
        orphans: list[OrphanRecord] = []
        superseded: list[OrphanRecord] = []
        for candidate in sorted(self.commits_dir.iterdir(), key=lambda path: path.name):
            if not candidate.is_dir():
                continue
            if current_generation is not None and candidate.name == current_generation and loaded is not None:
                continue
            try:
                manifest = self._validate_manifest(candidate)
                candidate_generation = str(manifest.get("generation"))
                candidate_sequence = self._generation_sequence(candidate_generation)
                if (
                    loaded is not None
                    and current_sequence is not None
                    and candidate_sequence is not None
                    and candidate_sequence < current_sequence
                ):
                    record = OrphanRecord(
                        path=str(candidate),
                        classification="superseded",
                        reason="older complete commit has been superseded by CURRENT",
                        generation=candidate_generation,
                    )
                    superseded.append(record)
                    destination = self.history_dir / candidate.name
                else:
                    record = OrphanRecord(
                        path=str(candidate),
                        classification="orphan",
                        reason=(
                            "complete snapshot was never referenced by CURRENT"
                            if candidate.name != current_generation
                            else "CURRENT snapshot failed full validation"
                        ),
                        generation=candidate_generation,
                    )
                    orphans.append(record)
                    destination = self.orphans_dir / candidate.name
            except Exception as exc:
                record = OrphanRecord(
                    path=str(candidate),
                    classification="orphan",
                    reason=f"uncommitted or invalid snapshot: {exc}",
                    generation=candidate.name if GENERATION_PATTERN.match(candidate.name) else None,
                )
                orphans.append(record)
                destination = self.orphans_dir / candidate.name
            if destination.exists():
                destination = destination.with_name(destination.name + "-" + uuid.uuid4().hex[:8])
            os.replace(candidate, destination)
        for pointer_tmp in self.state_dir.glob(".CURRENT-*.tmp"):
            record = OrphanRecord(
                path=str(pointer_tmp),
                classification="orphan",
                reason="unreferenced CURRENT pointer temporary file",
                generation=None,
            )
            orphans.append(record)
            destination = self.orphans_dir / pointer_tmp.name
            if destination.exists():
                destination = destination.with_name(destination.name + "-" + uuid.uuid4().hex[:8])
            os.replace(pointer_tmp, destination)
        log_path = self._append_recovery_log((*orphans, *superseded))
        if loaded is None:
            return RecoveryResult(
                state=None,
                current_generation=current_generation,
                state_hash=None,
                orphans=tuple(orphans),
                superseded=tuple(superseded),
                recovery_log_path=log_path,
                reason="no valid state referenced by CURRENT",
            )
        state, generation, state_hash = loaded
        return RecoveryResult(
            state=state,
            current_generation=generation,
            state_hash=state_hash,
            orphans=tuple(orphans),
            superseded=tuple(superseded),
            recovery_log_path=log_path,
            reason="restored most recent committed generation",
        )

    def clear(self) -> None:
        if self.state_dir.exists():
            shutil.rmtree(self.state_dir)

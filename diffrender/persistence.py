"""Atomic snapshots, version pointers, and force-termination recovery."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import DiffRenderError, E_SNAPSHOT_VERSION
from .hashing import canonical_json_bytes, canonical_json_hash, sha256_bytes


POINTER_SCHEMA = "diffrender-pointer-v1"
CLOSED_SCHEMA = "diffrender-closed-v1"
RECOVERY_SCHEMA = "diffrender-recovery-v1"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


@dataclass(frozen=True)
class RecoveryReport:
    clean_shutdown: bool
    forced_termination_detected: bool
    recovered_snapshot_version: int
    recovered_generation: int
    orphans: tuple[dict[str, Any], ...]
    fabricated_exit_report: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean_shutdown": self.clean_shutdown,
            "forced_termination_detected": self.forced_termination_detected,
            "recovered_snapshot_version": self.recovered_snapshot_version,
            "recovered_generation": self.recovered_generation,
            "orphans": list(self.orphans),
            "fabricated_exit_report": self.fabricated_exit_report,
        }


class StateStore:
    """Generation-based commit store.

    A new snapshot is written to an immutable generation file, then CURRENT is
    atomically replaced.  Readers never consult staging or old generation files.
    """

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace)
        self.state_root = self.workspace / "state"
        self.committed_dir = self.state_root / "committed"
        self.staging_dir = self.state_root / "staging"
        self.orphan_dir = self.state_root / "orphans"
        for directory in (self.committed_dir, self.staging_dir, self.orphan_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.pointer_path = self.committed_dir / "CURRENT.json"
        self.closed_path = self.committed_dir / "CLOSED.json"

    # ------------------------------------------------------------------
    # Read primitives

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            with open(path, "rb") as handle:
                raw = handle.read()
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON root is not an object")
            return value
        except FileNotFoundError:
            raise DiffRenderError(E_SNAPSHOT_VERSION, f"missing snapshot file {path.name}")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise DiffRenderError(E_SNAPSHOT_VERSION, f"cannot read {path.name}: {exc}") from exc

    def pointer(self) -> dict[str, Any] | None:
        if not self.pointer_path.exists():
            return None
        pointer = self._read_json(self.pointer_path)
        if pointer.get("schema") != POINTER_SCHEMA:
            raise DiffRenderError(E_SNAPSHOT_VERSION, "unsupported CURRENT pointer schema")
        return pointer

    def read_snapshot(
        self,
        expected_snapshot_version: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        pointer = self.pointer()
        if pointer is None:
            return None
        try:
            generation = int(pointer["generation"])
            snapshot_version = int(pointer["snapshot_version"])
            filename = str(pointer["snapshot_file"])
            expected_hash = str(pointer["sha256"])
            closed = bool(pointer["closed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DiffRenderError(E_SNAPSHOT_VERSION, f"invalid CURRENT pointer: {exc}") from exc
        if expected_snapshot_version is not None and snapshot_version != expected_snapshot_version:
            raise DiffRenderError(
                E_SNAPSHOT_VERSION,
                "snapshot version does not match caller expectation",
                {
                    "expected": expected_snapshot_version,
                    "actual": snapshot_version,
                    "generation": generation,
                },
            )
        if Path(filename).name != filename:
            raise DiffRenderError(E_SNAPSHOT_VERSION, "snapshot file name escapes commit directory")
        snapshot_path = self.committed_dir / filename
        try:
            payload = snapshot_path.read_bytes()
        except OSError as exc:
            raise DiffRenderError(E_SNAPSHOT_VERSION, f"cannot read snapshot generation: {exc}") from exc
        actual_hash = sha256_bytes(payload)
        if actual_hash != expected_hash:
            raise DiffRenderError(
                E_SNAPSHOT_VERSION,
                "snapshot content hash differs from CURRENT pointer",
                {
                    "expected": expected_hash,
                    "actual": actual_hash,
                    "generation": generation,
                },
            )
        try:
            state = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DiffRenderError(E_SNAPSHOT_VERSION, f"invalid snapshot JSON: {exc}") from exc
        if not isinstance(state, dict):
            raise DiffRenderError(E_SNAPSHOT_VERSION, "snapshot root is not an object")
        if int(state.get("snapshot_version", -1)) != snapshot_version:
            raise DiffRenderError(
                E_SNAPSHOT_VERSION,
                "snapshot internal version differs from CURRENT pointer",
                {
                    "pointer": snapshot_version,
                    "snapshot": state.get("snapshot_version"),
                },
            )
        if bool(state.get("closed", False)) != closed:
            raise DiffRenderError(E_SNAPSHOT_VERSION, "snapshot closed flag differs from pointer")
        return state, pointer

    # ------------------------------------------------------------------
    # Commit

    def save(self, state: Mapping[str, Any], closed: bool = False) -> int:
        pointer = self.pointer()
        if pointer is None:
            generation = 1
            snapshot_version = 1
        else:
            try:
                generation = int(pointer["generation"]) + 1
                snapshot_version = int(pointer["snapshot_version"]) + 1
            except (KeyError, TypeError, ValueError) as exc:
                raise DiffRenderError(E_SNAPSHOT_VERSION, f"invalid CURRENT pointer: {exc}") from exc
        snapshot = dict(state)
        snapshot["snapshot_version"] = snapshot_version
        snapshot["closed"] = bool(closed)
        snapshot["clean_shutdown"] = bool(closed)
        if closed:
            graph = snapshot.get("graph", {})
            nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
            node_hashes = {
                str(node["key"]): str(node["content_hash"])
                for node in sorted(nodes, key=lambda item: str(item.get("key", "")))
            }
            report = {
                "closed": True,
                "forced_termination": False,
                "revision": int(snapshot.get("revision", 0)),
                "ledger_entries": len(snapshot.get("ledger", [])),
                "node_hashes": node_hashes,
                "node_hash_manifest_hash": canonical_json_hash(node_hashes),
            }
            snapshot["exit_report"] = report
            snapshot["exit_report_hash"] = canonical_json_hash(report)
        else:
            snapshot.pop("exit_report", None)
            snapshot.pop("exit_report_hash", None)
        payload = canonical_json_bytes(snapshot)
        snapshot_hash = sha256_bytes(payload)
        filename = f"snapshot-{generation:06d}.json"
        _atomic_write(self.committed_dir / filename, payload)
        closed_marker = {
            "schema": CLOSED_SCHEMA,
            "closed": bool(closed),
            "snapshot_version": snapshot_version,
            "generation": generation,
            "sha256": snapshot_hash,
        }
        if closed:
            marker_payload = canonical_json_bytes(closed_marker)
            _atomic_write(self.closed_path, marker_payload)
        elif self.closed_path.exists():
            self.closed_path.unlink()
        pointer_payload = {
            "schema": POINTER_SCHEMA,
            "generation": generation,
            "snapshot_version": snapshot_version,
            "snapshot_file": filename,
            "sha256": snapshot_hash,
            "closed": bool(closed),
        }
        _atomic_write(self.pointer_path, canonical_json_bytes(pointer_payload))
        return snapshot_version

    # ------------------------------------------------------------------
    # Recovery

    def _closed_marker_status(self, pointer: Mapping[str, Any]) -> bool:
        if not self.closed_path.exists():
            return False
        marker = self._read_json(self.closed_path)
        return (
            marker.get("schema") == CLOSED_SCHEMA
            and bool(marker.get("closed"))
            and int(marker.get("snapshot_version", -1)) == int(pointer["snapshot_version"])
            and int(marker.get("generation", -1)) == int(pointer["generation"])
            and str(marker.get("sha256")) == str(pointer["sha256"])
        )

    def write_staging(self, name: str, data: Mapping[str, Any] | bytes) -> Path:
        safe_name = Path(name).name
        payload = data if isinstance(data, bytes) else canonical_json_bytes(data)
        path = self.staging_dir / safe_name
        _atomic_write(path, payload)
        return path

    def recover(self, expected_snapshot_version: int | None = None) -> tuple[dict[str, Any] | None, RecoveryReport | None]:
        loaded = self.read_snapshot(expected_snapshot_version=expected_snapshot_version)
        if loaded is None:
            return None, None
        state, pointer = loaded
        clean = bool(pointer.get("closed")) and self._closed_marker_status(pointer)
        staging_files = sorted(path for path in self.staging_dir.iterdir() if path.is_file())
        orphans: list[dict[str, Any]] = []
        for index, path in enumerate(staging_files, start=1):
            data = path.read_bytes()
            target_name = f"orphan-{index:04d}-{path.name}"
            target = self.orphan_dir / target_name
            os.replace(path, target)
            orphans.append(
                {
                    "orphan": target_name,
                    "original": path.name,
                    "sha256": sha256_bytes(data),
                    "bytes": len(data),
                    "committed_snapshot_version": int(pointer["snapshot_version"]),
                }
            )
        if staging_files:
            ledger = {
                "schema": RECOVERY_SCHEMA,
                "recovered_snapshot_version": int(pointer["snapshot_version"]),
                "orphans": orphans,
            }
            _atomic_write(
                self.orphan_dir / "orphan_ledger.json", canonical_json_bytes(ledger)
            )
        if not clean:
            report = {
                "schema": RECOVERY_SCHEMA,
                "forced_termination_detected": True,
                "clean_shutdown": False,
                "fabricated_exit_report": False,
                "recovered_snapshot_version": int(pointer["snapshot_version"]),
                "recovered_generation": int(pointer["generation"]),
                "orphans": orphans,
                "committed_exit_report_present": "exit_report" in state,
            }
            _atomic_write(
                self.orphan_dir / "RECOVERY_REPORT.json", canonical_json_bytes(report)
            )
        return state, RecoveryReport(
            clean_shutdown=clean,
            forced_termination_detected=not clean,
            recovered_snapshot_version=int(pointer["snapshot_version"]),
            recovered_generation=int(pointer["generation"]),
            orphans=tuple(orphans),
            fabricated_exit_report=False,
        )

    def latest_committed_hash(self) -> str | None:
        pointer = self.pointer()
        return None if pointer is None else str(pointer.get("sha256"))

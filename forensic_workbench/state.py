"""Atomic dependency snapshots, version pointers, and orphan recovery."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .errors import snapshot_version
from .graph import DependencyGraph
from .util import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
)


@dataclass
class Snapshot:
    version: int
    snapshot_hash: str
    path: Path
    graph: DependencyGraph
    base_version: int | None
    base_snapshot_hash: str | None


@dataclass
class CommitResult:
    version: int
    snapshot_hash: str
    path: Path


class SnapshotStore:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.state_dir = self.workspace / "state"
        self.snapshot_dir = self.state_dir / "snapshots"
        self.pending_dir = self.state_dir / "pending"
        self.orphan_dir = self.state_dir / "orphans"
        self.recovery_dir = self.state_dir / "recovery"
        self.current_path = self.state_dir / "CURRENT.json"
        for directory in (
            self.state_dir,
            self.snapshot_dir,
            self.pending_dir,
            self.orphan_dir,
            self.recovery_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def load(self) -> Snapshot | None:
        if not self.current_path.exists():
            return None
        try:
            pointer = read_json(self.current_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise snapshot_version(
                "current snapshot pointer cannot be read",
                path=str(self.current_path),
                reason=str(exc),
            ) from exc
        if not isinstance(pointer, dict) or pointer.get("format") != "forensic-current-v1":
            raise snapshot_version(
                "current snapshot pointer has an invalid format",
                path=str(self.current_path),
                pointer=pointer,
            )
        try:
            version = int(pointer.get("version", 0))
            snapshot_hash = str(pointer.get("snapshot_hash", ""))
            relative_file = str(pointer.get("snapshot_file", ""))
        except (TypeError, ValueError) as exc:
            raise snapshot_version(
                "current snapshot pointer contains invalid version data",
                path=str(self.current_path),
                reason=str(exc),
                pointer=pointer,
            ) from exc
        if version <= 0 or not snapshot_hash or not relative_file:
            raise snapshot_version(
                "current snapshot pointer is incomplete",
                path=str(self.current_path),
                pointer=pointer,
            )
        path = (self.state_dir / relative_file).resolve()
        state_root = self.state_dir.resolve()
        if state_root != path and state_root not in path.parents:
            raise snapshot_version(
                "current snapshot pointer escapes the state directory",
                path=str(self.current_path),
                snapshot_file=relative_file,
            )
        if not path.exists():
            raise snapshot_version(
                "committed snapshot file is missing",
                version=version,
                snapshot_hash=snapshot_hash,
                path=str(path),
            )
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise snapshot_version(
                "committed snapshot cannot be read",
                version=version,
                path=str(path),
                reason=str(exc),
            ) from exc
        if not isinstance(value, dict) or value.get("format") != "forensic-snapshot-v1":
            raise snapshot_version(
                "committed snapshot has an invalid format",
                version=version,
                path=str(path),
            )
        declared_hash = str(value.get("snapshot_hash", ""))
        snapshot_without_hash = {
            key: item for key, item in value.items() if key != "snapshot_hash"
        }
        actual_hash = sha256_json(snapshot_without_hash)
        if declared_hash != snapshot_hash or actual_hash != snapshot_hash:
            raise snapshot_version(
                "snapshot version or hash validation failed",
                version=version,
                pointer_hash=snapshot_hash,
                declared_hash=declared_hash,
                actual_hash=actual_hash,
                path=str(path),
            )
        try:
            embedded_version = int(value.get("version", 0))
        except (TypeError, ValueError) as exc:
            raise snapshot_version(
                "snapshot version is not an integer",
                version=version,
                path=str(path),
                reason=str(exc),
            ) from exc
        if embedded_version != version:
            raise snapshot_version(
                "snapshot version differs from its current pointer",
                pointer_version=version,
                snapshot_version=value.get("version"),
                path=str(path),
            )
        try:
            graph = DependencyGraph.from_dict(value["graph"])
            base = value.get("base", {})
            base_version = None if base.get("version") is None else int(base["version"])
            base_hash = None if base.get("snapshot_hash") is None else str(base["snapshot_hash"])
        except (KeyError, TypeError, ValueError) as exc:
            raise snapshot_version(
                "snapshot graph or base metadata is invalid",
                version=version,
                path=str(path),
                reason=str(exc),
            ) from exc
        if base_version is not None and base_version >= version:
            raise snapshot_version(
                "snapshot base version is not older than the snapshot",
                version=version,
                base_version=base_version,
            )
        return Snapshot(
            version=version,
            snapshot_hash=snapshot_hash,
            path=path,
            graph=graph,
            base_version=base_version,
            base_snapshot_hash=base_hash,
        )

    def commit(
        self,
        graph: DependencyGraph,
        *,
        expected_base_version: int | None,
        expected_base_hash: str | None,
        crash_hook: Callable[[str], None] | None = None,
    ) -> CommitResult:
        current = self.load()
        current_version = 0 if current is None else current.version
        current_hash = None if current is None else current.snapshot_hash
        if expected_base_version != current_version or expected_base_hash != current_hash:
            raise snapshot_version(
                "snapshot changed after the incremental plan was built",
                expected_base_version=expected_base_version,
                expected_base_hash=expected_base_hash,
                current_version=current_version,
                current_hash=current_hash,
            )
        new_version = current_version + 1
        if graph.version != new_version:
            raise snapshot_version(
                "graph version does not match the next committed version",
                graph_version=graph.version,
                expected_version=new_version,
            )
        graph.seal()
        payload = {
            "format": "forensic-snapshot-v1",
            "version": new_version,
            "base": {"version": current_version or None, "snapshot_hash": current_hash},
            "graph": graph.to_dict(),
        }
        snapshot_hash = sha256_json(payload)
        payload["snapshot_hash"] = snapshot_hash
        filename = f"snapshot-{new_version:06d}-{snapshot_hash[:16]}.json"
        relative_file = f"snapshots/{filename}"
        snapshot_path = self.snapshot_dir / filename
        if snapshot_path.exists():
            existing_hash = sha256_file(snapshot_path)
            if existing_hash != sha256_bytes(canonical_bytes(payload) + b"\n"):
                raise snapshot_version(
                    "snapshot target already exists with a different payload",
                    path=str(snapshot_path),
                    version=new_version,
                    snapshot_hash=snapshot_hash,
                )
        atomic_write_json(snapshot_path, payload)
        if crash_hook is not None:
            crash_hook("after_snapshot_file")
        pointer = {
            "format": "forensic-current-v1",
            "version": new_version,
            "snapshot_hash": snapshot_hash,
            "snapshot_file": relative_file,
        }
        atomic_write_json(self.current_path, pointer)
        if crash_hook is not None:
            crash_hook("after_current_pointer")
        return CommitResult(new_version, snapshot_hash, snapshot_path)

    def write_pending(self, run_id: str, value: dict[str, Any]) -> tuple[Path, str]:
        path = self.pending_dir / f"{run_id}.json"
        digest = atomic_write_json(path, value)
        return path, digest

    def clear_pending(self, path: Path | None) -> None:
        if path is None:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def recover_orphans(self) -> dict[str, Any]:
        current = self.load() if self.current_path.exists() else None
        chain: set[Path] = set()
        chain_issues: list[dict[str, Any]] = []
        if current is not None:
            chain.add(current.path.resolve())
            cursor = self.load_snapshot_metadata(current.path)
            while cursor is not None:
                base = cursor.get("base", {})
                base_version = base.get("version")
                base_hash = base.get("snapshot_hash")
                if base_version is None or base_hash is None:
                    break
                base_path = self.find_snapshot_file(int(base_version), str(base_hash))
                if base_path is None:
                    chain_issues.append(
                        self._issue(
                            "committed_ancestor_missing",
                            "a snapshot in the committed ancestry is missing",
                            {
                                "version": int(base_version),
                                "snapshot_hash": str(base_hash),
                                "from": str(cursor.get("path")),
                            },
                        )
                    )
                    break
                if base_path.resolve() in chain:
                    break
                chain.add(base_path.resolve())
                cursor = self.load_snapshot_metadata(base_path)
        issues: list[dict[str, Any]] = []
        orphaned: list[dict[str, Any]] = []
        for path in sorted(self.snapshot_dir.glob("*.json")):
            if path.resolve() in chain:
                continue
            orphaned.append(self._quarantine(path, "unreferenced_snapshot", current))
        for path in sorted(self.pending_dir.glob("*")):
            if path.is_file():
                orphaned.append(self._quarantine(path, "uncommitted_pending_summary", current))
        for path in sorted(self.snapshot_dir.glob(".*.tmp")):
            if path.is_file():
                orphaned.append(self._quarantine(path, "incomplete_atomic_snapshot", current))
        for item in chain_issues + orphaned:
            issues.append(
                self._issue(
                    "orphan_recovered",
                    "uncommitted state was recovered and quarantined",
                    item,
                )
            )
        report = {
            "format": "forensic-recovery-report-v1",
            "current": None
            if current is None
            else {
                "version": current.version,
                "snapshot_hash": current.snapshot_hash,
                "snapshot_file": str(current.path.relative_to(self.state_dir)),
            },
            "orphaned": orphaned,
            "issues": issues,
            "recovered_cleanly": current is not None,
        }
        existing = sorted(self.recovery_dir.glob("recovery-*.json"))
        sequence = len(existing) + 1
        report_path = self.recovery_dir / f"recovery-{sequence:06d}.json"
        atomic_write_json(report_path, report)
        return {"report": report, "report_path": str(report_path)}

    def load_snapshot_metadata(self, path: Path) -> dict[str, Any] | None:
        try:
            value = read_json(path)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        value = dict(value)
        value["path"] = str(path)
        return value

    def find_snapshot_file(self, version: int, snapshot_hash: str) -> Path | None:
        pattern = f"snapshot-{version:06d}-{snapshot_hash[:16]}.json"
        path = self.snapshot_dir / pattern
        if path.exists():
            return path
        for candidate in self.snapshot_dir.glob(f"snapshot-{version:06d}-*.json"):
            metadata = self.load_snapshot_metadata(candidate)
            if metadata and metadata.get("snapshot_hash") == snapshot_hash:
                return candidate
        return None

    def _quarantine(
        self, path: Path, reason: str, current: Snapshot | None
    ) -> dict[str, Any]:
        digest = sha256_file(path)
        size = path.stat().st_size
        context = {
            "file": str(path),
            "reason": reason,
            "sha256": digest,
            "bytes": size,
            "current_version": None if current is None else current.version,
            "current_snapshot_hash": None if current is None else current.snapshot_hash,
        }
        target = self.orphan_dir / f"{path.name}.orphan"
        counter = 1
        while target.exists():
            target = self.orphan_dir / f"{path.name}.{counter}.orphan"
            counter += 1
        os.replace(path, target)
        context["quarantined_as"] = str(target)
        return context

    @staticmethod
    def _issue(kind: str, message: str, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "code": "E_SNAPSHOT_VERSION",
            "kind": kind,
            "message": message,
            "context": context,
        }

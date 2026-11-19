"""Atomic snapshot commits and committed-state-only recovery."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .canonical import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_bytes,
    read_json,
    sha256_bytes,
)
from .errors import ErrorCode, TacticalError
from .models import Route


POINTER_SCHEMA = "commit-pointer-v1"
SNAPSHOT_SCHEMA = "tactical-snapshot-v1"


class StateStore:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.state = self.workspace / "state"
        self.snapshots = self.state / "snapshots"
        self.pending = self.state / "pending"
        self.orphans = self.state / "orphans"
        self.pointer_path = self.state / "COMMITTED.json"
        self.content_hashes_path = self.state / "content_hashes.json"
        self.graph_path = self.state / "dependency_graph.json"
        self.recovery_path = self.state / "recovery_report.json"
        for directory in (self.state, self.snapshots, self.pending, self.orphans):
            directory.mkdir(parents=True, exist_ok=True)

    def committed_pointer(self) -> dict[str, Any] | None:
        if not self.pointer_path.exists():
            return None
        try:
            pointer = read_json(self.pointer_path)
        except (OSError, ValueError) as exc:
            raise TacticalError(
                ErrorCode.SNAPSHOT_VERSION,
                "committed pointer is unreadable",
                context={"path": str(self.pointer_path)},
            ) from exc
        if pointer.get("schema") != POINTER_SCHEMA:
            raise TacticalError(
                ErrorCode.SNAPSHOT_VERSION,
                "unsupported commit pointer schema",
                context={"schema": pointer.get("schema")},
            )
        return pointer

    def load_committed(self) -> dict[str, Any] | None:
        pointer = self.committed_pointer()
        if pointer is None:
            return None
        relative = str(pointer["snapshot_path"])
        snapshot_path = self.state / relative
        try:
            data = snapshot_path.read_bytes()
        except OSError as exc:
            raise TacticalError(
                ErrorCode.SNAPSHOT_VERSION,
                "committed snapshot file is missing",
                context={"snapshot_path": relative},
            ) from exc
        actual_hash = sha256_bytes(data)
        if actual_hash != pointer["sha256"]:
            raise TacticalError(
                ErrorCode.SNAPSHOT_VERSION,
                "committed snapshot hash mismatch",
                context={
                    "snapshot_path": relative,
                    "expected": pointer["sha256"],
                    "actual": actual_hash,
                },
            )
        try:
            import json

            payload = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise TacticalError(
                ErrorCode.SNAPSHOT_VERSION,
                "committed snapshot JSON is invalid",
                context={"snapshot_path": relative},
            ) from exc
        if payload.get("schema") != SNAPSHOT_SCHEMA:
            raise TacticalError(
                ErrorCode.SNAPSHOT_VERSION,
                "unsupported snapshot schema",
                context={"schema": payload.get("schema")},
            )
        return payload

    def next_snapshot_sequence(self) -> int:
        pointer = self.committed_pointer()
        if pointer is None:
            return 1
        return int(pointer["commit_seq"]) + 1

    def commit(self, engine: Any, *, decisions: dict[str, Any], label: str) -> dict[str, Any]:
        sequence = self.next_snapshot_sequence()
        snapshot_id = f"snap-{sequence:06d}"
        payload = engine.snapshot_payload(
            snapshot_id=snapshot_id,
            decisions=decisions,
        )
        payload["commit_label"] = label
        snapshot_bytes = canonical_bytes(payload)
        snapshot_hash = sha256_bytes(snapshot_bytes)
        relative = f"snapshots/{snapshot_id}/snapshot.json"
        snapshot_path = self.state / relative
        atomic_write_bytes(snapshot_path, snapshot_bytes)
        pointer = {
            "schema": POINTER_SCHEMA,
            "snapshot_id": snapshot_id,
            "snapshot_path": relative,
            "sha256": snapshot_hash,
            "commit_seq": sequence,
            "label": label,
        }
        atomic_write_json(self.pointer_path, pointer)
        self._write_views(payload, snapshot_hash)
        return {
            "snapshot_id": snapshot_id,
            "snapshot_sha256": snapshot_hash,
            "snapshot_path": str(snapshot_path),
            "commit_seq": sequence,
            "label": label,
        }

    def _write_views(self, payload: dict[str, Any], snapshot_hash: str) -> None:
        graph = payload.get("dependency_graph", {})
        atomic_write_json(self.graph_path, graph)
        routes = {
            item["query_id"]: item["route_hash"]
            for item in payload.get("path_cache", [])
        }
        atomic_write_json(
            self.content_hashes_path,
            {
                "schema": "content-hashes-v1",
                "snapshot_id": payload["snapshot_id"],
                "snapshot_sha256": snapshot_hash,
                "tier": payload["tier"],
                "map_content_hash": payload["map"]["content_hash"],
                "map_version": payload["map"]["map_version"],
                "dependency_graph_hash": graph.get("content_hash"),
                "route_hashes": dict(sorted(routes.items())),
            },
        )

    def _quarantine(self, source: Path, index: int) -> str:
        name = source.name
        destination = self.orphans / name
        if destination.exists():
            destination = self.orphans / f"{name}.duplicate-{index:04d}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        return str(destination.relative_to(self.state)).replace(os.sep, "/")

    def recover(self) -> dict[str, Any]:
        payload = self.load_committed()
        if payload is None:
            raise TacticalError(
                ErrorCode.SNAPSHOT_VERSION,
                "no committed snapshot is available for recovery",
            )
        pointer = self.committed_pointer()
        assert pointer is not None
        referenced = str(pointer["snapshot_path"]).split("/")[1]
        committed_seq = int(pointer["commit_seq"])
        orphaned: list[str] = []
        index = 0
        for directory in sorted(self.snapshots.iterdir()):
            if not directory.is_dir() or directory.name == referenced:
                continue
            if directory.name.startswith("."):
                continue
            try:
                sequence = int(directory.name.split("-")[-1])
            except ValueError:
                sequence = committed_seq + 1
            if sequence <= committed_seq:
                # Earlier sequence numbers were committed in a prior run and are
                # retained as committed history; only future/unreferenced
                # snapshots are crash leftovers.
                continue
            index += 1
            orphaned.append(self._quarantine(directory, index))
        for item in sorted(self.pending.iterdir()):
            index += 1
            orphaned.append(self._quarantine(item, index))
        for item in sorted(self.state.glob("*.tmp")):
            index += 1
            orphaned.append(self._quarantine(item, index))

        verified_routes: dict[str, str] = {}
        width = int(payload["map"]["width"])
        for entry in sorted(payload.get("path_cache", []), key=lambda item: item["query_id"]):
            route_data = entry["route"]
            route = Route(
                query_id=route_data["query_id"],
                strategy=route_data["strategy"],
                path=tuple(
                    y * width + x for x, y in route_data["path"]
                ),
                cost=int(route_data["cost"]),
                expansions=int(route_data.get("expansions", 0)),
                visited=int(route_data.get("visited", 0)),
                peak_frontier=int(route_data.get("peak_frontier", 0)),
                estimated_peak_bytes=int(route_data.get("estimated_peak_bytes", 0)),
                work_units=int(route_data.get("work_units", 0)),
                flow_key=route_data.get("flow_key"),
                jump_regions=tuple(route_data.get("jump_regions", [])),
                route_regions=tuple(route_data.get("route_regions", [])),
            )
            actual = route.route_hash(width)
            if actual != entry["route_hash"]:
                raise TacticalError(
                    ErrorCode.DEPENDENCY_STALE,
                    "committed route hash does not verify",
                    context={
                        "query_id": entry["query_id"],
                        "expected": entry["route_hash"],
                        "actual": actual,
                    },
                )
            verified_routes[entry["query_id"]] = actual
        self._write_views(payload, str(pointer["sha256"]))
        report = {
            "schema": "recovery-report-v1",
            "status": "recovered",
            "committed_snapshot_id": pointer["snapshot_id"],
            "committed_snapshot_sha256": pointer["sha256"],
            "commit_seq": pointer["commit_seq"],
            "orphans_recovered": orphaned,
            "orphan_count": len(orphaned),
            "route_hashes_verified": dict(sorted(verified_routes.items())),
            "routes_verified": len(verified_routes),
            "source": "COMMITTED.json only",
        }
        atomic_write_json(self.recovery_path, report)
        return report


def verify_recovered_hashes(
    recovery: dict[str, Any], expected_hashes: dict[str, str]
) -> dict[str, Any]:
    actual = recovery.get("route_hashes_verified", {})
    mismatched = {
        query_id: {
            "expected": expected,
            "actual": actual.get(query_id),
        }
        for query_id, expected in sorted(expected_hashes.items())
        if actual.get(query_id) != expected
    }
    return {
        "all_match": not mismatched,
        "expected_queries": len(expected_hashes),
        "verified_queries": len(actual),
        "mismatched": mismatched,
    }

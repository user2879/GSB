from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import SNAPSHOT_SCHEMA, SNAPSHOT_VERSION
from .errors import Codes
from .hashing import full_fingerprint


class SnapshotEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (bytes, bytearray)):
            return {"__bytes_b64__": base64.b64encode(bytes(obj)).decode("ascii")}
        if isinstance(obj, tuple):
            return list(obj)
        return super().default(obj)


def snapshot_object_hook(value: dict[str, Any]) -> Any:
    if "__bytes_b64__" in value and len(value) == 1:
        return base64.b64decode(value["__bytes_b64__"])
    return value


def json_loads_snapshot(text: str) -> dict[str, Any]:
    return json.loads(text, object_hook=snapshot_object_hook)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions = self.root / "versions"
        self.current_link = self.root / "current"
        self.versions.mkdir(parents=True, exist_ok=True)
        self.global_version = 0
        current = self.load_current()
        if current:
            self.global_version = int(current.get("global_version", 0))

    def load_current(self) -> dict[str, Any] | None:
        manifest = self.current_link / "snapshot.json"
        if not manifest.exists():
            return None
        try:
            return json_loads_snapshot(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def commit(self, engine: Any) -> dict[str, Any]:
        previous = self.load_current() or {}
        self.global_version = max(self.global_version, int(previous.get("global_version", 0))) + 1
        config_hash = full_fingerprint(engine.config)
        state_fingerprint = full_fingerprint(engine.state)
        completed = list(getattr(engine, "completed_frames", []))
        from .engine import tile_bounds, tile_order

        rows = int(engine.config["tile_rows"])
        height = int(engine.config["height"])
        bounds = tile_bounds(height, rows)
        schedule_order = tile_order(rows, str(engine.config.get("tile_scheduling", "top_down")))
        tile_indices: dict[int, set[int]] = {}
        for record in engine.cache.values():
            if record.get("kind") != "tile_decode":
                continue
            frame = record.get("frame")
            index = record.get("index")
            data = record.get("data", {})
            if frame is None or index is None or not (0 <= int(index) < rows):
                continue
            y0, y1 = bounds[int(index)]
            if data.get("y0") == y0 and data.get("y1") == y1:
                tile_indices.setdefault(int(frame), set()).add(int(index))

        tile_cursors: dict[str, Any] = {}
        for frame in range(int(engine.config["frames"])):
            indices = sorted(tile_indices.get(frame, set()))
            index_set = set(indices)
            next_schedule_position = next(
                (position for position, tile_index in enumerate(schedule_order)
                 if tile_index not in index_set),
                len(schedule_order),
            )
            next_row = next((row for row in range(rows) if row not in index_set), rows)
            tile_cursors[str(frame)] = {
                "completed_indices": indices,
                "completed_count": len(indices),
                "next_row_cursor": next_row,
                "schedule_order": schedule_order,
                "next_schedule_position": next_schedule_position,
            }
        pending_queue: list[dict[str, Any]] = []
        total_frames = int(engine.config["frames"])
        tile_rows = int(engine.config["tile_rows"])
        for frame in completed:
            tile_cursors[str(frame)] = {
                "completed": list(range(tile_rows)),
                "next_cursor": tile_rows,
            }
        for frame in range(total_frames):
            if frame not in completed:
                pending_queue.append({
                    "frame": frame,
                    "stages": ["text", "image", "motion", "fusion", "initial",
                               "denoise_steps", "tiles", "compose"],
                })

        artifact_versions = [
            {
                "kind": record.get("kind"),
                "frame": record.get("frame"),
                "index": record.get("index"),
                "key": key,
                "fingerprint": record.get("fingerprint"),
                "version": record.get("fingerprint", "")[:16],
            }
            for key, record in sorted(engine.cache.items())
        ]
        stage_versions: dict[str, list[str]] = {}
        for item in artifact_versions:
            stage_versions.setdefault(str(item["kind"]), []).append(item["key"])

        payload = {
            "snapshot_schema": SNAPSHOT_SCHEMA,
            "snapshot_version": SNAPSHOT_VERSION,
            "global_version": self.global_version,
            "config_hash": config_hash,
            "state_fingerprint": state_fingerprint,
            "state": engine.state,
            "condition_fingerprints": (engine.last_summary or {}).get(
                "condition_fingerprints", {}
            ),
            "condition_fingerprint": (engine.last_summary or {}).get("condition_fingerprint"),
            "cache": engine.cache,
            "ledger": engine.ledger,
            "consumed_points": engine.consumed_points,
            "last_summary": engine.last_summary,
            "completed_frames": completed,
            "tile_cursors": tile_cursors,
            "pending_queue": pending_queue,
            "artifact_versions": artifact_versions,
            "stage_versions": stage_versions,
            "command_log": getattr(engine, "command_log", []),
            "failures": getattr(engine, "snapshot_failures", []),
        }

        staging = Path(tempfile.mkdtemp(prefix=".tmp-", dir=str(self.root)))
        try:
            manifest_path = staging / "snapshot.json"
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=False,
                          cls=SnapshotEncoder)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(staging)

            target = self.versions / (
                f"v{self.global_version:06d}-{config_hash[:12]}-{state_fingerprint[:12]}"
            )
            if target.exists():
                shutil.rmtree(target)
            os.rename(staging, target)
            _fsync_directory(self.versions)

            link_name = f".current-tmp-{os.getpid()}-{self.global_version}"
            tmp_link = self.root / link_name
            relative_target = os.path.relpath(target, self.root)
            os.symlink(relative_target, tmp_link)
            os.replace(tmp_link, self.current_link)
            _fsync_directory(self.root)

            old_versions = sorted(
                path for path in self.versions.glob("v*") if path.is_dir()
            )
            for old_path in old_versions[:-2]:
                shutil.rmtree(old_path, ignore_errors=True)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        return payload


def decide_restore(payload: dict[str, Any] | None, state: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"decision": "blank_full_render", "code": Codes.OK}
    if payload.get("snapshot_schema") != SNAPSHOT_SCHEMA or \
            payload.get("snapshot_version") != SNAPSHOT_VERSION:
        return {
            "decision": "full_rebuild_snapshot_version",
            "code": Codes.SNAPSHOT_SCHEMA,
            "expected": SNAPSHOT_VERSION,
            "actual": payload.get("snapshot_version"),
        }
    config_hash = full_fingerprint(state["config"])
    state_fp = full_fingerprint(state)
    if payload.get("config_hash") != config_hash:
        return {"decision": "incremental_config_changed", "code": Codes.OK,
                "config_hash": config_hash, "committed_config_hash": payload.get("config_hash")}
    if payload.get("state_fingerprint") != state_fp:
        return {"decision": "incremental_state_changed", "code": Codes.OK}
    return {"decision": "resume_pending", "code": Codes.OK,
            "global_version": payload.get("global_version")}


def restore_payload_into_engine(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    decision = decide_restore(payload, engine.state)
    if decision["decision"].startswith("full_rebuild"):
        return decision
    engine.cache = payload.get("cache", {})
    engine.ledger = payload.get("ledger", [])
    engine.consumed_points = int(payload.get("consumed_points", 0))
    engine.last_summary = payload.get("last_summary")
    engine.completed_frames = list(payload.get("completed_frames", []))
    engine.command_log = payload.get("command_log", [])
    engine.snapshot_failures = payload.get("failures", [])
    return decision

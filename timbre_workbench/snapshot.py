"""Atomic staged snapshots with version and configuration validation."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .errors import E_SNAPSHOT_STALE, E_SNAPSHOT_WRITE, WorkbenchError
from .util import atomic_write_json, canonical_json, safe_join, sha256_bytes, sha256_obj
from .wavio import AudioData, read_wav, write_wav


class SnapshotManager:
    def __init__(
        self,
        work_dir: Path,
        *,
        base_config_hash: str,
        labels_hash: str,
        engine_schema_version: int,
    ) -> None:
        self.work_dir = work_dir
        self.snapshot_root = work_dir / "snapshots"
        self.temp_root = work_dir / "tmp"
        self.pointer_path = work_dir / "CURRENT.json"
        self.base_config_hash = base_config_hash
        self.labels_hash = labels_hash
        self.engine_schema_version = engine_schema_version
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def _stale(self, reason: str, **context: Any) -> dict[str, Any]:
        error = WorkbenchError(E_SNAPSHOT_STALE, reason, context or None)
        return error.as_failure(stage="snapshot_load", sort_key=f"__snapshot__|{E_SNAPSHOT_STALE}|{reason}")

    def load(
        self,
        *,
        expected_global_version: int | None = None,
        commands: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not self.pointer_path.exists():
            return None, None
        try:
            pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
            snapshot_dir = self.snapshot_root / str(pointer["snapshot"])
            snapshot = json.loads((snapshot_dir / "snapshot.json").read_text(encoding="utf-8"))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return None, self._stale("snapshot pointer or manifest is unreadable", os_error=str(exc))

        if int(snapshot.get("engine_schema_version", -1)) != self.engine_schema_version:
            return None, self._stale(
                "snapshot engine schema version mismatch",
                expected=self.engine_schema_version,
                actual=snapshot.get("engine_schema_version"),
            )
        if snapshot.get("base_config_hash") != self.base_config_hash:
            return None, self._stale(
                "snapshot configuration hash mismatch",
                expected=self.base_config_hash,
                actual=snapshot.get("base_config_hash"),
            )
        if snapshot.get("labels_hash") != self.labels_hash:
            return None, self._stale(
                "snapshot labels hash mismatch",
                expected=self.labels_hash,
                actual=snapshot.get("labels_hash"),
            )
        if expected_global_version is not None and int(snapshot.get("global_version", -1)) != int(
            expected_global_version
        ):
            return None, self._stale(
                "snapshot global version mismatch",
                expected=int(expected_global_version),
                actual=snapshot.get("global_version"),
            )
        if commands is not None:
            expected_prefix = command_prefix_hash(commands, int(snapshot.get("command_cursor", 0)))
            if snapshot.get("command_prefix_hash") != expected_prefix:
                return None, self._stale(
                    "snapshot command prefix hash mismatch",
                    expected=expected_prefix,
                    actual=snapshot.get("command_prefix_hash"),
                )

        media_root = snapshot_dir / "media"
        assets: dict[str, AudioData] = {}
        try:
            for entry in snapshot.get("media", []):
                logical_name = str(entry["path"])
                path = safe_join(media_root, logical_name)
                asset = read_wav(path, logical_name=logical_name)
                if asset.sha256 != entry["sha256"]:
                    return None, self._stale(
                        "snapshot media hash mismatch",
                        file=logical_name,
                        expected=entry["sha256"],
                        actual=asset.sha256,
                    )
                assets[logical_name] = asset
        except (OSError, KeyError, ValueError) as exc:
            return None, self._stale("snapshot media cannot be restored", os_error=str(exc))
        snapshot["assets"] = assets
        snapshot["_snapshot_dir"] = str(snapshot_dir)
        return snapshot, None

    def commit(
        self,
        *,
        assets: dict[str, AudioData],
        state: dict[str, Any],
        keep: int = 5,
    ) -> dict[str, Any]:
        stage = self.temp_root / f"snapshot-{uuid.uuid4().hex}"
        media_dir = stage / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        media_manifest: list[dict[str, Any]] = []
        try:
            for logical_name, asset in sorted(assets.items(), key=lambda item: (item[0].casefold(), item[0])):
                target = safe_join(media_dir, logical_name)
                write_wav(target, asset.samples, sample_rate=asset.sample_rate)
                media_manifest.append(
                    {
                        "path": logical_name,
                        "sha256": asset.sha256,
                        "sample_count": asset.sample_count,
                        "sample_rate": asset.sample_rate,
                    }
                )
            snapshot_payload = {
                **state,
                "engine_schema_version": self.engine_schema_version,
                "base_config_hash": self.base_config_hash,
                "labels_hash": self.labels_hash,
                "media": media_manifest,
            }
            atomic_write_json(stage / "snapshot.json", snapshot_payload)
            snapshot_name = (
                f"snap-v{int(state['global_version']):08d}-c{int(state['command_cursor']):04d}-"
                f"{str(state['effective_config_hash'])[:12]}-{uuid.uuid4().hex[:8]}"
            )
            final_dir = self.snapshot_root / snapshot_name
            os.replace(stage, final_dir)
        except Exception as exc:
            shutil.rmtree(stage, ignore_errors=True)
            raise WorkbenchError(E_SNAPSHOT_WRITE, "unable to commit staged snapshot", {"os_error": str(exc)}) from exc

        pointer = {
            "snapshot": snapshot_name,
            "global_version": int(state["global_version"]),
            "command_cursor": int(state["command_cursor"]),
            "effective_config_hash": state["effective_config_hash"],
            "manifest_hash": sha256_bytes((final_dir / "snapshot.json").read_bytes()),
        }
        try:
            atomic_write_json(self.pointer_path, pointer)
        except Exception as exc:
            raise WorkbenchError(
                E_SNAPSHOT_WRITE,
                "unable to atomically replace CURRENT snapshot pointer",
                {"os_error": str(exc), "snapshot_id": snapshot_name},
            ) from exc
        self._cleanup(keep=max(1, int(keep)))
        return {
            "snapshot_id": snapshot_name,
            "path": str(final_dir),
            "global_version": pointer["global_version"],
            "command_cursor": pointer["command_cursor"],
            "manifest_hash": pointer["manifest_hash"],
            "atomic_commit": True,
        }

    def _cleanup(self, keep: int) -> None:
        directories = sorted(
            [path for path in self.snapshot_root.iterdir() if path.is_dir()],
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        for path in directories[keep:]:
            shutil.rmtree(path, ignore_errors=True)


def command_prefix_hash(commands: list[dict[str, Any]], cursor: int) -> str:
    return sha256_obj({"commands": commands[: max(0, cursor)]})


def snapshot_state_payload(
    *,
    global_version: int,
    command_cursor: int,
    command_prefix: str,
    effective_config: dict[str, Any],
    effective_config_hash: str,
    tier_overrides: dict[str, int | None],
    segments_by_file: dict[str, list[dict[str, Any]]],
    caches: dict[str, Any],
    failures: list[dict[str, Any]],
    last_result: dict[str, Any] | None,
    labels: dict[str, Any],
    product_versions: dict[str, str],
    quantization: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "snapshot_schema": 1,
        "global_version": int(global_version),
        "command_cursor": int(command_cursor),
        "command_prefix_hash": command_prefix,
        "effective_config": effective_config,
        "effective_config_hash": effective_config_hash,
        "tier_overrides": tier_overrides,
        "segments_by_file": segments_by_file,
        "caches": caches,
        "failures": failures,
        "last_result": last_result,
        "labels": labels,
        "product_versions": product_versions,
        "quantization": quantization or {},
    }

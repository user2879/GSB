"""Immutable snapshot directories with an atomically replaced CURRENT pointer."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .engine import IncrementalEngine
from .errors import E_STATE_INVALID, E_RESUME_MISMATCH, StateError
from .imageio import save_fixed_png
from .model import ConfigBundle
from .util import (
    atomic_write_json,
    read_json,
    sha256_bytes,
    sha256_json,
)


SNAPSHOT_SCHEMA = "ipw-snapshot-v1"


class SnapshotManager:
    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir).resolve()
        self.snapshots_dir = self.state_dir / "snapshots"
        self.current_path = self.state_dir / "CURRENT"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def current_info(self) -> dict[str, Any] | None:
        if not self.current_path.is_file():
            return None
        try:
            return read_json(self.current_path)
        except Exception as exc:
            raise StateError(E_STATE_INVALID, f"cannot parse CURRENT pointer: {exc}") from exc

    def _resolve_committed(self, info: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        relative = info.get("snapshot_dir")
        if not isinstance(relative, str):
            raise StateError(E_STATE_INVALID, "CURRENT pointer is missing snapshot_dir")
        snapshot_dir = (self.state_dir / relative).resolve()
        if snapshot_dir.parent != self.snapshots_dir.resolve():
            raise StateError(E_STATE_INVALID, "CURRENT snapshot path escapes snapshot directory")
        manifest_path = snapshot_dir / "manifest.json"
        complete_path = snapshot_dir / "COMPLETE"
        if not manifest_path.is_file() or not complete_path.is_file():
            raise StateError(E_STATE_INVALID, "committed snapshot is missing manifest or COMPLETE marker")
        manifest_bytes = manifest_path.read_bytes()
        expected_manifest_hash = info.get("manifest_sha256")
        if expected_manifest_hash != sha256_bytes(manifest_bytes):
            raise StateError(E_STATE_INVALID, "snapshot manifest hash does not match CURRENT")
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            complete = read_json(complete_path)
            state = read_json(snapshot_dir / "state.json")
        except Exception as exc:
            raise StateError(E_STATE_INVALID, f"cannot read committed snapshot: {exc}") from exc
        if complete.get("manifest_sha256") != expected_manifest_hash:
            raise StateError(E_STATE_INVALID, "COMPLETE marker does not match manifest")
        if manifest.get("global_version") != info.get("version"):
            raise StateError(E_STATE_INVALID, "snapshot version mismatch")
        if manifest.get("state_hash") != state.get("state_hash"):
            raise StateError(E_STATE_INVALID, "manifest/state hash mismatch")
        return snapshot_dir, manifest, state

    def recover_info(self) -> dict[str, Any]:
        info = self.current_info()
        if info is None:
            raise StateError(E_STATE_INVALID, "no committed snapshot pointer exists")
        _, manifest, state = self._resolve_committed(info)
        return {
            "recovered_version": int(manifest["global_version"]),
            "snapshot_dir": info["snapshot_dir"],
            "command_cursor": int(manifest.get("command_cursor", -1)),
            "completed_commands": int(manifest.get("completed_commands", 0)),
            "pending_tiles": manifest.get("pending_tiles", []),
            "composite_fingerprint": manifest.get("composite_fingerprint"),
            "state_hash": manifest.get("state_hash"),
            "config_hash": manifest.get("effective_config_hash"),
        }

    def load(
        self,
        bundle: ConfigBundle,
        commands_hash: str | None = None,
    ) -> tuple[IncrementalEngine, dict[str, Any]]:
        info = self.current_info()
        if info is None:
            raise StateError(E_STATE_INVALID, "no committed snapshot pointer exists")
        snapshot_dir, manifest, state = self._resolve_committed(info)
        if manifest.get("base_config_hash") != bundle.base_config_hash:
            raise StateError(
                E_RESUME_MISMATCH,
                "committed snapshot was created from a different base config or image set",
                {
                    "snapshot_base_config_hash": manifest.get("base_config_hash"),
                    "current_base_config_hash": bundle.base_config_hash,
                },
            )
        if commands_hash is not None and state.get("commands_hash") not in (None, commands_hash):
            raise StateError(
                E_RESUME_MISMATCH,
                "committed snapshot was created from a different command stream",
                {
                    "snapshot_commands_hash": state.get("commands_hash"),
                    "current_commands_hash": commands_hash,
                },
            )

        def tile_loader(level: int, row: int, col: int) -> np.ndarray:
            path = snapshot_dir / "tiles" / f"l{level:02d}_r{row:04d}_c{col:04d}.npy"
            if not path.is_file():
                raise StateError(E_STATE_INVALID, f"snapshot tile file is missing: {path.name}")
            return np.load(path, allow_pickle=False)

        engine = IncrementalEngine.restore(
            bundle,
            state,
            tile_loader=tile_loader,
            resumed_version=int(manifest["global_version"]),
        )
        engine.commands_hash = commands_hash
        return engine, manifest

    def commit(
        self,
        engine: IncrementalEngine,
        report: dict[str, Any],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        info = self.current_info()
        current_version = 0 if info is None else int(info.get("version", 0))
        state_payload = engine.state_payload()
        state_hash = sha256_json(state_payload)
        state_payload["state_hash"] = state_hash
        if info is not None:
            _, current_manifest, _ = self._resolve_committed(info)
            if current_manifest.get("state_hash") == state_hash and not force:
                return {
                    "committed": False,
                    "reused_current": True,
                    "version": current_version,
                    "snapshot_dir": info["snapshot_dir"],
                    "state_hash": state_hash,
                }

        version = current_version + 1
        snapshot_name = f"v{version:08d}_{state_hash[:12]}"
        target = self.snapshots_dir / snapshot_name
        while target.exists():
            version += 1
            snapshot_name = f"v{version:08d}_{state_hash[:12]}"
            target = self.snapshots_dir / snapshot_name

        temporary = Path(tempfile.mkdtemp(prefix=".tmp-snapshot-", dir=self.snapshots_dir))
        try:
            (temporary / "tiles").mkdir(parents=True, exist_ok=True)
            (temporary / "layers").mkdir(parents=True, exist_ok=True)
            atomic_write_json(temporary / "state.json", state_payload)

            for key in sorted(engine.valid_tiles):
                level, row, col = key
                x0, y0, x1, y1 = engine.spec.tile_bounds(*key)
                tile = np.ascontiguousarray(engine.layer_outputs[level][y0:y1, x0:x1], dtype=np.int64)
                np.save(
                    temporary / "tiles" / f"l{level:02d}_r{row:04d}_c{col:04d}.npy",
                    tile,
                    allow_pickle=False,
                )
            for level in range(engine.spec.levels):
                save_fixed_png(
                    temporary / "layers" / f"layer_{level:02d}.png",
                    engine.layer_outputs[level],
                    engine.spec.fraction_bits,
                )
            if engine.composite is not None:
                save_fixed_png(
                    temporary / "composite.png",
                    engine.composite.image,
                    engine.spec.fraction_bits,
                )
                np.savez_compressed(
                    temporary / "contributions.npz",
                    **{
                        f"level_{level:02d}": array
                        for level, array in enumerate(engine.composite.contribution_arrays)
                    },
                )
            retention_arrays: dict[str, np.ndarray] = {}
            for level in range(engine.spec.levels):
                mask = engine.masks[level]
                retained_source = mask & (
                    engine.mutable.mask_fill == "preserve_source"
                ) & (engine.mutable.style_inside == "repaired")
                style_mapped_inside = mask & (engine.mutable.style_inside == "mapped")
                style_mapped_outside = (~mask) & (engine.mutable.style_outside == "mapped")
                retention_arrays[f"mask_{level:02d}"] = mask
                retention_arrays[f"source_preserved_{level:02d}"] = retained_source
                retention_arrays[f"style_mapped_inside_{level:02d}"] = style_mapped_inside
                retention_arrays[f"style_mapped_outside_{level:02d}"] = style_mapped_outside
            np.savez_compressed(temporary / "retention_masks.npz", **retention_arrays)

            layer_info = []
            for summary in engine.layer_summary():
                layer_info.append(
                    {
                        "level": summary["level"],
                        "size": summary["size"],
                        "fingerprint": summary["fingerprint"],
                        "tile_rows": summary["tile_rows"],
                        "tile_cols": summary["tile_cols"],
                        "retention": summary["retention"],
                        "tiles": summary["tiles"],
                    }
                )
            manifest = {
                "schema": SNAPSHOT_SCHEMA,
                "global_version": version,
                "state_hash": state_hash,
                "base_config_hash": engine.bundle.base_config_hash,
                "effective_config_hash": engine.bundle.effective_config_hash(engine.mutable),
                "source_fingerprint": engine.bundle.source_fingerprint,
                "style_fingerprint": engine.bundle.style_fingerprint,
                "commands_hash": engine.commands_hash,
                "command_cursor": engine.cursor,
                "completed_commands": engine.cursor + 1,
                "pending_tiles": [list(key) for key in sorted(engine.dirty_tiles)],
                "tile_versions": {
                    f"{key[0]}/{key[1]}/{key[2]}": value
                    for key, value in sorted(engine.tile_versions.items())
                },
                "layer_info": layer_info,
                "composite_fingerprint": None
                if engine.composite is None
                else engine.composite.fingerprint,
                "composite_stale": engine.composite_stale,
                "concurrency_peak": engine.concurrency_peak,
                "thread_limit": engine.bundle.threads,
                "tile_budget": engine.bundle.recompute_tile_budget,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "interpretation": {
                    "downsample": "block_mean_half_up",
                    "upsample": "bilinear_clamped_edge_half_up_ties",
                    "tile_order": "level,row,col row-major",
                    "retention_artifact": "retention_masks.npz",
                },
            }
            manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
            (temporary / "manifest.json").write_bytes(manifest_bytes)
            manifest_hash = sha256_bytes(manifest_bytes)

            snapshot_report = dict(report)
            snapshot_report["snapshot"] = {
                "global_version": version,
                "snapshot_dir": f"snapshots/{snapshot_name}",
                "state_hash": state_hash,
                "manifest_sha256": manifest_hash,
                "pending_tiles": [list(key) for key in sorted(engine.dirty_tiles)],
            }
            report_bytes = (json.dumps(snapshot_report, indent=2, sort_keys=True) + "\n").encode("utf-8")
            (temporary / "report.json").write_bytes(report_bytes)
            atomic_write_json(
                temporary / "COMPLETE",
                {
                    "manifest_sha256": manifest_hash,
                    "state_hash": state_hash,
                    "report_sha256": sha256_bytes(report_bytes),
                },
            )
            for path in temporary.rglob("*"):
                if path.is_file():
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
            os.replace(temporary, target)
            atomic_write_json(
                self.current_path,
                {
                    "schema": SNAPSHOT_SCHEMA,
                    "version": version,
                    "snapshot_dir": f"snapshots/{snapshot_name}",
                    "manifest_sha256": manifest_hash,
                },
            )
            return {
                "committed": True,
                "reused_current": False,
                "version": version,
                "snapshot_dir": f"snapshots/{snapshot_name}",
                "state_hash": state_hash,
                "manifest_sha256": manifest_hash,
                "report": snapshot_report,
            }
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .configuration import WorkbenchConfig
from .errors import WorkbenchError


class SnapshotStore:
    def __init__(self, config: WorkbenchConfig, root: str | Path | None = None) -> None:
        self.config = config
        self.root = Path(root).resolve() if root else config.snapshot_dir
        self.tmp_dir = self.root / "tmp"
        self.snapshots_dir = self.root / "snapshots"
        self.manifest_path = self.root / "commit_manifest.json"

    def reset_for_demo(self, remove_all: bool = True) -> None:
        if remove_all and self.root.exists():
            shutil.rmtree(self.root)
            return
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        if self.manifest_path.exists():
            self.manifest_path.unlink()

    def prepare_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "committed snapshot manifest is unreadable; refusing partial overwrite",
                {"manifest_path": str(self.manifest_path), "reason": str(exc)},
            ) from exc
        if not isinstance(manifest, dict) or manifest.get("manifest_version") != 1:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "committed snapshot manifest has an unsupported version",
                {"manifest_path": str(self.manifest_path)},
            )
        return manifest

    def check_compatible(self, manifest: dict[str, Any] | None) -> None:
        if manifest is None:
            return
        reasons = []
        if manifest.get("global_version") != self.config.global_version:
            reasons.append({
                "field": "global_version",
                "committed": manifest.get("global_version"),
                "current": self.config.global_version,
            })
        if manifest.get("config_hash") != self.config.config_hash:
            reasons.append({
                "field": "config_hash",
                "committed": manifest.get("config_hash"),
                "current": self.config.config_hash,
            })
        snapshot_dir = manifest.get("snapshot_dir")
        if not snapshot_dir or not (self.root / str(snapshot_dir)).exists():
            reasons.append({"field": "snapshot_dir", "committed": snapshot_dir, "current": None})
        if reasons:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "snapshot version or configuration hash changed; use whole-snapshot recovery",
                {"reasons": reasons, "manifest_path": str(self.manifest_path)},
            )

    def load_products(self, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
        snapshot_dir = self.root / str(manifest["snapshot_dir"])
        products_path = snapshot_dir / "products.npz"
        if not products_path.exists():
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "committed product archive is missing; refusing partial overwrite",
                {"products_path": str(products_path)},
            )
        try:
            with np.load(products_path, allow_pickle=False) as archive:
                return {key: archive[key].copy() for key in archive.files}
        except Exception as exc:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                f"committed product archive cannot be read: {exc}",
                {"products_path": str(products_path)},
            ) from exc

    def commit(
        self,
        manifest: dict[str, Any],
        arrays: dict[str, np.ndarray],
        images: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        self.prepare_dirs()
        commit_seq = int(manifest.get("commit_seq", 1))
        applied_cursor = int(manifest.get("applied_command_cursor", manifest.get("command_cursor", 0)))
        state_fp = str(manifest.get("runtime_state_fingerprint", "state"))[:12]
        snapshot_name = f"snapshot-{commit_seq:06d}-cmd{applied_cursor:04d}-{state_fp}"
        staging = self.tmp_dir / "staging"
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "tiles").mkdir(parents=True)

        np.savez_compressed(staging / "products.npz", **arrays)

        tile_manifest = manifest.get("tiles", {})
        for tile_id, tile_data in tile_manifest.items():
            if tile_data.get("status") not in {"recomputed", "reused", "clean", "done"}:
                continue
            level = int(tile_data["level"])
            row = int(tile_data["row"])
            col = int(tile_data["col"])
            size = self.config.tile_sizes[level]
            y0, x0 = row * size, col * size
            y1, x1 = y0 + size, x0 + size
            repaired = arrays[f"repaired_{level}"][y0:y1, x0:x1]
            styled = arrays[f"styled_{level}"][y0:y1, x0:x1]
            repaired.tofile(staging / "tiles" / f"{tile_id}.repair.bin")
            styled.tofile(staging / "tiles" / f"{tile_id}.style.bin")

        from .image_io import write_png

        for image_name, pixels in (images or {}).items():
            write_png(staging / image_name, pixels)

        staged_manifest = dict(manifest)
        staged_manifest["snapshot_name"] = snapshot_name
        with (staging / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(staged_manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        stages_path = staging / "stages.json"
        stages_path.write_text(
            json.dumps(staged_manifest.get("stages", {}), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        target = self.snapshots_dir / snapshot_name
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)

        committed_manifest = dict(staged_manifest)
        committed_manifest["snapshot_dir"] = str(Path("snapshots") / snapshot_name)
        temporary_pointer = self.tmp_dir / "commit_manifest.json.tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        with temporary_pointer.open("w", encoding="utf-8") as handle:
            json.dump(committed_manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_pointer, self.manifest_path)
        return committed_manifest

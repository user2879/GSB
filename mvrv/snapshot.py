from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .hashing import file_sha256, sha256_object


STAGE_FILES = {
    "validation": "validation.json",
    "features_matches": "features_matches.json",
    "triangulation": "triangulation.json",
    "depth": "depth.json",
    "voxels": "voxels.json",
}


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snapshot_root = self.root / "runtime" / "snapshots"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.current_pointer = self.snapshot_root / "current.json"

    def _next_version(self) -> int:
        versions = []
        for path in self.snapshot_root.glob("version-*"):
            try:
                versions.append(int(path.name.split("-", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max(versions, default=0) + 1

    def _write_json(self, directory: Path, name: str, value: Any) -> str:
        path = directory / name
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        return name

    def commit(
        self,
        *,
        validation_report: dict[str, Any],
        state: dict[str, Any],
        ledger: dict[str, Any],
        global_version: int | None = None,
    ) -> dict[str, Any]:
        version = int(global_version if global_version is not None else self._next_version())
        state = dict(state)
        state["global_version"] = version
        staging = Path(tempfile.mkdtemp(prefix=f"staging-v{version:06d}-", dir=self.snapshot_root))
        try:
            self._write_json(staging, STAGE_FILES["validation"], validation_report)
            self._write_json(
                staging,
                STAGE_FILES["features_matches"],
                {"views": state["views"], "pairs": state["pairs"], "strategy": state["strategy"]},
            )
            self._write_json(staging, STAGE_FILES["triangulation"], {"points": state["points"]})
            self._write_json(staging, STAGE_FILES["depth"], {"depth_maps": state["depth_maps"]})
            self._write_json(staging, STAGE_FILES["voxels"], {"voxels": state["voxels"]})
            self._write_json(staging, "ledger.json", ledger)
            self._write_json(staging, "todos.json", {"todo_queue": state.get("todo_queue", [])})
            self._write_json(staging, "metrics.json", state.get("metrics", {}))

            stage_versions = {
                "validation": sha256_object(validation_report),
                "features_matches": sha256_object(
                    {
                        "views": [v["cache_key"] for v in state["views"]],
                        "pairs": [p["cache_key"] for p in state["pairs"]],
                    }
                ),
                "triangulation": sha256_object([p["cache_key"] for p in state["points"]]),
                "depth": sha256_object(
                    [pixel["cache_key"] for dm in state["depth_maps"] for pixel in dm["pixels"]]
                ),
                "voxels": sha256_object([v["cache_key"] for v in state["voxels"]]),
            }
            files = {}
            for path in sorted(staging.iterdir()):
                if path.is_file() and path.name != "manifest.json":
                    files[path.name] = file_sha256(str(path))
            manifest = {
                "global_version": version,
                "config_hash": state["config_fingerprint"],
                "camera_hash": state["camera_hash"],
                "stage_versions": stage_versions,
                "files": files,
                "todo_count": len(state.get("todo_queue", [])),
                "schema_version": "mvrv-snapshot-1",
            }
            self._write_json(staging, "manifest.json", manifest)
            final_dir = self.snapshot_root / f"version-{version:06d}"
            if final_dir.exists():
                raise FileExistsError(final_dir)
            os.replace(staging, final_dir)
            pointer = {"global_version": version, "snapshot": final_dir.name}
            pointer_tmp = self.current_pointer.with_suffix(".json.tmp")
            pointer_tmp.write_text(json.dumps(pointer, sort_keys=True), encoding="utf-8")
            os.replace(pointer_tmp, self.current_pointer)
            return {"status": "committed", "snapshot": str(final_dir), "manifest": manifest}
        finally:
            if staging.exists():
                import shutil

                shutil.rmtree(staging, ignore_errors=True)

    def verify_directory(self, directory: Path) -> tuple[bool, dict[str, Any] | None, str]:
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return False, None, "missing_manifest"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False, None, "corrupt_manifest"
        for name, expected in manifest.get("files", {}).items():
            path = directory / name
            if not path.exists():
                return False, manifest, f"missing_file:{name}"
            if file_sha256(str(path)) != expected:
                return False, manifest, f"hash_mismatch:{name}"
        return True, manifest, "valid"

    def recover(self) -> dict[str, Any]:
        if self.current_pointer.exists():
            try:
                pointer = json.loads(self.current_pointer.read_text(encoding="utf-8"))
                directory = self.snapshot_root / pointer["snapshot"]
                valid, manifest, reason = self.verify_directory(directory)
                if valid:
                    return {"status": "current_valid", "global_version": manifest["global_version"], "reason": reason}
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        candidates = []
        for directory in self.snapshot_root.glob("version-*"):
            valid, manifest, reason = self.verify_directory(directory)
            if valid and manifest:
                candidates.append((int(manifest["global_version"]), directory, manifest))
        if not candidates:
            return {"status": "no_valid_snapshot", "global_version": 0}
        version, directory, manifest = max(candidates, key=lambda item: item[0])
        pointer = {"global_version": version, "snapshot": directory.name}
        pointer_tmp = self.current_pointer.with_suffix(".json.recover-tmp")
        pointer_tmp.write_text(json.dumps(pointer, sort_keys=True), encoding="utf-8")
        os.replace(pointer_tmp, self.current_pointer)
        return {"status": "restored", "global_version": version, "snapshot": str(directory), "reason": "latest_valid_manifest"}

    def load_latest_state(self) -> dict[str, Any] | None:
        recovery = self.recover()
        if recovery["status"] == "no_valid_snapshot":
            return None
        directory = self.snapshot_root / f"version-{int(recovery['global_version']):06d}"
        state: dict[str, Any] = {}
        features = json.loads((directory / STAGE_FILES["features_matches"]).read_text(encoding="utf-8"))
        tri = json.loads((directory / STAGE_FILES["triangulation"]).read_text(encoding="utf-8"))
        depth = json.loads((directory / STAGE_FILES["depth"]).read_text(encoding="utf-8"))
        voxels = json.loads((directory / STAGE_FILES["voxels"]).read_text(encoding="utf-8"))
        state.update(features)
        state.update(tri)
        state.update(depth)
        state.update(voxels)
        metrics_path = directory / "metrics.json"
        if metrics_path.exists():
            state["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        state["config_fingerprint"] = manifest["config_hash"]
        state["camera_hash"] = manifest["camera_hash"]
        state["global_version"] = manifest["global_version"]
        return state

    def load_latest_ledger(self) -> dict[str, Any] | None:
        recovery = self.recover()
        if recovery["status"] == "no_valid_snapshot":
            return None
        ledger_path = (
            self.snapshot_root
            / f"version-{int(recovery['global_version']):06d}"
            / "ledger.json"
        )
        if not ledger_path.exists():
            return None
        return json.loads(ledger_path.read_text(encoding="utf-8"))

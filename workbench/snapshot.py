"""Staged snapshot store: artifacts are written to a temp directory and
atomically swapped into place; the manifest is committed last."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from .config import canonical_json


class SnapshotStore:
    def __init__(self, root):
        self.root = Path(root)
        self.tiles_dir = self.root / "tiles"
        self.fusion_dir = self.root / "fusion"
        self.global_dir = self.root / "global"

    # ---- low level helpers -------------------------------------------------
    def _atomic_write_text(self, path: Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def write_json(self, rel: str, obj):
        self._atomic_write_text(self.root / rel, canonical_json(obj))

    def read_json(self, rel: str, default=None):
        path = self.root / rel
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    # ---- manifest ----------------------------------------------------------
    def load_manifest_state(self):
        return self.read_json("manifest_state.json", default=None)

    def commit_manifest_state(self, state: dict):
        self.write_json("manifest_state.json", state)

    # ---- tile results ------------------------------------------------------
    def save_tile_result(self, tile_id: str, result: dict, meta: dict):
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.tiles_dir / f"{tile_id}.tmp.npz"
        final = self.tiles_dir / f"{tile_id}.npz"
        np.savez(tmp, classes=result["classes"],
                 probs=result["probs"].astype(np.int64))
        os.replace(tmp, final)
        self.write_json(f"tiles/{tile_id}.json", meta)

    def load_tile_result(self, tile_id: str):
        npz_path = self.tiles_dir / f"{tile_id}.npz"
        meta_path = self.tiles_dir / f"{tile_id}.json"
        if not npz_path.exists() or not meta_path.exists():
            return None
        data = np.load(npz_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "classes": data["classes"],
            "probs": data["probs"].astype(np.int64),
            "score_summary": meta["score_summary"],
            "prob_summary": meta["prob_summary"],
        }

    def tile_meta(self, tile_id: str):
        return self.read_json(f"tiles/{tile_id}.json", default=None)

    # ---- fusion results ----------------------------------------------------
    def save_fusion_result(self, pair_key: str, result: dict, meta: dict):
        safe = pair_key.replace("::", "__")
        self.fusion_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.fusion_dir / f"{safe}.tmp.npz"
        np.savez(tmp, fused=result["fused"])
        os.replace(tmp, self.fusion_dir / f"{safe}.npz")
        payload = dict(meta)
        payload["records"] = result["records"]
        payload["boundary_events"] = result["boundary_events"]
        self.write_json(f"fusion/{safe}.json", payload)

    def load_fusion_result(self, pair_key: str):
        safe = pair_key.replace("::", "__")
        npz_path = self.fusion_dir / f"{safe}.npz"
        meta_path = self.fusion_dir / f"{safe}.json"
        if not npz_path.exists() or not meta_path.exists():
            return None
        data = np.load(npz_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "fused": data["fused"],
            "records": meta["records"],
            "boundary_events": meta["boundary_events"],
            "geometry": meta["geometry"],
        }

    def fusion_meta(self, pair_key: str):
        safe = pair_key.replace("::", "__")
        return self.read_json(f"fusion/{safe}.json", default=None)

    # ---- global artifacts --------------------------------------------------
    def save_global(self, global_class: np.ndarray, refills, report):
        self.global_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.global_dir / "global_class.tmp.npy"
        np.save(tmp, global_class)
        os.replace(tmp, self.global_dir / "global_class.npy")
        self.write_json("global/refills.json", refills)
        self.write_json("global/report.json", report)

    # ---- staged commit -----------------------------------------------------
    def stage_commit(self, stage: str):
        """Marker that a stage completed: recorded inside the manifest state."""
        marker = self.read_json("stage_markers.json", default={})
        marker[stage] = True
        self.write_json("stage_markers.json", marker)

    def reset(self):
        if self.root.exists():
            shutil.rmtree(self.root)

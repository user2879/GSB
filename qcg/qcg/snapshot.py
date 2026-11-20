"""Staged snapshot directory with atomic replace and resume判定."""
import json
import os
import shutil
from typing import Optional

from .errors import QCGError, E_CONFIG_MISMATCH

STAGES = ("load_validate", "integer_conv", "dequant_reduce", "summary")
MANIFEST_NAME = "manifest.json"


class SnapshotManager:
    def __init__(self, root: str):
        self.root = root
        self.stage_dir = os.path.join(root, "staging")
        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root, "products"), exist_ok=True)

    def manifest_path(self) -> str:
        return os.path.join(self.root, MANIFEST_NAME)

    def load_manifest(self) -> Optional[dict]:
        path = self.manifest_path()
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def check_resume(self, global_version: int, graph_hash: str,
                     config_hash: str) -> Optional[dict]:
        manifest = self.load_manifest()
        if manifest is None:
            return None
        mismatches = []
        if manifest.get("global_version") != global_version:
            mismatches.append(("global_version",
                               manifest.get("global_version"), global_version))
        if manifest.get("graph_hash") != graph_hash:
            mismatches.append(("graph_hash",
                               manifest.get("graph_hash"), graph_hash))
        if manifest.get("config_hash") != config_hash:
            mismatches.append(("config_hash",
                               manifest.get("config_hash"), config_hash))
        if mismatches:
            raise QCGError(E_CONFIG_MISMATCH,
                           "resume identity mismatch: %s" % mismatches,
                           field="manifest")
        return manifest

    def write_stage(self, stage: str, payload: dict) -> str:
        stage_tmp = os.path.join(self.stage_dir, stage + ".tmp")
        if os.path.exists(stage_tmp):
            shutil.rmtree(stage_tmp)
        os.makedirs(stage_tmp, exist_ok=True)
        path = os.path.join(stage_tmp, "result.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        final = os.path.join(self.stage_dir, stage)
        if os.path.exists(final):
            shutil.rmtree(final)
        os.replace(stage_tmp, final)
        return final

    def commit(self, manifest: dict) -> None:
        tmp_manifest = self.manifest_path() + ".tmp"
        with open(tmp_manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp_manifest, self.manifest_path())

    def reset_staging(self) -> None:
        if os.path.exists(self.stage_dir):
            shutil.rmtree(self.stage_dir)
        os.makedirs(self.stage_dir, exist_ok=True)

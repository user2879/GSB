"""Artifact store with staged, atomically-replaced stage snapshots and a
versioned manifest used for resume decisions."""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

MANIFEST = "manifest.json"


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"not json serializable: {type(obj)}")


def canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, default=_json_default)


class Store:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def manifest_path(self):
        return os.path.join(self.root, MANIFEST)

    def load_manifest(self):
        try:
            with open(self.manifest_path(), "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def write_manifest(self, manifest):
        tmp = self.manifest_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(canonical_json(manifest))
        os.replace(tmp, self.manifest_path())

    def stage_dir(self, stage: str):
        return os.path.join(self.root, "stages", stage)

    def commit_stage(self, stage: str, files: dict):
        """Atomically replace a whole stage directory with new contents.

        ``files`` maps relative file names to str payloads, or to
        {"json": payload} / {"npy": ndarray} wrappers.
        """
        final = self.stage_dir(stage)
        tmp = final + f".tmp-{os.getpid()}"
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        for rel, payload in files.items():
            path = os.path.join(tmp, rel)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if isinstance(payload, dict) and "npy" in payload:
                with open(path, "wb") as fh:
                    np.save(fh, payload["npy"])
            elif isinstance(payload, dict) and "json" in payload:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(canonical_json(payload["json"]))
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(str(payload))
        old = final + f".old-{os.getpid()}"
        if os.path.isdir(final):
            os.rename(final, old)
        os.rename(tmp, final)
        if os.path.isdir(old):
            shutil.rmtree(old)

    def read_stage_file(self, stage: str, rel: str):
        path = os.path.join(self.stage_dir(stage), rel)
        if not os.path.isfile(path):
            return None
        if rel.endswith(".npy"):
            with open(path, "rb") as fh:
                return np.load(fh)
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def list_stage_files(self, stage: str):
        base = self.stage_dir(stage)
        if not os.path.isdir(base):
            return []
        out = []
        for dirpath, _dirs, names in os.walk(base):
            for name in sorted(names):
                out.append(os.path.relpath(os.path.join(dirpath, name), base))
        return sorted(out)

    def snapshot(self, version: int):
        """Full copy of the store (minus snapshots) as snapshots/v<version>."""
        dst = os.path.join(self.root, "snapshots", f"v{version}")
        tmp = dst + f".tmp-{os.getpid()}"
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(
            self.root,
            tmp,
            ignore=shutil.ignore_patterns("snapshots", "*.tmp-*", "*.old-*"),
        )
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.rename(tmp, dst)

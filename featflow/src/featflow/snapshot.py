"""Stage snapshots with atomic directory replacement and stale detection."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any

from .common import (
    CODE_SNAPSHOT_STALE,
    Failure,
    canonical_dumps,
    read_json,
    sha256_text,
    write_json,
)

MANIFEST = "manifest.json"


class SnapshotStore:
    def __init__(self, root: str) -> None:
        self.root = root
        self.stage_dir = os.path.join(root, "stages")
        os.makedirs(self.stage_dir, exist_ok=True)

    # ------------------------------------------------------------- manifest IO
    def load_manifest(self) -> dict[str, Any] | None:
        path = os.path.join(self.root, MANIFEST)
        if not os.path.exists(path):
            return None
        return read_json(path)

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        write_json(os.path.join(self.root, MANIFEST), manifest)

    # ---------------------------------------------------------- stage artifacts
    def stage_path(self, stage: str, filename: str) -> str:
        directory = os.path.join(self.stage_dir, stage)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)

    def write_stage_file(self, stage: str, filename: str,
                         writer: CallableWrite, payload: Any) -> str:
        """Write into a stage temp directory then atomically move into place."""
        directory = os.path.join(self.stage_dir, stage)
        os.makedirs(directory, exist_ok=True)
        tmp_dir = os.path.join(self.root, f".tmp-{stage}")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, filename)
        writer(tmp_path, payload)
        final_path = os.path.join(directory, filename)
        os.replace(tmp_path, final_path)
        os.rmdir(tmp_dir)
        return final_path

    def hash_stage(self, stage: str) -> str:
        directory = os.path.join(self.stage_dir, stage)
        h = sha256_text("")
        import hashlib

        digest = hashlib.sha256()
        for name in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
            digest.update(name.encode("utf-8"))
            with open(os.path.join(directory, name), "rb") as fh:
                digest.update(fh.read())
        return digest.hexdigest()

    # --------------------------------------------------------------- staleness
    @staticmethod
    def check_stale(expected: dict[str, Any], current: dict[str, Any]) -> Failure | None:
        """Identify definition/cursor version drift before replacing a snapshot."""
        for key in ("definition_hash", "config_hash"):
            if expected.get(key) != current.get(key):
                return Failure("snapshot", key, CODE_SNAPSHOT_STALE,
                               f"expected {expected.get(key)} got {current.get(key)}")
        if expected.get("cursor_event_id") != current.get("cursor_event_id"):
            return Failure("snapshot", "cursor_event_id", CODE_SNAPSHOT_STALE,
                           "cursor moved since snapshot was staged")
        return None

    def reset(self) -> None:
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.stage_dir, exist_ok=True)


# type alias to keep the writer signature readable
CallableWrite = Any

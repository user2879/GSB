"""基于内容指纹的阶段产物缓存（命中即复用、不得重算）。"""
from __future__ import annotations

import os

from .utils import ensure_dir, read_json, write_json


class ArtifactCache:
    """以缓存键为文件名的 JSON 产物缓存。"""

    def __init__(self, root: str):
        self.root = ensure_dir(root)
        self.reused: list = []
        self.computed: list = []

    def _path(self, key: str) -> str:
        return os.path.join(self.root, f"{key}.json")

    def has(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def get(self, key: str):
        if self.has(key):
            self.reused.append(key)
            return read_json(self._path(key))
        return None

    def put(self, key: str, payload) -> None:
        self.computed.append(key)
        write_json(self._path(key), payload)

    def reset_ledger(self) -> None:
        self.reused.clear()
        self.computed.clear()

    def ledger(self) -> dict:
        return {"reused": sorted(self.reused), "computed": sorted(self.computed)}

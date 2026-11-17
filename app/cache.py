"""基于内容指纹的步骤结果缓存（增量失效）。"""
from __future__ import annotations

import json
from pathlib import Path


class StepCache:
    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str):
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, payload: dict) -> None:
        path = self._path(key)
        tmp = self.dir / f"{key}.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(path)

    def clear(self) -> None:
        for path in self.dir.glob("*.json"):
            path.unlink()

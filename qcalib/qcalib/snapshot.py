"""阶段快照：独立临时目录写入后整体替换，并识别版本变化。"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .fingerprint import content_fingerprint

MANIFEST_NAME = "manifest.json"


class Snapshot:
    """对目标目录做原子化整体替换的快照写入器。"""

    def __init__(self, target_dir: str | Path):
        self.target_dir = Path(target_dir)
        self.tmp_dir: Optional[Path] = None

    def __enter__(self) -> "Snapshot":
        self.target_dir.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{self.target_dir.name}.",
                dir=str(self.target_dir.parent),
            )
        )
        return self

    @property
    def path(self) -> Path:
        if self.tmp_dir is None:
            raise RuntimeError("Snapshot 尚未进入上下文")
        return self.tmp_dir

    def write_json(self, relative: str, payload: Any) -> Path:
        out = self.tmp_dir / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return out

    def write_text(self, relative: str, text: str) -> Path:
        out = self.tmp_dir / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return out

    def commit(self, manifest: Dict[str, Any]) -> str:
        manifest = dict(manifest)
        manifest["content_fingerprint"] = content_fingerprint(
            {k: v for k, v in manifest.items() if k != "content_fingerprint"}
        )
        self.write_json(MANIFEST_NAME, manifest)
        previous = self._previous_manifest()
        if self.target_dir.exists():
            shutil.rmtree(self.target_dir)
        os.replace(self.tmp_dir, self.target_dir)  # 同目录下原子 rename
        self.tmp_dir = None
        return manifest["content_fingerprint"]

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.tmp_dir is not None and self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
            self.tmp_dir = None

    def _previous_manifest(self) -> Optional[Dict[str, Any]]:
        path = self.target_dir / MANIFEST_NAME
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None


def version_changed(target_dir: str | Path, new_fingerprint: str) -> bool:
    """比较目标目录现有 manifest 与新指纹，识别版本是否变化。"""
    path = Path(target_dir) / MANIFEST_NAME
    if not path.exists():
        return True
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest.get("content_fingerprint") != new_fingerprint

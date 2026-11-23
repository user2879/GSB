"""阶段产物的内容寻址缓存、精确失效判定与稳定复用清单。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class CacheRecord:
    cache_key: str
    stage: str
    group_id: str
    column: Optional[str]
    row_refs: List[str]
    param_refs: Dict[str, Any]
    blob_path: str

    def index_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "group_id": self.group_id,
            "column": self.column,
            "row_refs": self.row_refs,
            "param_refs": self.param_refs,
            "blob_path": self.blob_path,
        }


class StageStore:
    """以 cache_key 为主键的磁盘缓存（index + blob）。"""

    INDEX_NAME = "cache_index.json"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.blobs_dir = self.root / "blobs"
        self.index_path = self.root / self.INDEX_NAME
        self.index: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        else:
            self.index = {}

    def has(self, cache_key: str) -> bool:
        return cache_key in self.index

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        meta = self.index.get(cache_key)
        if meta is None:
            return None
        blob = self.root / meta["blob_path"]
        if not blob.exists():
            return None
        return json.loads(blob.read_text(encoding="utf-8"))

    def put(
        self,
        record: CacheRecord,
        payload: Dict[str, Any],
    ) -> None:
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        blob = self.root / record.blob_path
        blob.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.index[record.cache_key] = record.index_dict()

    def save_index(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.index, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(self.index_path)

    def keys_for_group(self, group_id: str) -> List[str]:
        return sorted(
            key for key, meta in self.index.items() if meta["group_id"] == group_id
        )

    def invalidate_group(self, group_id: str) -> List[str]:
        removed = self.keys_for_group(group_id)
        for key in removed:
            meta = self.index.pop(key)
            blob = self.root / meta["blob_path"]
            if blob.exists():
                blob.unlink()
        return removed

    def group_stage_keys(self, group_id: str) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for key, meta in self.index.items():
            if meta["group_id"] != group_id:
                continue
            out.setdefault(meta["stage"], []).append(key)
        for stage in out:
            out[stage].sort()
        return out


def affected_groups(
    changed_rows: Set[str],
    changed_groups: Set[str],
    table_columns_by_group: Dict[str, List[str]],
    row_to_groups: Dict[str, List[str]],
) -> List[str]:
    """根据变更行与变更组，稳定地求出受影响列组集合。"""
    affected: Set[str] = set(changed_groups)
    for row_id in changed_rows:
        affected.update(row_to_groups.get(row_id, []))
    return sorted(affected)

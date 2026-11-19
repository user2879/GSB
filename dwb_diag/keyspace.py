"""键空间: 把用户键值变更编译为固定页上的负载变更。

页负载为规范化 JSON: {"v":1, "entries": {"key": value, ...}}
键通过确定性散列映射到页号; 值限定为 int, 供值区间视图排序。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .paging import MainDataFile, Page
from .util import canonical_json


class KeyspaceNotReadyError(RuntimeError):
    """恢复未完成前禁止按键空间作答。"""


def hash_key_to_page(key: str, capacity: int) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % capacity


def encode_payload(entries: dict[str, int]) -> bytes:
    return canonical_json({"v": 1, "entries": entries}).encode("utf-8")


def decode_payload(payload: bytes) -> dict[str, int]:
    if not payload:
        return {}
    obj = json.loads(payload.decode("utf-8"))
    return dict(obj.get("entries", {}))


@dataclass
class PageRead:
    page_no: int
    entries: dict[str, int]


class Keyspace:
    """基于主数据文件的键空间视图; 恢复完成前 _ready=False。"""

    def __init__(self, main: MainDataFile) -> None:
        self.main = main
        self.page = Page(main.page_size)
        self._ready = False

    def mark_ready(self) -> None:
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def page_for(self, key: str) -> int:
        return hash_key_to_page(key, self.main.capacity)

    def _require_ready(self) -> None:
        if not self._ready:
            raise KeyspaceNotReadyError(
                "keyspace answers are blocked until page recovery completes"
            )

    def read_page_entries(self, page_no: int) -> dict[str, int]:
        frame = self.main.read_slot(page_no)
        if frame is None:
            return {}
        try:
            return decode_payload(self.page.decode_payload(frame))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            # 损坏/半写页在恢复前可能无法解码; 依赖图构建等只读场景容错为空,
            # 真正的修复由恢复引擎依据缓冲镜像完成
            return {}

    def apply_change(
        self, page_no: int, mutations: dict[str, int | None]
    ) -> tuple[bytes, dict[str, int]]:
        """在指定页上应用变更(value=None 表示删除), 返回新帧与新条目集。"""
        entries = self.read_page_entries(page_no)
        for key, value in mutations.items():
            if value is None:
                entries.pop(key, None)
            else:
                entries[key] = int(value)
        entries = dict(sorted(entries.items()))
        payload = encode_payload(entries)
        return self.page.encode(payload), entries

    def get(self, key: str) -> int | None:
        self._require_ready()
        page_no = self.page_for(key)
        return self.read_page_entries(page_no).get(key)

    def scan(self) -> dict[str, int]:
        """恢复完成后的权威键空间全量读取, 按键排序。"""
        self._require_ready()
        merged: dict[str, int] = {}
        for page_no in range(self.main.capacity):
            merged.update(self.read_page_entries(page_no))
        return dict(sorted(merged.items()))

    def all_page_entries(self) -> dict[int, dict[str, int]]:
        return {
            page_no: self.read_page_entries(page_no)
            for page_no in range(self.main.capacity)
        }

    def digest(self) -> str:
        from .util import digest_of_json

        return digest_of_json(self.scan())

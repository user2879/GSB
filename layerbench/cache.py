"""跨构建缓存索引：文件 + SQLite 双持久化，容量预算下的稳定淘汰。

缓存判定与层摘要的关系（显式规则）
----------------------------------
执行类指令在运行前无法知道退出码与 stdout，因此缓存采用两级键：

1. ``precursor``（前驱指纹）= 哈希(指令类型, 规范化参数,
   稳定序输入文件摘要集合, 前驱层摘要集合)。该指纹在执行前即可计算，
   且与并行交错、就绪顺序、缓存状态无关。
2. ``layer``（层摘要）= 哈希(precursor, 命令结果摘要)。命令结果摘要含
   退出码与标准化 stdout 摘要，因此缓存命中跳过执行时，复用的是上次真实
   执行得到的结果，层摘要保持字节级一致。

非执行指令（copy/env/label）没有命令结果，层摘要直接等于
``hash(precursor, "no-command-result")``。

稳定淘汰规则（绝不依赖 dict/set 遍历顺序）
------------------------------------------
容量超限时按以下全序选择被逐出条目，取最小者：
``(last_used_ms 升序, 引用计数 refcount 升序, layer_digest 字典序)``，
即确定性 LRU + 引用计数 + 摘要字典序兜底。淘汰同步移除两个索引并记录原因。
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass

from . import digest

INDEX_FILENAME = "cache_index.json"
INDEX_VERSION = 1


@dataclass
class CacheEntry:
    layer_digest: str
    precursor: str
    input_digests: dict          # {相对路径: 摘要}，稳定序
    result_digest: str           # 命令结果摘要；非执行指令为固定标记
    exit_code: int | None
    stdout_digest: str | None
    refcount: int
    created_seq: int             # 写入顺序（单调），便于审计
    last_used_seq: int           # 最近命中/写入顺序
    last_used_ms: int

    def to_record(self) -> dict:
        return {
            "layer_digest": self.layer_digest,
            "precursor": self.precursor,
            "input_digests": dict(sorted(self.input_digests.items())),
            "result_digest": self.result_digest,
            "exit_code": self.exit_code,
            "stdout_digest": self.stdout_digest,
            "refcount": self.refcount,
            "created_seq": self.created_seq,
            "last_used_seq": self.last_used_seq,
            "last_used_ms": self.last_used_ms,
        }


class CacheIndex:
    """线程安全的缓存索引；所有遍历均显式排序。"""

    def __init__(self, path: str, capacity: int, clock_ms, now_ms):
        self.path = os.path.abspath(path)
        self.capacity = capacity
        self._clock_ms = clock_ms      # 单调时钟毫秒
        self._now_ms = now_ms          # 墙钟毫秒（仅用于 last_used_ms 审计）
        self._lock = threading.RLock()
        self._by_layer: dict[str, CacheEntry] = {}
        self._by_precursor: dict[str, str] = {}
        self._seq = 0
        self.evictions: list[dict] = []   # 本次会话内的淘汰记录

    # ---------- 持久化 ----------
    @classmethod
    def load(cls, directory: str, capacity: int, clock_ms, now_ms) -> "CacheIndex":
        path = os.path.join(directory, INDEX_FILENAME)
        index = cls(path, capacity, clock_ms, now_ms)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            for record in raw.get("entries", []):
                entry = CacheEntry(**record)
                index._by_layer[entry.layer_digest] = entry
                index._by_precursor[entry.precursor] = entry.layer_digest
            index._seq = raw.get("next_seq", 0)
        return index

    def save(self) -> None:
        with self._lock:
            payload = {
                "index_version": INDEX_VERSION,
                "capacity": self.capacity,
                "next_seq": self._seq,
                "entries": [self._by_layer[k].to_record()
                            for k in sorted(self._by_layer)],
            }
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True,
                          ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(tmp, self.path)


    @classmethod
    def fingerprint_file(cls, directory: str) -> str:
        """对磁盘上的缓存索引计算规范化内容摘要（不依赖文件排版）。

        容量从索引文件自身读取，避免用占位容量污染摘要。
        """
        index_path = os.path.join(directory, INDEX_FILENAME)
        if not os.path.isfile(index_path):
            return digest.stable_digest_of_mapping(
                {"capacity": 0, "entries": []}
            )
        with open(index_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        index = cls(index_path, raw.get("capacity", 0),
                    clock_ms=lambda: 0, now_ms=lambda: 0)
        for record in raw.get("entries", []):
            entry = CacheEntry(**record)
            index._by_layer[entry.layer_digest] = entry
            index._by_precursor[entry.precursor] = entry.layer_digest
        return index.fingerprint()

    def fingerprint(self) -> str:
        """缓存索引整体内容摘要（快照比对使用，稳定序）。"""
        records = [self._by_layer[k].to_record()
                   for k in sorted(self._by_layer)]
        return digest.stable_digest_of_mapping(
            {"capacity": self.capacity, "entries": records}
        )

    # ---------- 判定 ----------
    def lookup(self, precursor: str) -> CacheEntry | None:
        """命中则更新最近使用顺序与引用计数并返回条目；未命中返回 None。"""
        with self._lock:
            layer = self._by_precursor.get(precursor)
            if layer is None:
                return None
            entry = self._by_layer.get(layer)
            if entry is None:
                # 双索引不自洽属于索引损坏，交由上层按快照规则处理
                return None
            self._seq += 1
            entry.last_used_seq = self._seq
            entry.last_used_ms = self._now_ms()
            entry.refcount += 1
            return entry

    def insert(self, precursor: str, layer_digest: str, input_digests: dict,
               result_digest: str, exit_code: int | None,
               stdout_digest: str | None) -> tuple[CacheEntry, list[dict]]:
        """写入新层并按容量预算稳定淘汰；返回 (条目, 淘汰记录列表)。"""
        with self._lock:
            existing = self._by_precursor.get(precursor)
            if existing is not None:
                # 同前驱已存在（确定性结果下 layer 必相同），仅刷新复用
                entry = self._by_layer[existing]
                self._seq += 1
                entry.last_used_seq = self._seq
                entry.last_used_ms = self._now_ms()
                entry.refcount += 1
                return entry, []
            self._seq += 1
            entry = CacheEntry(
                layer_digest=layer_digest,
                precursor=precursor,
                input_digests=dict(sorted(input_digests.items())),
                result_digest=result_digest,
                exit_code=exit_code,
                stdout_digest=stdout_digest,
                refcount=1,
                created_seq=self._seq,
                last_used_seq=self._seq,
                last_used_ms=self._now_ms(),
            )
            self._by_layer[layer_digest] = entry
            self._by_precursor[precursor] = layer_digest
            evicted = self._enforce_capacity()
            return entry, evicted

    def _enforce_capacity(self) -> list[dict]:
        evictions: list[dict] = []
        # 显式全序排序，不依赖 dict/set 遍历顺序
        while len(self._by_layer) > self.capacity:
            ordered = sorted(
                self._by_layer.values(),
                key=lambda e: (e.last_used_seq, e.refcount, e.layer_digest),
            )
            victim = ordered[0]
            del self._by_layer[victim.layer_digest]
            self._by_precursor.pop(victim.precursor, None)
            record = {
                "layer_digest": victim.layer_digest,
                "precursor": victim.precursor,
                "reason": "capacity_lru_evict",
                "rule": "min(last_used_seq,refcount,layer_digest)",
                "refcount": victim.refcount,
                "abandoned_reuse": (
                    "容量超限被逐出，后续相同前驱只能未命中并重新真实执行"
                ),
                "at_ms": self._now_ms(),
            }
            evictions.append(record)
            self.evictions.append(record)
        return evictions

    def size(self) -> int:
        with self._lock:
            return len(self._by_layer)

    def layer_digests(self) -> list[str]:
        with self._lock:
            return sorted(self._by_layer)

    def replace_from_directory(self, directory: str) -> None:
        """整体替换内存索引（快照恢复使用，禁止增量合并）。"""
        path = os.path.join(directory, INDEX_FILENAME)
        with self._lock:
            self._by_layer.clear()
            self._by_precursor.clear()
            self.evictions.clear()
            self._seq = 0
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                for record in raw.get("entries", []):
                    entry = CacheEntry(**record)
                    self._by_layer[entry.layer_digest] = entry
                    self._by_precursor[entry.precursor] = entry.layer_digest
                self._seq = raw.get("next_seq", 0)
            self.path = path

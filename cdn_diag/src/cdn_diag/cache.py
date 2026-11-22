"""边缘节点缓存模型：按缓存键存取、TTL 递减、确定性逐出。

时间基准为编排器下发的逻辑单调时钟 tick（毫秒），保证相同请求序列
在多次运行中得到逐字节一致的命中与逐出结论。
"""

from dataclasses import dataclass, asdict

SOURCE_AUTHORITATIVE = "authoritative"   # 权威内容（传播替换）
SOURCE_REVALIDATED = "revalidated"       # 过期重校验
SOURCE_MISS_FETCH = "miss_fetch"         # 未命中回源


@dataclass
class Entry:
    cache_key: str
    path: str
    etag: str
    body: bytes
    ttl_ms: int
    write_tick_ms: int
    entry_version: int
    source: str
    last_access_tick_ms: int

    def remaining_ttl(self, tick_ms):
        return self.ttl_ms - (tick_ms - self.write_tick_ms)

    def to_dict(self):
        d = asdict(self)
        d["body"] = self.body.hex()
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["body"] = bytes.fromhex(d["body"])
        return Entry(**d)


class NodeCache:
    def __init__(self, node_id, capacity, rule, key_fields):
        self.node_id = node_id
        self.capacity = capacity
        self.rule = rule
        self.key_fields = tuple(key_fields)
        self._entries = {}       # cache_key -> Entry
        self._versions = {}      # cache_key -> 最近条目版本号

    def make_key(self, record):
        return "|".join(str(record[f]) for f in self.key_fields)

    def _key_for(self, path):
        return self.make_key({"node_id": self.node_id, "path": path})

    def lookup(self, path, tick_ms):
        """返回 (status, entry, remaining_ttl, evicted)。expired 时回收条目。"""
        key = self._key_for(path)
        entry = self._entries.get(key)
        if entry is None:
            return "miss", None, 0, None
        remaining = entry.remaining_ttl(tick_ms)
        if remaining <= 0:
            evicted = self._entries.pop(key)
            return "expired", None, 0, evicted
        entry.last_access_tick_ms = tick_ms
        return "hit", entry, remaining, None

    def _next_version(self, key):
        version = self._versions.get(key, 0) + 1
        self._versions[key] = version
        return version

    def _evict_until_fit(self):
        evicted = []
        while len(self._entries) >= self.capacity:
            if self.rule == "lru":
                victim_key = min(
                    self._entries,
                    key=lambda k: (self._entries[k].last_access_tick_ms, k),
                )
            else:  # fifo
                victim_key = min(
                    self._entries,
                    key=lambda k: (self._entries[k].write_tick_ms, k),
                )
            evicted.append(self._entries.pop(victim_key))
        return evicted

    def store(self, path, etag, body, ttl_ms, tick_ms, source):
        key = self._key_for(path)
        evicted = []
        if key not in self._entries:
            evicted = self._evict_until_fit()
        entry = Entry(
            cache_key=key,
            path=path,
            etag=etag,
            body=body,
            ttl_ms=ttl_ms,
            write_tick_ms=tick_ms,
            entry_version=self._next_version(key),
            source=source,
            last_access_tick_ms=tick_ms,
        )
        self._entries[key] = entry
        return entry, evicted

    def replace(self, path, etag, body, ttl_ms, tick_ms):
        """派生失效传播：整体替换条目；不存在则返回 None。"""
        key = self._key_for(path)
        old = self._entries.get(key)
        if old is None:
            return None
        old_version = old.entry_version
        entry = Entry(
            cache_key=key,
            path=path,
            etag=etag,
            body=body,
            ttl_ms=ttl_ms,
            write_tick_ms=tick_ms,
            entry_version=self._next_version(key),
            source=SOURCE_AUTHORITATIVE,
            last_access_tick_ms=tick_ms,
        )
        self._entries[key] = entry
        return old_version, entry.entry_version

    def restore(self, entries):
        """仅依据已持久化条目恢复，不保留任何内存状态。"""
        self._entries = {}
        self._versions = {}
        for entry in entries:
            self._entries[entry.cache_key] = entry
            self._versions[entry.cache_key] = max(
                self._versions.get(entry.cache_key, 0), entry.entry_version
            )

    def dump(self):
        return [self._entries[k] for k in sorted(self._entries)]

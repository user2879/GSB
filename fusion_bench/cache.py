"""缓存驻留与淘汰：块级驻留，LRU 淘汰，同访问序号按块键字典序决断。

被淘汰块再次引用时从原始输入重算，并与首次计算结果逐元素比对，
淘汰与重算事件写入审计清单。
"""

import numpy as np


class BlockCache:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self.store = {}          # key -> (value, last_access_seq)
        self.clock = 0
        self.evictions = []      # 淘汰审计
        self.recomputes = []     # 重算审计
        self._first_values = {}  # key -> 首次计算值（用于重算一致性核验）

    def access(self, key, compute_fn):
        """访问块：命中返回值；未命中调用 compute_fn 从原始输入重算。"""
        self.clock += 1
        seq = self.clock
        if key in self.store:
            value, _ = self.store[key]
            self.store[key] = (value, seq)
            return value
        value = compute_fn()
        if key in self._first_values:
            verified = bool(np.array_equal(value, self._first_values[key]))
            self.recomputes.append({
                "seq": seq,
                "block_key": key,
                "verified_elementwise": verified,
            })
        else:
            self._first_values[key] = np.asarray(value).copy()
        if len(self.store) >= self.capacity:
            victim_key = min(self.store.items(),
                             key=lambda kv: (kv[1][1], kv[0]))[0]
            del self.store[victim_key]
            self.evictions.append({
                "seq": seq,
                "evicted_key": victim_key,
                "inserted_key": key,
            })
        self.store[key] = (value, seq)
        return value

    def audit(self):
        return {"evictions": list(self.evictions),
                "recomputes": list(self.recomputes)}

    def state(self):
        """可快照状态：驻留块、访问序号与首次计算值，保证续跑后
        淘汰/重算序列与一次性全量执行逐字段一致。"""
        return {
            "capacity": self.capacity,
            "clock": self.clock,
            "store": {k: [[int(x) for x in v], seq]
                      for k, (v, seq) in self.store.items()},
            "first_values": {k: [int(x) for x in v]
                             for k, v in self._first_values.items()},
        }

    def restore(self, state):
        self.clock = int(state["clock"])
        self.store = {k: (np.array(v, dtype=np.int64), int(seq))
                      for k, (v, seq) in state["store"].items()}
        self._first_values = {k: np.array(v, dtype=np.int64)
                              for k, v in state["first_values"].items()}

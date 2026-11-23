"""基于内容指纹的块级缓存键与精确失效判定。

缓存键派生自：上游块内容指纹 + 算子参数指纹 + 量化档位。
任何上游内容或参数变化都会改变键，从而只失效受影响块及其下游。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np


def _sha(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, str):
            p = p.encode("utf-8")
        h.update(p)
        h.update(b"|")
    return h.hexdigest()[:24]


def params_fingerprint(cfg) -> str:
    """算子参数指纹：含块长、头维度、定点位宽、量化档位、稳定偏移与容差。"""
    return _sha(json.dumps(cfg.canonical(), sort_keys=True))


def tensor_fp(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    return _sha(str(arr.shape), str(arr.dtype), arr.tobytes())


class BlockCache:
    """三级缓存：键块乘积（score）、在线 softmax 状态（state）、输出累加（out）。"""

    def __init__(self, cfg):
        self.params_fp = params_fingerprint(cfg)
        self.quant_level = cfg.quantization_level_bits
        self._store = {}
        self.hits = []      # [(kind, block_id)]
        self.misses = []    # [(kind, block_id)]
        # 输出块 -> 引用的键块与状态集合
        self.deps = {}

    def score_key(self, qi, ki, q_block_fp, k_block_fp):
        return _sha("score", self.params_fp, str(self.quant_level),
                    q_block_fp, k_block_fp)

    def state_key(self, qi, score_keys):
        return _sha("state", self.params_fp, str(self.quant_level),
                    *[score_keys[k] for k in sorted(score_keys)])

    def out_key(self, qi, state_keys, v_block_fps):
        return _sha("out", self.params_fp, str(self.quant_level),
                    *[state_keys[k] for k in sorted(state_keys)],
                    *[v_block_fps[k] for k in sorted(v_block_fps)])

    def lookup(self, kind, block_id, key):
        if key in self._store:
            self.hits.append((kind, block_id))
            return self._store[key]
        self.misses.append((kind, block_id))
        return None

    def store(self, key, value):
        self._store[key] = value

    def record_out_deps(self, out_key, key_blocks, state_keys):
        self.deps[out_key] = {
            "key_blocks": sorted(key_blocks),
            "state_keys": sorted(state_keys),
        }

    def reuse_recompute_lists(self):
        """输出复用与重算清单。"""
        return {
            "reuse": [{"kind": k, "block": b} for k, b in self.hits],
            "recompute": [{"kind": k, "block": b} for k, b in self.misses],
        }

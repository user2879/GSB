"""固定种子确定性样本数据集：基础张量 + 突发查询块逐批变化。"""

from __future__ import annotations

import hashlib

import numpy as np

from .tensor_pack import read_pack, write_pack


def pad_to_multiple(length: int, block: int) -> int:
    return ((length + block - 1) // block) * block


class SampleDataset:
    """本地确定性样本，不依赖外部下载或真实模型权重。"""

    def __init__(self, seed=20240501, s_q=20, s_k=24, head_dim=8,
                 query_block_len=8, key_block_len=8, burst_query_blocks=()):
        self.seed = seed
        self.query_block_len = query_block_len
        self.key_block_len = key_block_len
        self.head_dim = head_dim
        self.orig_s_q = s_q
        self.orig_s_k = s_k
        self.s_q = pad_to_multiple(s_q, query_block_len)
        self.s_k = pad_to_multiple(s_k, key_block_len)
        self.burst_query_blocks = tuple(burst_query_blocks)
        self._burst_overrides = {}
        rng = np.random.default_rng(seed)
        self.Q = rng.normal(0.0, 0.5, (self.s_q, head_dim))
        self.K = rng.normal(0.0, 0.5, (self.s_k, head_dim))
        self.V = rng.normal(0.0, 0.5, (self.s_k, head_dim))
        self.meta = {
            "seed": seed,
            "pad_rule": "zero",
            "orig_shape": {
                "Q": [s_q, head_dim],
                "K": [s_k, head_dim],
                "V": [s_k, head_dim],
            },
            "burst_query_blocks": list(self.burst_query_blocks),
        }

    @property
    def num_query_blocks(self):
        return self.s_q // self.query_block_len

    @property
    def num_key_blocks(self):
        return self.s_k // self.key_block_len

    def tensors_for_batch(self, batch: int) -> dict:
        """突发查询块的查询数据在批次间变化，其余块稳定。

        突发清单变更后，旧突发块保留其最后一批数据并恢复为稳定块，
        因此其已有产物可被内容指纹缓存直接复用。
        """
        q = self.Q.copy()
        for qi, data in self._burst_overrides.items():
            if qi not in self.burst_query_blocks:
                lo = qi * self.query_block_len
                q[lo:lo + self.query_block_len] = data
        for qi in self.burst_query_blocks:
            rng = np.random.default_rng([self.seed, batch, qi])
            lo = qi * self.query_block_len
            hi = lo + self.query_block_len
            data = rng.normal(0.0, 0.5, (self.query_block_len, self.head_dim))
            q[lo:hi] = data
            self._burst_overrides[qi] = data
        return {"Q": q, "K": self.K.copy(), "V": self.V.copy()}

    def set_burst_query_blocks(self, blocks) -> None:
        self.burst_query_blocks = tuple(blocks)

    def write_pack(self, path) -> str:
        write_pack(path, {"Q": self.Q, "K": self.K, "V": self.V}, self.meta)
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]

    @classmethod
    def from_pack(cls, path, query_block_len, key_block_len,
                  burst_query_blocks=None):
        tensors, meta = read_pack(path)
        ds = cls.__new__(cls)
        ds.seed = meta.get("seed", 0)
        ds.query_block_len = query_block_len
        ds.key_block_len = key_block_len
        ds.head_dim = tensors["Q"].shape[1]
        ds.s_q, ds.s_k = tensors["Q"].shape[0], tensors["K"].shape[0]
        ds.orig_s_q = meta.get("orig_shape", {}).get("Q", [ds.s_q])[0]
        ds.orig_s_k = meta.get("orig_shape", {}).get("K", [ds.s_k])[0]
        ds.burst_query_blocks = tuple(
            burst_query_blocks if burst_query_blocks is not None
            else meta.get("burst_query_blocks", []))
        ds._burst_overrides = {}
        ds.Q, ds.K, ds.V = tensors["Q"], tensors["K"], tensors["V"]
        ds.meta = meta
        return ds

"""Block slicing with zero padding and content fingerprints."""
from __future__ import annotations

import numpy as np

from .config import LayoutInfo, PrefilterConfig
from .util import short_hash


def pad_and_slice(tensor: np.ndarray, seq_len: int, block: int, pad: int) -> list[np.ndarray]:
    if pad:
        padding = np.zeros((pad, tensor.shape[1]), dtype=tensor.dtype)
        tensor = np.concatenate([tensor, padding], axis=0)
    return [np.ascontiguousarray(tensor[start:start + block])
            for start in range(0, seq_len + pad, block)]


def build_blocks(pack_tensors: dict[str, np.ndarray], layout: LayoutInfo,
                 pre: PrefilterConfig) -> dict[str, list[np.ndarray]]:
    query = pack_tensors["query"]
    key = pack_tensors["key"]
    value = pack_tensors["value"]
    return {
        "query": pad_and_slice(query, layout.seq_q, pre.q_block, layout.q_pad),
        "key": pad_and_slice(key, layout.seq_k, pre.k_block, layout.k_pad),
        "value": pad_and_slice(value, layout.seq_k, pre.k_block, layout.k_pad),
    }


def block_fingerprint(block: np.ndarray) -> str:
    return short_hash(np.ascontiguousarray(block, dtype="<f4").tobytes())


def all_block_fingerprints(blocks: list[np.ndarray]) -> list[str]:
    return [block_fingerprint(block) for block in blocks]

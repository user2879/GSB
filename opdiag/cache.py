"""节点缓存键：参数指纹 + 输入内容指纹 + 精度档位 + 算子实现版本号。"""

from __future__ import annotations

from .hashing import canonical_hash
from . import IMPLEMENTATION_VERSION


def node_cache_key(node_spec: dict, input_fingerprints: list[str],
                   precision: str, bit_width: int, quant_mode: str) -> str:
    material = {
        "impl_version": IMPLEMENTATION_VERSION,
        "op": node_spec.get("op"),
        "id": node_spec.get("id"),
        "params": node_spec.get("params", {}),
        "perm": node_spec.get("perm"),
        "inplace": bool(node_spec.get("inplace", False)),
        "input_fingerprints": input_fingerprints,
        "precision": precision,
        "bit_width": bit_width,
        "quant_mode": quant_mode,
    }
    return canonical_hash(material)

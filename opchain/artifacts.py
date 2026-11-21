"""Artifact records and deterministic cache-key derivation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .chain import Node
from .config import ExecutionConfig
from .numeric import ACTIVATION_IMPL_VERSION, FUSION_IMPL_VERSION, PERMUTE_IMPL_VERSION
from .util import sha256_json, tensor_fingerprint, to_jsonable


@dataclass
class Artifact:
    node_id: str
    op: str
    values: np.ndarray
    cache_key: str
    metadata: dict[str, Any]
    raw_values: np.ndarray | None = None

    @property
    def values_hash(self) -> str:
        return tensor_fingerprint(self.values)

    @property
    def raw_values_hash(self) -> str | None:
        return tensor_fingerprint(self.raw_values) if self.raw_values is not None else None

    def as_record(self, *, include_values: bool = True, status: str = "computed", topo_rank: int | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": self.node_id,
            "op": self.op,
            "status": status,
            "cache_key": self.cache_key,
            "values_hash": self.values_hash,
            "shape": list(self.values.shape),
            "dtype": self.values.dtype.str,
            "metadata": to_jsonable(self.metadata),
        }
        if topo_rank is not None:
            record["topological_rank"] = topo_rank
        if include_values:
            record["values"] = to_jsonable(self.values)
        return record


def node_parameters_fingerprint(node: Node) -> str:
    return sha256_json({"op": node.op, "params": node.normalized_params, "inplace": node.inplace})


def make_cache_key(
    node: Node,
    input_fingerprints: list[str],
    config: ExecutionConfig,
    *,
    fusion_descriptor: dict[str, Any] | None = None,
) -> str:
    if node.op in {"relu", "gelu", "sigmoid", "tanh"}:
        implementation_version = ACTIVATION_IMPL_VERSION
    elif fusion_descriptor is not None:
        implementation_version = FUSION_IMPL_VERSION
    else:
        implementation_version = PERMUTE_IMPL_VERSION
    material = {
        "node_id": node.id,
        "op": node.op,
        "parameters": node.normalized_params,
        "parameters_fingerprint": node_parameters_fingerprint(node),
        "input_fingerprints": list(input_fingerprints),
        "inplace": node.inplace,
        "precision_tier": config.precision_tier,
        "quantization_tier": config.quantization_tier,
        "implementation_version": implementation_version,
        "fusion": fusion_descriptor,
    }
    return sha256_json(material)


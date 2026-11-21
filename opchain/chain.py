"""Operator-chain parsing, topology checks, parameter checks and shape inference."""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from typing import Any, Iterable

from .errors import (
    E_CHAIN_INVALID,
    E_CYCLE_DETECTED,
    E_PERM_INVALID,
    BusinessError,
)
from .input_spec import InputSpec
from .util import canonical_json, sha256_json


ACTIVATION_OPS = {"relu", "gelu", "sigmoid", "tanh"}
PERMUTE_OPS = {"permute", "transpose"}
SUPPORTED_OPS = ACTIVATION_OPS | PERMUTE_OPS


@dataclass
class Node:
    id: str
    op: str
    inputs: list[str]
    params: dict[str, Any]
    inplace: bool
    order: int
    raw: dict[str, Any]
    normalized_params: dict[str, Any]
    output_shape: tuple[int, ...] = ()

    @property
    def is_activation(self) -> bool:
        return self.op in ACTIVATION_OPS

    @property
    def is_permute(self) -> bool:
        return self.op in PERMUTE_OPS

    def descriptor(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "inputs": list(self.inputs),
            "params": self.normalized_params,
            "inplace": self.inplace,
        }


@dataclass
class Chain:
    nodes: list[Node]
    by_id: dict[str, Node]
    input_ids: list[str]
    consumers: dict[str, list[str]]
    producers: dict[str, str]
    topo_order: list[str]
    topo_rank: dict[str, int]
    outputs: list[str]
    raw: dict[str, Any]

    def downstream(self, seeds: Iterable[str]) -> set[str]:
        result: set[str] = set()
        stack = [seed for seed in seeds if seed in self.by_id]
        while stack:
            current = stack.pop()
            for child in self.consumers.get(current, []):
                if child not in result:
                    result.add(child)
                    stack.append(child)
        return result

    def nodes_in_topo(self) -> list[Node]:
        return [self.by_id[node_id] for node_id in self.topo_order]

    def signature(self) -> str:
        return sha256_json({
            "nodes": [self.by_id[node_id].descriptor() for node_id in self.topo_order],
            "inputs": self.input_ids,
            "outputs": self.outputs,
        })

    def node_descriptor_map(self) -> dict[str, dict[str, Any]]:
        return {node.id: node.descriptor() for node in self.nodes}


def _invalid(message: str, details: dict[str, Any] | None = None) -> BusinessError:
    return BusinessError(E_CHAIN_INVALID, message, stage="chain.validation", details=details or {})


def _parse_inputs(raw_node: dict[str, Any]) -> list[str]:
    inputs = raw_node.get("inputs", raw_node.get("input", []))
    if isinstance(inputs, str):
        inputs = [inputs]
    if inputs is None:
        inputs = []
    if not isinstance(inputs, list) or any(not isinstance(item, str) for item in inputs):
        raise _invalid("node inputs must be a list of tensor IDs")
    return list(dict.fromkeys(inputs))


def _normalize_activation_params(op: str, params: Any) -> dict[str, Any]:
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise _invalid(f"activation node {op!r} parameters must be an object")

    if op == "relu":
        allowed = {"negative_slope", "alpha"}
        unknown = set(params) - allowed
        if unknown:
            raise _invalid(f"unknown ReLU parameters: {sorted(unknown)}")
        key = "negative_slope" if "negative_slope" in params else "alpha" if "alpha" in params else None
        slope = 0.0 if key is None else params[key]
        if isinstance(slope, bool) or not isinstance(slope, (int, float)):
            raise _invalid("ReLU negative_slope/alpha must be numeric")
        if slope < 0:
            raise _invalid("ReLU negative_slope/alpha must be non-negative")
        return {"negative_slope": float(slope)}

    if op == "gelu":
        allowed = {"approximation", "approximate"}
        unknown = set(params) - allowed
        if unknown:
            raise _invalid(f"unknown GELU parameters: {sorted(unknown)}")
        approximation = params.get("approximation", params.get("approximate", "erf"))
        if approximation not in {"erf", "tanh"}:
            raise _invalid("GELU approximation must be 'erf' or 'tanh'")
        return {"approximation": approximation}

    if params:
        raise _invalid(f"{op} does not accept parameters")
    return {}


def _normalize_perm_params(op: str, params: Any, input_shape: tuple[int, ...]) -> tuple[dict[str, Any], tuple[int, ...]]:
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise BusinessError(E_PERM_INVALID, f"{op} parameters must be an object", stage="chain.permutation")
    allowed = {"perm", "permutation", "dims", "dimensions", "axes"}
    unknown = set(params) - allowed
    if unknown:
        raise BusinessError(
            E_PERM_INVALID,
            f"unknown permutation parameters: {sorted(unknown)}",
            stage="chain.permutation",
        )
    perm = params.get("perm", params.get("permutation", params.get("dims", params.get("dimensions", params.get("axes")))))
    rank = len(input_shape)
    if perm is None:
        if op == "transpose" and rank == 2:
            perm = [1, 0]
        else:
            raise BusinessError(
                E_PERM_INVALID,
                "permutation list is required unless a 2-D transpose is requested",
                stage="chain.permutation",
                details={"rank": rank},
            )
    if not isinstance(perm, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in perm):
        raise BusinessError(E_PERM_INVALID, "permutation must be a list of integers", stage="chain.permutation")
    if len(perm) != rank:
        raise BusinessError(
            E_PERM_INVALID,
            "permutation length must equal input rank",
            stage="chain.permutation",
            details={"rank": rank, "perm": perm},
        )
    if sorted(perm) != list(range(rank)):
        raise BusinessError(
            E_PERM_INVALID,
            "permutation must be zero-based, unique and cover every dimension",
            stage="chain.permutation",
            details={"rank": rank, "perm": perm},
        )
    normalized = tuple(int(item) for item in perm)
    return {"perm": list(normalized)}, normalized


def parse_chain(raw: dict[str, Any] | str, input_spec: InputSpec) -> Chain:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise _invalid("chain must be a JSON object")
    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list):
        raise _invalid("chain.nodes must be a list")

    node_ids: list[str] = []
    seen: set[str] = set()
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            raise _invalid("each node must be an object")
        node_id = raw_node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise _invalid("each node must have a non-empty string id")
        if node_id in seen:
            raise _invalid(f"duplicate node id: {node_id}", {"node_id": node_id})
        seen.add(node_id)
        node_ids.append(node_id)

    external_ids_raw = raw.get("inputs", raw.get("input", [input_spec.name]))
    if isinstance(external_ids_raw, str):
        external_ids_raw = [external_ids_raw]
    if not isinstance(external_ids_raw, list) or any(not isinstance(item, str) for item in external_ids_raw):
        raise _invalid("chain.inputs must be a string or a list of strings")
    input_ids = list(dict.fromkeys(external_ids_raw))
    if input_spec.name not in input_ids:
        raise _invalid(
            f"chain does not reference the provided input tensor {input_spec.name!r}",
            {"input_ids": input_ids, "available_input": input_spec.name},
        )
    if any(node_id in input_ids for node_id in node_ids):
        raise _invalid("input tensor IDs and node output IDs must be disjoint")

    explicit_inputs: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for index, raw_node in enumerate(raw_nodes):
        node_id = node_ids[index]
        explicit_inputs[node_id] = _parse_inputs(raw_node)

    edges = raw.get("edges", [])
    if edges is None:
        edges = []
    if not isinstance(edges, list):
        raise _invalid("chain.edges must be a list")
    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            edge = {"from": edge[0], "to": edge[1]}
        if not isinstance(edge, dict):
            raise _invalid("each edge must be an object or a two-item pair")
        source = edge.get("from", edge.get("source"))
        target = edge.get("to", edge.get("target"))
        if not isinstance(source, str) or not isinstance(target, str):
            raise _invalid("edge endpoints must be strings", {"edge": edge})
        if target not in seen:
            raise _invalid(f"edge target does not exist: {target}", {"edge": edge})
        if source not in seen and source not in input_ids:
            raise _invalid(f"edge source does not exist: {source}", {"edge": edge})
        if source not in explicit_inputs[target]:
            explicit_inputs[target].append(source)

    # Each tensor is produced by at most one node; node IDs are unique, and external IDs are disjoint.
    producers: dict[str, str] = {node_id: node_id for node_id in node_ids}
    for input_id in input_ids:
        producers[input_id] = input_id

    for node_id in node_ids:
        values = explicit_inputs[node_id]
        if len(values) != 1:
            raise _invalid(
                f"node {node_id!r} must have exactly one input tensor",
                {"node_id": node_id, "inputs": values},
            )
        source = values[0]
        if source not in producers:
            raise _invalid(
                f"node {node_id!r} references unknown tensor {source!r}",
                {"node_id": node_id, "tensor": source},
            )

    indegree = {node_id: 0 for node_id in node_ids}
    consumers: dict[str, list[str]] = {tensor_id: [] for tensor_id in producers}
    for node_id in node_ids:
        source = explicit_inputs[node_id][0]
        if source in seen:
            indegree[node_id] += 1
        consumers.setdefault(source, []).append(node_id)

    heap: list[tuple[int, str]] = []
    index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
    for node_id in node_ids:
        if indegree[node_id] == 0:
            heapq.heappush(heap, (index_by_id[node_id], node_id))
    topo_order: list[str] = []
    while heap:
        _, node_id = heapq.heappop(heap)
        topo_order.append(node_id)
        for child in consumers.get(node_id, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(heap, (index_by_id[child], child))
    if len(topo_order) != len(node_ids):
        raise BusinessError(E_CYCLE_DETECTED, "operator chain contains a cycle", stage="chain.cycle")

    nodes: dict[str, Node] = {}
    shape_by_tensor: dict[str, tuple[int, ...]] = {input_spec.name: tuple(input_spec.shape)}
    for node_id in topo_order:
        index = index_by_id[node_id]
        raw_node = raw_nodes[index]
        op = str(raw_node.get("op", "")).lower()
        op = {"transpose2d": "transpose", "transpose_2d": "transpose", "reorder": "permute"}.get(op, op)
        if op not in SUPPORTED_OPS:
            raise _invalid(f"unsupported operator: {raw_node.get('op')!r}", {"node_id": node_id})
        input_id = explicit_inputs[node_id][0]
        if input_id not in shape_by_tensor:
            raise _invalid(f"cannot infer input shape for node {node_id!r}", {"node_id": node_id})
        input_shape = shape_by_tensor[input_id]
        raw_params = raw_node.get("params", {})
        try:
            if op in ACTIVATION_OPS:
                normalized_params = _normalize_activation_params(op, raw_params)
                output_shape = input_shape
            else:
                normalized_params, perm = _normalize_perm_params(op, raw_params, input_shape)
                output_shape = tuple(input_shape[axis] for axis in perm)
        except BusinessError as exc:
            if exc.node_id is None:
                exc.node_id = node_id
            exc.details.setdefault("node_id", node_id)
            raise
        inplace = raw_node.get("inplace", False)
        if not isinstance(inplace, bool):
            raise _invalid(f"node {node_id!r} inplace flag must be boolean", {"node_id": node_id})
        node = Node(
            id=node_id,
            op=op,
            inputs=[input_id],
            params=dict(raw_params) if isinstance(raw_params, dict) else {},
            inplace=inplace,
            order=index,
            raw=raw_node,
            normalized_params=normalized_params,
            output_shape=output_shape,
        )
        nodes[node_id] = node
        shape_by_tensor[node_id] = output_shape

    outputs_raw = raw.get("outputs")
    if outputs_raw is None:
        outputs = [node_id for node_id in topo_order if not consumers.get(node_id)]
    else:
        if isinstance(outputs_raw, str):
            outputs_raw = [outputs_raw]
        if not isinstance(outputs_raw, list) or any(not isinstance(item, str) for item in outputs_raw):
            raise _invalid("chain.outputs must be a list of node IDs")
        outputs = list(dict.fromkeys(outputs_raw))
        for output in outputs:
            if output not in nodes:
                raise _invalid(f"unknown chain output: {output!r}")

    ordered_nodes = [nodes[node_id] for node_id in node_ids]
    return Chain(
        nodes=ordered_nodes,
        by_id=nodes,
        input_ids=input_ids,
        consumers={key: list(value) for key, value in consumers.items()},
        producers=producers,
        topo_order=topo_order,
        topo_rank={node_id: index for index, node_id in enumerate(topo_order)},
        outputs=outputs,
        raw=raw,
    )


def normalization_signature(value: Any) -> str:
    return canonical_json(value)


def fusion_candidates(chain: Chain, enabled: bool) -> dict[str, str]:
    """Map activation node ID to its directly-consumed permutation node ID."""

    if not enabled:
        return {}
    candidates: dict[str, str] = {}
    for node in chain.nodes:
        if not node.is_activation or node.inplace or node.id in chain.outputs:
            continue
        users = chain.consumers.get(node.id, [])
        if len(users) != 1:
            continue
        target = chain.by_id[users[0]]
        if target.is_permute and not target.inplace and target.inputs == [node.id]:
            candidates[node.id] = target.id
    return candidates

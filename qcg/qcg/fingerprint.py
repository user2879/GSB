"""Content fingerprints and per-node cache keys."""
import hashlib
import json
from typing import Dict, List, Set

import numpy as np

from .kernel import KERNEL_IMPL_VERSION
from .dequant import DEQUANT_IMPL_VERSION
from .models import Graph, Node, NODE_INPUT


def tensor_fingerprint(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.astype(np.int64).tobytes())
    return h.hexdigest()


def node_param_fingerprint(node: Node) -> str:
    payload = {
        "kind": node.kind,
        "bitwidth": node.bitwidth,
        "scale_permille": node.scale_permille,
        "scale_float": node.scale_float,
        "precision_tag": node.precision_tag,
        "rounding": node.rounding,
        "inplace": node.inplace,
        "kernel": node.kernel,
        "stride": node.stride,
        "padding": node.padding,
        "kernel_impl": KERNEL_IMPL_VERSION,
        "dequant_impl": DEQUANT_IMPL_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def upstream_sets(graph: Graph) -> Dict[str, Set[str]]:
    """Transitive upstream node set for every node."""
    by_id = graph.by_id()

    def collect(nid: str, memo: dict) -> Set[str]:
        if nid in memo:
            return memo[nid]
        result: Set[str] = set()
        for parent in by_id[nid].inputs:
            result.add(parent)
            result |= collect(parent, memo)
        memo[nid] = result
        return result

    memo: Dict[str, Set[str]] = {}
    return {n.node_id: collect(n.node_id, memo) for n in graph.nodes}


def compute_cache_key(node: Node, tensor_fp: Dict[str, str]) -> str:
    """Cache key derived from quant params, input tensor fingerprints,
    rounding mode, inplace flag and implementation version."""
    h = hashlib.sha256()
    h.update(node_param_fingerprint(node).encode("utf-8"))
    for parent in node.inputs:
        h.update(parent.encode("utf-8"))
        h.update(tensor_fp.get(parent, "").encode("utf-8"))
    if node.kind == NODE_INPUT:
        h.update(b"input-node")
    return h.hexdigest()


def compute_cache_keys(graph: Graph, tensor_fp: Dict[str, str]) -> Dict[str, str]:
    return {n.node_id: compute_cache_key(n, tensor_fp)
            for n in graph.topo_order()}


def graph_content_hash(graph: Graph) -> str:
    payload = [n.__dict__ for n in graph.topo_order()]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def downstream_closure(graph: Graph, seeds: List[str]) -> Set[str]:
    """All nodes that read any seed tensor, transitively (includes seeds)."""
    children: Dict[str, List[str]] = {n.node_id: [] for n in graph.nodes}
    for n in graph.nodes:
        for parent in n.inputs:
            children[parent].append(n.node_id)
    affected: Set[str] = set()
    stack = list(seeds)
    while stack:
        nid = stack.pop()
        if nid in affected:
            continue
        affected.add(nid)
        stack.extend(children.get(nid, ()))
    return affected

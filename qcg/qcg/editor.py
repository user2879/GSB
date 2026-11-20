"""Deterministic graph edit commands used by the CLI and tests."""
from typing import List

from .errors import QCGError, E_GRAPH_INVALID
from .models import Graph, Node


def _next_id(graph: Graph) -> str:
    nums = [int(n.node_id[1:]) for n in graph.nodes
            if n.node_id.startswith("n") and n.node_id[1:].isdigit()]
    return "n%04d" % (max(nums, default=0) + 1)


def edit_bitwidth(graph: Graph, node_id: str, bitwidth: int) -> None:
    graph.by_id()[node_id].bitwidth = bitwidth


def edit_scale(graph: Graph, node_id: str, scale_permille: int = None,
               scale_float: float = None) -> None:
    node = graph.by_id()[node_id]
    if scale_permille is not None:
        node.scale_permille = int(scale_permille)
    if scale_float is not None:
        node.scale_float = float(scale_float)


def edit_rounding(graph: Graph, node_id: str, rounding: str) -> None:
    graph.by_id()[node_id].rounding = rounding


def add_node(graph: Graph, kind: str, bitwidth: int, scale_permille: int,
             inputs: List[str], kernel: int = 0, stride: int = 0,
             padding: int = 0, node_id: str = None) -> str:
    nid = node_id or _next_id(graph)
    if nid in graph.by_id():
        raise QCGError(E_GRAPH_INVALID, "node id already exists",
                       node_id=nid, field="id")
    graph.nodes.append(Node(node_id=nid, kind=kind, bitwidth=bitwidth,
                            scale_permille=scale_permille,
                            kernel=kernel, stride=stride, padding=padding,
                            inputs=list(inputs)))
    return nid


def remove_node(graph: Graph, node_id: str) -> None:
    if node_id not in graph.by_id():
        raise QCGError(E_GRAPH_INVALID, "node not found",
                       node_id=node_id, field="id")
    graph.nodes = [n for n in graph.nodes if n.node_id != node_id]
    for n in graph.nodes:
        n.inputs = [p for p in n.inputs if p != node_id]


def replace_subgraph(graph: Graph, new_nodes: List[Node]) -> None:
    """Replace the whole node list with an externally built subgraph."""
    graph.nodes = [n.clone() for n in new_nodes]

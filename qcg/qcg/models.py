"""Unified intermediate representation shared by v1 and v2 graphs."""
from dataclasses import dataclass, field, replace
from typing import List, Optional

BITWIDTHS = (4, 8, 16)
ROUNDING_MODES = ("ROUND_HALF_UP", "ROUND_HALF_DOWN", "ROUND_HALF_EVEN",
                  "ROUND_TRUNCATE", "ROUND_FLOOR")
NODE_INPUT = "input"
NODE_CONV = "conv"


@dataclass
class Node:
    node_id: str
    kind: str
    bitwidth: int
    scale_permille: int
    scale_float: float = 0.0
    precision_tag: str = "permille"
    rounding: str = "ROUND_HALF_UP"
    inplace: bool = False
    kernel: int = 0
    stride: int = 0
    padding: int = 0
    inputs: List[str] = field(default_factory=list)
    shape: Optional[List[int]] = None
    boundary_samples: List[List[int]] = field(default_factory=list)

    def clone(self) -> "Node":
        return replace(self, inputs=list(self.inputs),
                       shape=None if self.shape is None else list(self.shape),
                       boundary_samples=[list(b) for b in self.boundary_samples])


@dataclass
class Graph:
    nodes: List[Node]
    source: str = "v2"

    def by_id(self) -> dict:
        return {n.node_id: n for n in self.nodes}

    def clone(self) -> "Graph":
        return Graph(nodes=[n.clone() for n in self.nodes], source=self.source)

    def topo_order(self) -> List[Node]:
        """Stable topological order; call validate() first to guarantee a DAG."""
        ids = [n.node_id for n in self.nodes]
        index = {nid: i for i, nid in enumerate(ids)}
        indeg = {nid: 0 for nid in ids}
        children = {nid: [] for nid in ids}
        for n in self.nodes:
            for parent in n.inputs:
                indeg[n.node_id] += 1
                children[parent].append(n.node_id)
        ready = sorted((nid for nid in ids if indeg[nid] == 0),
                       key=lambda x: index[x])
        order = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for child in sorted(children[nid], key=lambda x: index[x]):
                indeg[child] -= 1
                if indeg[child] == 0:
                    ready.append(child)
            ready.sort(key=lambda x: index[x])
        return [self.by_id()[nid] for nid in order]

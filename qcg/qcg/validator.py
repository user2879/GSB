"""Structural and semantic validation of the unified graph IR."""
from .errors import QCGError, E_GRAPH_INVALID
from .models import Graph, NODE_INPUT, NODE_CONV, BITWIDTHS


def validate(graph: Graph) -> None:
    ids = set()
    for n in graph.nodes:
        if n.node_id in ids:
            raise QCGError(E_GRAPH_INVALID, "duplicate node id",
                           node_id=n.node_id, field="id")
        ids.add(n.node_id)
        if n.kind not in (NODE_INPUT, NODE_CONV):
            raise QCGError(E_GRAPH_INVALID, "unknown node kind",
                           node_id=n.node_id, field="kind")
        if n.bitwidth not in BITWIDTHS:
            raise QCGError(E_GRAPH_INVALID,
                           "bitwidth not in %s" % (BITWIDTHS,),
                           node_id=n.node_id, field="bitwidth")
        if n.scale_permille <= 0:
            raise QCGError(E_GRAPH_INVALID,
                           "scale_permille must be positive",
                           node_id=n.node_id, field="scale_permille")
        if n.kind == NODE_CONV:
            for fname, value in (("kernel", n.kernel), ("stride", n.stride),
                                 ("padding", n.padding)):
                if value < 0:
                    raise QCGError(E_GRAPH_INVALID,
                                   "%s must be non-negative" % fname,
                                   node_id=n.node_id, field=fname)
            if n.kernel == 0:
                raise QCGError(E_GRAPH_INVALID, "kernel must be >= 1",
                               node_id=n.node_id, field="kernel")
            if n.stride == 0:
                raise QCGError(E_GRAPH_INVALID, "stride must be >= 1",
                               node_id=n.node_id, field="stride")
    indeg = {nid: 0 for nid in ids}
    for n in graph.nodes:
        for parent in n.inputs:
            if parent not in ids:
                raise QCGError(E_GRAPH_INVALID,
                               "edge references missing node %r" % parent,
                               node_id=n.node_id, field="inputs")
            if parent == n.node_id:
                raise QCGError(E_GRAPH_INVALID, "self loop",
                               node_id=n.node_id, field="inputs")
            indeg[n.node_id] += 1
    # single-producer rule: every tensor (producer node id) is defined by
    # exactly one node; consumers may fan out, but a tensor cannot be defined
    # twice.
    producers = set()
    for n in graph.nodes:
        if n.node_id in producers:
            raise QCGError(E_GRAPH_INVALID,
                           "tensor has multiple producers",
                           node_id=n.node_id, field="id")
        producers.add(n.node_id)
    # every consumed tensor must resolve to exactly one producer node
    for n in graph.nodes:
        for parent in n.inputs:
            if sum(1 for p in graph.nodes if p.node_id == parent) != 1:
                raise QCGError(E_GRAPH_INVALID,
                               "input tensor lacks a unique producer",
                               node_id=n.node_id, field="inputs")
    # Kahn's algorithm doubles as DAG and stable topo order check.
    order = graph.topo_order()
    if len(order) != len(graph.nodes):
        raise QCGError(E_GRAPH_INVALID, "graph contains a cycle",
                       field="inputs")

"""v2 JSON graph format with named ids, edge lists and full quant metadata."""
import json
import math

from .errors import QCGError, E_GRAPH_INVALID
from .models import Node, Graph, NODE_INPUT, NODE_CONV, BITWIDTHS, ROUNDING_MODES

SCHEMA_VERSION = "qcg-v2"


def parse_v2(text: str) -> Graph:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QCGError(E_GRAPH_INVALID, "invalid JSON: %s" % exc,
                       field="document")
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA_VERSION:
        raise QCGError(E_GRAPH_INVALID,
                       "schema field must equal %r" % SCHEMA_VERSION,
                       field="schema")
    raw_nodes = doc.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise QCGError(E_GRAPH_INVALID, "nodes must be a non-empty list",
                       field="nodes")
    nodes = []
    seen = set()
    for raw in raw_nodes:
        nid = raw.get("id")
        if not isinstance(nid, str) or not nid:
            raise QCGError(E_GRAPH_INVALID, "node id missing", field="id")
        if nid in seen:
            raise QCGError(E_GRAPH_INVALID, "duplicate node id",
                           node_id=nid, field="id")
        seen.add(nid)
        kind = raw.get("kind")
        if kind not in (NODE_INPUT, NODE_CONV):
            raise QCGError(E_GRAPH_INVALID, "kind must be input/conv",
                           node_id=nid, field="kind")
        bw_name = raw.get("bitwidth")
        if not isinstance(bw_name, str) or not bw_name.startswith("INT"):
            raise QCGError(E_GRAPH_INVALID,
                           "bitwidth must be enum name like INT8",
                           node_id=nid, field="bitwidth")
        try:
            bw = int(bw_name[3:])
        except ValueError:
            bw = -1
        if bw not in BITWIDTHS:
            raise QCGError(E_GRAPH_INVALID,
                           "bitwidth enum %r not in %s"
                           % (bw_name, ["INT%d" % b for b in BITWIDTHS]),
                           node_id=nid, field="bitwidth")
        scale = raw.get("scale")
        if not isinstance(scale, (int, float)) or isinstance(scale, bool) \
                or not math.isfinite(float(scale)) or scale <= 0:
            raise QCGError(E_GRAPH_INVALID,
                           "scale must be a positive finite float",
                           node_id=nid, field="scale")
        tag = raw.get("precision_tag")
        if not isinstance(tag, str) or not tag:
            raise QCGError(E_GRAPH_INVALID, "precision_tag missing",
                           node_id=nid, field="precision_tag")
        rounding = raw.get("rounding", "ROUND_HALF_UP")
        if rounding not in ROUNDING_MODES:
            raise QCGError(E_GRAPH_INVALID,
                           "rounding not in %s" % (ROUNDING_MODES,),
                           node_id=nid, field="rounding")
        inplace = bool(raw.get("inplace", False))
        inputs = raw.get("inputs", [])
        if not isinstance(inputs, list) or not all(isinstance(x, str)
                                                   for x in inputs):
            raise QCGError(E_GRAPH_INVALID, "inputs must be a string list",
                           node_id=nid, field="inputs")
        if kind == NODE_INPUT:
            nodes.append(Node(node_id=nid, kind=NODE_INPUT, bitwidth=bw,
                              scale_permille=permille_from_scale(float(scale)),
                              scale_float=float(scale), precision_tag=tag,
                              rounding=rounding, inplace=inplace,
                              inputs=list(inputs),
                              boundary_samples=raw.get("boundary_samples", [])))
            continue
        conv = raw.get("conv", {})
        if not isinstance(conv, dict):
            raise QCGError(E_GRAPH_INVALID, "conv must be an object",
                           node_id=nid, field="conv")
        params = {}
        for name in ("kernel", "stride", "padding"):
            value = conv.get(name)
            if not isinstance(value, int) or isinstance(value, bool) \
                    or value < 0:
                raise QCGError(E_GRAPH_INVALID,
                               "conv.%s must be a non-negative integer" % name,
                               node_id=nid, field="conv.%s" % name)
            params[name] = value
        nodes.append(Node(node_id=nid, kind=NODE_CONV, bitwidth=bw,
                          scale_permille=permille_from_scale(float(scale)),
                          scale_float=float(scale), precision_tag=tag,
                          rounding=rounding, inplace=inplace,
                          kernel=params["kernel"], stride=params["stride"],
                          padding=params["padding"], inputs=list(inputs),
                          boundary_samples=raw.get("boundary_samples", [])))
    return Graph(nodes=nodes, source="v2")


def dump_v2(graph: Graph) -> str:
    doc = {"schema": SCHEMA_VERSION, "nodes": []}
    for n in graph.nodes:
        item = {
            "id": n.node_id,
            "kind": n.kind,
            "bitwidth": "INT%d" % n.bitwidth,
            "scale": (n.scale_float if n.scale_float > 0
                      else scale_from_permille(n.scale_permille)),
            "precision_tag": n.precision_tag,
            "rounding": n.rounding,
            "inplace": n.inplace,
            "inputs": list(n.inputs),
        }
        if n.boundary_samples:
            item["boundary_samples"] = [list(b) for b in n.boundary_samples]
        if n.kind == NODE_CONV:
            item["conv"] = {"kernel": n.kernel, "stride": n.stride,
                            "padding": n.padding}
        doc["nodes"].append(item)
    return json.dumps(doc, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def scale_from_permille(permille: int) -> float:
    return permille / 1000.0


def permille_from_scale(scale: float) -> int:
    return int(round(scale * 1000.0))

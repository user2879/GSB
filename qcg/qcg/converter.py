"""Cross-version conversion with sidecar metadata for lossy downgrade."""
import hashlib
import json
import os
from typing import List, Tuple

from .errors import FaultRecord, E_LOSSY_DOWNGRADE
from .models import Graph, Node
from .v2_codec import scale_from_permille, permille_from_scale

SIDECAR_VERSION = "qcg-sidecar-1"
DEFAULT_ROUNDING = "ROUND_HALF_UP"


def upgrade_v1_to_v2(graph: Graph, sidecar: dict = None) -> Tuple[Graph, List[FaultRecord]]:
    """v1 IR -> v2 IR. Sidecar entries restore rounding/inplace/residual."""
    sidecar = sidecar or {}
    entries = {e["node_id"]: e for e in sidecar.get("nodes", [])}
    losses: List[FaultRecord] = []
    out = graph.clone()
    out.source = "v2"
    for node in out.nodes:
        node.scale_permille = int(node.scale_permille)
        node.scale_float = scale_from_permille(node.scale_permille)
        entry = entries.get(node.node_id)
        if entry is not None:
            if "original_scale" in entry:
                node.scale_float = float(entry["original_scale"])
            node.precision_tag = entry.get("precision_tag", node.precision_tag)
            node.rounding = entry.get("rounding", node.rounding)
            node.inplace = bool(entry.get("inplace", False))
            node.boundary_samples = [list(b)
                                     for b in entry.get("boundary_samples", [])]
            for unrecoverable in entry.get("unrecoverable", []):
                losses.append(FaultRecord(
                    code=E_LOSSY_DOWNGRADE,
                    message="downgrade loss cannot be restored: %s"
                            % unrecoverable,
                    node_id=node.node_id, field=unrecoverable))
        else:
            node.precision_tag = "permille-exact"
    return out, losses


def downgrade_v2_to_v1(graph: Graph) -> Tuple[Graph, dict, List[FaultRecord]]:
    """v2 IR -> v1 IR plus sidecar; every non-expressible fact is recorded."""
    out = graph.clone()
    out.source = "v1"
    entries = []
    losses: List[FaultRecord] = []
    for node in out.nodes:
        scale = _node_scale(node)
        rounded = permille_from_scale(scale)
        residual = scale - rounded / 1000.0
        unrecoverable = []
        if abs(residual) > 1e-12:
            unrecoverable.append("scale_residual")
        if node.precision_tag != "permille-exact":
            # the tag itself survives in the sidecar, only v1 cannot show it
            pass
        entry = {
            "node_id": node.node_id,
            "original_scale": scale,
            "scale_permille": rounded,
            "truncation_residual": residual,
            "precision_tag": node.precision_tag,
            "rounding": node.rounding,
            "inplace": bool(node.inplace),
            "bitwidth_enum": "INT%d" % node.bitwidth,
            "boundary_samples": [list(b) for b in node.boundary_samples],
            "unrecoverable": unrecoverable,
        }
        entries.append(entry)
        node.scale_permille = rounded
        node.scale_float = scale
        node.precision_tag = "permille-exact"
        if node.rounding != DEFAULT_ROUNDING:
            losses.append(FaultRecord(
                code=E_LOSSY_DOWNGRADE,
                message="rounding mode %r not expressible in v1; parked in "
                        "sidecar" % node.rounding,
                node_id=node.node_id, field="rounding"))
        if node.inplace:
            losses.append(FaultRecord(
                code=E_LOSSY_DOWNGRADE,
                message="inplace writeback not expressible in v1; parked in "
                        "sidecar",
                node_id=node.node_id, field="inplace"))
        if abs(residual) > 1e-12:
            losses.append(FaultRecord(
                code=E_LOSSY_DOWNGRADE,
                message="scale residual %.12g lost in permille encoding"
                        % residual,
                node_id=node.node_id, field="scale_permille"))
    sidecar = {
        "version": SIDECAR_VERSION,
        "source_graph_hash": graph_fingerprint(graph),
        "nodes": entries,
    }
    return out, sidecar, losses


def _node_scale(node: Node) -> float:
    if node.scale_float > 0:
        return node.scale_float
    return scale_from_permille(node.scale_permille)


def graph_fingerprint(graph: Graph) -> str:
    payload = json.dumps([n.__dict__ for n in graph.nodes],
                         sort_keys=True, default=list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_sidecar(path: str, sidecar: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def read_sidecar(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("version") != SIDECAR_VERSION:
        raise ValueError("unsupported sidecar version")
    return doc


def hash_sidecar(sidecar: dict) -> str:
    payload = json.dumps(sidecar, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

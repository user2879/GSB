"""Generate byte-reproducible local samples with a fixed seed."""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcg.models import Graph, Node, NODE_INPUT, NODE_CONV
from qcg.v1_codec import dump_v1
from qcg.v2_codec import dump_v2
from qcg.quant_pack import generate_sample_blocks, write_pack, write_shapes

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def build_graph() -> Graph:
    nodes = [
        Node(node_id="n0000", kind=NODE_INPUT, bitwidth=8,
             scale_permille=250, scale_float=0.25,
             precision_tag="permille-exact"),
        Node(node_id="n0001", kind=NODE_CONV, bitwidth=8,
             scale_permille=500, scale_float=0.5,
             precision_tag="permille-exact", rounding="ROUND_HALF_EVEN",
             inplace=True, kernel=3, stride=1, padding=1,
             inputs=["n0000"]),
        Node(node_id="n0002", kind=NODE_CONV, bitwidth=8,
             scale_permille=125, scale_float=0.125,
             precision_tag="permille-exact", kernel=2, stride=2, padding=0,
             inputs=["n0001"]),
        Node(node_id="n0003", kind=NODE_CONV, bitwidth=4,
             scale_permille=1000, scale_float=1.0,
             precision_tag="permille-exact", rounding="ROUND_HALF_DOWN",
             kernel=3, stride=1, padding=1, inputs=["n0002"]),
        Node(node_id="n0004", kind=NODE_CONV, bitwidth=16,
             scale_permille=333, scale_float=0.333,
             precision_tag="float-3digit", kernel=2, stride=1, padding=0,
             inputs=["n0003"]),
    ]
    return Graph(nodes=nodes, source="v2")


def main() -> None:
    graph = build_graph()
    v2_path = os.path.join(DATA, "graphs_v2", "sample_graph.v2.json")
    v1_path = os.path.join(DATA, "graphs_v1", "sample_graph.v1.txt")
    pack_path = os.path.join(DATA, "quant_pack", "sample_pack.le.bin")
    shape_path = os.path.join(DATA, "quant_pack", "sample_shapes.json")
    os.makedirs(os.path.dirname(v2_path), exist_ok=True)
    with open(v2_path, "w", encoding="utf-8") as fh:
        fh.write(dump_v2(graph))
    v1_graph = graph.clone()
    v1_graph.source = "v1"
    with open(v1_path, "w", encoding="utf-8") as fh:
        fh.write(dump_v1(v1_graph))
    blocks = generate_sample_blocks()
    write_pack(pack_path, blocks)
    write_shapes(shape_path, blocks)
    checksum = {}
    for path in (v1_path, v2_path, pack_path, shape_path):
        with open(path, "rb") as fh:
            checksum[os.path.relpath(path, os.path.join(DATA, ".."))] = \
                hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(DATA, "quant_pack", "sample_checksums.json"),
              "w", encoding="utf-8") as fh:
        json.dump(checksum, fh, indent=2, sort_keys=True)
    print(json.dumps(checksum, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

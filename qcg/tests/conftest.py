import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "..", "data")
sys.path.insert(0, ROOT)

from qcg.config import Config
from qcg.models import Graph, Node, NODE_INPUT, NODE_CONV
from qcg.quant_pack import generate_sample_blocks


@pytest.fixture
def sample_graph() -> Graph:
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


@pytest.fixture
def blocks():
    return generate_sample_blocks()


@pytest.fixture
def cfg() -> Config:
    return Config(max_workers=1, global_mac_budget=10 ** 9,
                  node_mac_quota=10 ** 9, cache_resident_bytes=10 ** 9,
                  batch_recompute_budget=10 ** 9)


@pytest.fixture
def tiny_cfg() -> Config:
    return Config(max_workers=1, global_mac_budget=300,
                  node_mac_quota=10 ** 9, cache_resident_bytes=10 ** 9,
                  batch_recompute_budget=200)

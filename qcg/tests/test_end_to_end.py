import json
import os
import subprocess
import sys

import numpy as np
import pytest

from qcg.config import Config
from qcg.converter import downgrade_v2_to_v1, upgrade_v1_to_v2
from qcg.editor import (add_node, edit_bitwidth, edit_rounding, edit_scale,
                        remove_node)
from qcg.engine import Engine
from qcg.quant_pack import write_pack
from qcg.v1_codec import parse_v1, dump_v1
from qcg.validator import validate


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pack(tmp_path, blocks):
    path = os.path.join(tmp_path, "pack.bin")
    write_pack(path, blocks)
    return path


def test_full_version_switch_pipeline(tmp_path, sample_graph, blocks, cfg):
    pack = _pack(tmp_path, blocks)
    v1_graph, sidecar, losses = downgrade_v2_to_v1(sample_graph)
    validate(v1_graph)
    v1_text = dump_v1(v1_graph)
    sidecar_path = os.path.join(tmp_path, "sidecar.json")
    from qcg.converter import write_sidecar
    write_sidecar(sidecar_path, sidecar)
    restored, _ = upgrade_v1_to_v2(parse_v1(v1_text), sidecar)
    # run the restored graph and compare against original
    root_a = os.path.join(tmp_path, "a")
    root_b = os.path.join(tmp_path, "b")
    Engine(cfg, root_a).run(sample_graph.clone(), pack)
    Engine(cfg, root_b).run(restored, pack)
    from tests.test_engine import _product_dump
    dump_a, dump_b = _product_dump(root_a), _product_dump(root_b)
    assert set(dump_a) == set(dump_b)
    for nid in dump_a:
        np.testing.assert_allclose(np.array(dump_a[nid]["dq"]),
                                   np.array(dump_b[nid]["dq"]))


def test_boundary_marker_survives_version_switch(cfg):
    from qcg.dequant import dequantize
    from qcg.models import Node
    acc = np.array([1, 2], dtype=np.int64)
    # total scale 1/512 with frac_bits 8 => fixed value lands on x.5
    node = Node("n0001", "conv", 8, 1000, 1.0,
                rounding="ROUND_HALF_UP", kernel=1, stride=1)
    first = dequantize(acc, node, 1.0 / 512, cfg)
    assert first.boundary_positions  # some midpoints exist
    v1_graph, sidecar, _ = downgrade_v2_to_v1(
        __import__("qcg.models", fromlist=["Graph"]).Graph(nodes=[node],
                                                            source="v2"))
    restored, _ = upgrade_v1_to_v2(parse_v1(dump_v1(v1_graph)), sidecar)
    again = dequantize(acc, restored.by_id()["n0001"], 1.0 / 512, cfg)
    assert again.boundary_positions == first.boundary_positions


def test_arbitrary_edit_sequence_matches_rebuild(tmp_path, sample_graph,
                                                 blocks, cfg):
    pack = _pack(tmp_path, blocks)
    inc_root = os.path.join(tmp_path, "inc")
    engine = Engine(cfg, inc_root)
    previous = sample_graph.clone()
    engine.run(previous.clone(), pack)
    edits = [
        lambda g: edit_bitwidth(g, "n0002", 16),
        lambda g: edit_scale(g, "n0001", scale_permille=800,
                             scale_float=0.8),
        lambda g: edit_rounding(g, "n0003", "ROUND_TRUNCATE"),
        lambda g: remove_node(g, "n0004"),
    ]
    current = previous
    for mutate in edits:
        current = current.clone()
        mutate(current)
        validate(current)
        engine.run(current.clone(), pack, previous_graph=previous)
        previous = current
    full_root = os.path.join(tmp_path, "full")
    Engine(cfg, full_root).run(current.clone(), pack)
    from tests.test_engine import _product_dump
    inc, full = _product_dump(inc_root), _product_dump(full_root)
    assert set(inc) == set(full)
    for nid, item in full.items():
        np.testing.assert_array_equal(np.array(inc[nid]["int"]),
                                      np.array(item["int"]))
        np.testing.assert_allclose(np.array(inc[nid]["dq"]),
                                   np.array(item["dq"]))
        assert inc[nid]["key"] == item["key"]


def test_cli_convert_and_run(tmp_path):
    data = os.path.join(ROOT, "..", "data")
    out_v1 = os.path.join(tmp_path, "out.v1.txt")
    sidecar = os.path.join(tmp_path, "sidecar.json")
    cmd = [sys.executable, "-m", "qcg.cli", "convert",
           "--input", os.path.join(data, "graphs_v2", "sample_graph.v2.json"),
           "--output", out_v1, "--target", "v1", "--sidecar", sidecar]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    assert os.path.exists(out_v1) and os.path.exists(sidecar)
    back_v2 = os.path.join(tmp_path, "back.v2.json")
    subprocess.run([sys.executable, "-m", "qcg.cli", "convert",
                    "--input", out_v1, "--output", back_v2, "--target", "v2",
                    "--sidecar", sidecar],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    report = os.path.join(tmp_path, "report.json")
    snap = os.path.join(tmp_path, "snap")
    subprocess.run([sys.executable, "-m", "qcg.cli", "run",
                    "--graph", back_v2,
                    "--config", os.path.join(data, "scheduler_config.yaml"),
                    "--pack", os.path.join(data, "quant_pack",
                                           "sample_pack.le.bin"),
                    "--snapshot-root", snap, "--report", report],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    doc = json.load(open(report, encoding="utf-8"))
    assert doc["topo_order"] == ["n0000", "n0001", "n0002", "n0003", "n0004"]
    assert len(doc["products"]) == 5


def test_reproducible_sample_bytes():
    data = os.path.join(ROOT, "..", "data", "quant_pack",
                        "sample_checksums.json")
    doc = json.load(open(data, encoding="utf-8"))
    for rel, expected in doc.items():
        path = os.path.join(ROOT, "..", rel)
        import hashlib
        with open(path, "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == expected

import os

from qcg.converter import (downgrade_v2_to_v1, upgrade_v1_to_v2,
                           write_sidecar, read_sidecar)
from qcg.v1_codec import parse_v1, dump_v1
from qcg.v2_codec import parse_v2, dump_v2
from qcg.validator import validate
from qcg.errors import E_LOSSY_DOWNGRADE


def test_roundtrip_restores_rounding_inplace_and_scale(sample_graph):
    v1_graph, sidecar, losses = downgrade_v2_to_v1(sample_graph)
    assert v1_graph.source == "v1"
    fields = {e["node_id"]: e for e in sidecar["nodes"]}
    assert fields["n0001"]["rounding"] == "ROUND_HALF_EVEN"
    assert fields["n0001"]["inplace"] is True
    assert fields["n0004"]["truncation_residual"] == \
        0.333 - round(0.333 * 1000) / 1000
    loss_nodes = {l.node_id for l in losses if l.code == E_LOSSY_DOWNGRADE}
    assert "n0001" in loss_nodes  # rounding + inplace parked
    # v1 text cannot express rounding/inplace; fields are fixed width
    text = dump_v1(v1_graph)
    reparsed_v1 = parse_v1(text)
    restored, restore_losses = upgrade_v1_to_v2(reparsed_v1, sidecar)
    validate(restored)
    r1 = restored.by_id()["n0001"]
    assert r1.rounding == "ROUND_HALF_EVEN"
    assert r1.inplace is True
    r4 = restored.by_id()["n0004"]
    assert r4.scale_float == 0.333
    assert r4.precision_tag == "float-3digit"
    assert restore_losses == []


def test_permille_exact_scale_conversion(sample_graph):
    v1_graph, sidecar, _ = downgrade_v2_to_v1(sample_graph)
    n1 = v1_graph.by_id()["n0001"]
    assert n1.scale_permille == 500
    restored, _ = upgrade_v1_to_v2(n1 and parse_v1(dump_v1(v1_graph)),
                                   sidecar)
    assert restored.by_id()["n0000"].scale_float == 0.25


def test_sidecar_file_roundtrip(tmp_path, sample_graph):
    _, sidecar, _ = downgrade_v2_to_v1(sample_graph)
    path = os.path.join(tmp_path, "sidecar.json")
    write_sidecar(path, sidecar)
    loaded = read_sidecar(path)
    assert loaded["nodes"] == sidecar["nodes"]


def test_unrecoverable_residual_recorded(sample_graph):
    graph = sample_graph.clone()
    graph.by_id()["n0002"].scale_float = 0.123456
    graph.by_id()["n0002"].scale_permille = 123
    v1_graph, sidecar, losses = downgrade_v2_to_v1(graph)
    entry = next(e for e in sidecar["nodes"] if e["node_id"] == "n0002")
    assert "scale_residual" in entry["unrecoverable"]
    assert any(l.code == E_LOSSY_DOWNGRADE and l.node_id == "n0002"
               for l in losses)

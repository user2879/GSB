import pytest

from qcg.errors import QCGError
from qcg.models import NODE_INPUT, NODE_CONV
from qcg.v1_codec import parse_v1, dump_v1
from qcg.v2_codec import parse_v2, dump_v2
from qcg.validator import validate


V1_TEXT = (
    "QCGV1 3\n"
    "N000008+002500000000    \n"
    "C000108+0050001010310000\n"
    "C000204+0100002000210001\n"
)


def test_v1_fixed_width_roundtrip(sample_graph):
    text = dump_v1(sample_graph)
    for line in text.splitlines()[1:]:
        assert len(line) == 24
    parsed = parse_v1(text)
    assert [n.node_id for n in parsed.nodes] == \
           [n.node_id for n in sample_graph.nodes]
    n1 = parsed.by_id()["n0001"]
    assert (n1.kind, n1.bitwidth, n1.scale_permille, n1.kernel,
            n1.stride, n1.padding, n1.inputs) == \
           (NODE_CONV, 8, 500, 3, 1, 1, ["n0000"])
    assert parsed.by_id()["n0000"].kind == NODE_INPUT


def test_v1_bad_width_rejected():
    bad = "QCGV1 1\nN000008+00250000000\n"
    with pytest.raises(QCGError) as exc:
        parse_v1(bad)
    assert exc.value.code == "E_GRAPH_INVALID"
    assert exc.value.field == "line"


def test_v1_duplicate_id():
    text = ("QCGV1 2\n"
            "N000008+002500000000    \n"
            "N000008+002500000000    \n")
    with pytest.raises(QCGError) as exc:
        parse_v1(text)
    assert exc.value.node_id == "n0000"


def test_v2_parse_keeps_float_scale_and_tag(sample_graph):
    text = dump_v2(sample_graph)
    parsed = parse_v2(text)
    n4 = parsed.by_id()["n0004"]
    assert n4.scale_float == pytest.approx(0.333)
    assert n4.precision_tag == "float-3digit"
    assert parsed.by_id()["n0001"].inplace is True
    assert parsed.by_id()["n0001"].rounding == "ROUND_HALF_EVEN"


def test_v2_bad_enum_and_scale():
    doc = ('{"schema": "qcg-v2", "nodes": [{"id": "x", "kind": "input",'
           ' "bitwidth": "INT3", "scale": 0.5, "precision_tag": "t"}]}')
    with pytest.raises(QCGError) as exc:
        parse_v2(doc)
    assert exc.value.field == "bitwidth"
    doc = ('{"schema": "qcg-v2", "nodes": [{"id": "x", "kind": "input",'
           ' "bitwidth": "INT8", "scale": 0, "precision_tag": "t"}]}')
    with pytest.raises(QCGError) as exc:
        parse_v2(doc)
    assert exc.value.field == "scale"


def test_validation_cycle_and_missing_edge():
    text = ("QCGV1 2\n"
            "C000108+0050001010310001\n"
            "C000208+0050001010310002\n")
    graph = parse_v1(text)
    with pytest.raises(QCGError) as exc:
        validate(graph)
    assert exc.value.code == "E_GRAPH_INVALID"


def test_topo_order_stable(sample_graph):
    validate(sample_graph)
    order = [n.node_id for n in sample_graph.topo_order()]
    assert order == ["n0000", "n0001", "n0002", "n0003", "n0004"]

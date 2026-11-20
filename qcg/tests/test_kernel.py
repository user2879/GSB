import numpy as np
import pytest

from qcg.config import Config
from qcg.errors import QCGError
from qcg.kernel import integer_conv, conv_output_shape, TRAVERSAL_ID
from qcg.models import Node


def _conv_node(**kw):
    defaults = dict(node_id="c1", kind="conv", bitwidth=8,
                    scale_permille=1000, scale_float=1.0,
                    kernel=2, stride=1, padding=0, inputs=[])
    defaults.update(kw)
    return Node(**defaults)


def test_known_no_padding_convolution(cfg):
    activation = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]], dtype=np.int8)
    weight = np.array([[1, 0],
                       [0, 1]], dtype=np.int8)
    node = _conv_node(kernel=2, stride=1, padding=0)
    result = integer_conv(activation, weight, node, cfg)
    expected = np.array([[6, 8], [12, 14]], dtype=np.int64)
    np.testing.assert_array_equal(result.output, expected)
    assert result.macs == 16
    assert result.traversal == TRAVERSAL_ID
    assert result.acc_bits == cfg.accumulator_bits


def test_padding_and_stride_shape(cfg):
    node = _conv_node(kernel=3, stride=2, padding=1)
    out = conv_output_shape(8, 8, node)
    assert out == (4, 4)


def test_deterministic_repeat(cfg, blocks):
    node = _conv_node(kernel=3, stride=1, padding=1)
    a = integer_conv(blocks["act:n0000"], blocks["w:n0001"], node, cfg)
    b = integer_conv(blocks["act:n0000"], blocks["w:n0001"], node, cfg)
    np.testing.assert_array_equal(a.output, b.output)


def test_overflow_error_policy(blocks):
    strict = Config(accumulator_bits=8, overflow_policy="ERROR")
    big_weight = np.full((3, 3), 120, dtype=np.int8)
    node = _conv_node(kernel=3, stride=1, padding=1)
    with pytest.raises(QCGError) as exc:
        integer_conv(blocks["act:n0000"], big_weight, node, strict)
    assert exc.value.code == "E_ACC_OVERFLOW"
    assert exc.value.node_id == "c1"
    assert len(exc.value.message) > 0


def test_saturate_and_wrap_policies(blocks):
    node = _conv_node(kernel=3, stride=1, padding=1)
    big_weight = np.full((3, 3), 120, dtype=np.int8)
    sat = Config(accumulator_bits=8, overflow_policy="SATURATE")
    wrap = Config(accumulator_bits=8, overflow_policy="WRAP")
    r_sat = integer_conv(blocks["act:n0000"], big_weight, node, sat)
    r_wrap = integer_conv(blocks["act:n0000"], big_weight, node, wrap)
    assert r_sat.overflows and r_wrap.overflows
    assert np.all(r_sat.output >= -128) and np.all(r_sat.output <= 127)
    assert not np.allclose(r_sat.output, r_wrap.output)

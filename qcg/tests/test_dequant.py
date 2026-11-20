import numpy as np

from qcg.config import Config
from qcg.dequant import dequantize
from qcg.models import Node


def _node(rounding="ROUND_HALF_UP", scale=1.0):
    return Node(node_id="c1", kind="conv", bitwidth=8,
                scale_permille=int(round(scale * 1000)),
                scale_float=scale, rounding=rounding)


def test_half_up_vs_half_down_midpoint(cfg):
    # acc=1, weight*act scale = 1/512 => fixed value is exactly x.5
    acc = np.array([1], dtype=np.int64)
    up = dequantize(acc, _node("ROUND_HALF_UP", 1.0 / 512), 1.0, cfg)
    down = dequantize(acc, _node("ROUND_HALF_DOWN", 1.0 / 512), 1.0, cfg)
    assert up.boundary_positions == [[0]]
    assert down.boundary_positions == [[0]]
    assert up.values[0] > down.values[0]
    assert up.values[0] - down.values[0] == 1.0 / (2 ** cfg.frac_bits)


def test_half_even_tie_to_even(cfg):
    acc = np.array([1, 3], dtype=np.int64)
    even = dequantize(acc, _node("ROUND_HALF_EVEN", 1.0 / 512), 1.0, cfg)
    # fixed values 0.5 and 1.5 => round to 0 and 2 respectively
    assert even.values.tolist() == [0.0, 2.0 / (2 ** cfg.frac_bits)]
    assert sorted(map(tuple, even.boundary_positions)) == [(0,), (1,)]


def test_truncate_and_floor(cfg):
    acc = np.array([1, -1], dtype=np.int64)
    trunc = dequantize(acc, _node("ROUND_TRUNCATE", 1.0 / 1024), 1.0, cfg)
    floor = dequantize(acc, _node("ROUND_FLOOR", 1.0 / 1024), 1.0, cfg)
    # fixed values are +/-0.25
    assert trunc.values.tolist() == [0.0, 0.0]
    assert floor.values[1] == -1.0 / (2 ** cfg.frac_bits)


def test_frac_bits_change_output(cfg):
    acc = np.array([7], dtype=np.int64)
    a = dequantize(acc, _node(scale=0.01), 1.0, cfg)
    cfg2 = Config(frac_bits=4, global_mac_budget=cfg.global_mac_budget,
                  node_mac_quota=cfg.node_mac_quota,
                  cache_resident_bytes=cfg.cache_resident_bytes,
                  batch_recompute_budget=cfg.batch_recompute_budget)
    b = dequantize(acc, _node(scale=0.01), 1.0, cfg2)
    assert not np.allclose(a.values, b.values)


def test_clipping_to_output_range(cfg):
    acc = np.array([10 ** 6], dtype=np.int64)
    out = dequantize(acc, _node(scale=1.0), 1.0, cfg)
    assert out.values[0] == cfg.output_max


def test_metadata_records_rounding_and_frac(cfg):
    acc = np.array([2], dtype=np.int64)
    out = dequantize(acc, _node("ROUND_HALF_EVEN"), 1.0, cfg)
    assert out.rounding == "ROUND_HALF_EVEN"
    assert out.frac_bits == cfg.frac_bits

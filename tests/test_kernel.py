import numpy as np

from fablock.config import OperatorConfig
from fablock.fixedpoint import fp_exp, quantize
from fablock.kernel import compute_score_block, online_softmax
from fablock.reference import split_blocks


def _blocks(cfg, dataset):
    tensors = dataset.tensors_for_batch(0)
    frac, level = cfg.fixed_point_frac_bits, cfg.quantization_level_bits
    q = split_blocks(quantize(tensors["Q"], frac, level), cfg.query_block_len)
    k = split_blocks(quantize(tensors["K"], frac, level), cfg.key_block_len)
    v = split_blocks(quantize(tensors["V"], frac, level), cfg.key_block_len)
    return q, k, v


def test_kernel_deterministic(cfg, dataset):
    q, k, v = _blocks(cfg, dataset)
    traces = []
    for _ in range(2):
        scores = [compute_score_block(q[0], kb, cfg, 0, ki)
                  for ki, kb in enumerate(k)]
        traces.append(online_softmax(scores, v, cfg, 0))
    assert traces[0].out_fp == traces[1].out_fp
    for s0, s1 in zip(traces[0].states, traces[1].states):
        assert s0.m_fp == s1.m_fp and s0.l_fp == s1.l_fp
        assert s0.out_acc == s1.out_acc


def test_frac_bits_change_integer_intermediates(cfg, dataset):
    q8, k8, v8 = _blocks(cfg.with_overrides(fixed_point_frac_bits=8), dataset)
    q14, k14, v14 = _blocks(cfg.with_overrides(fixed_point_frac_bits=14), dataset)
    s8 = compute_score_block(q8[0], k8[0],
                             cfg.with_overrides(fixed_point_frac_bits=8), 0, 0)
    s14 = compute_score_block(q14[0], k14[0],
                              cfg.with_overrides(fixed_point_frac_bits=14), 0, 0)
    assert s8.scores != s14.scores
    t8 = online_softmax([s8], [v8[0]],
                        cfg.with_overrides(fixed_point_frac_bits=8), 0)
    t14 = online_softmax([s14], [v14[0]],
                         cfg.with_overrides(fixed_point_frac_bits=14), 0)
    assert t8.out_fp != t14.out_fp


def test_quant_level_changes_intermediates(cfg, dataset):
    tensors = dataset.tensors_for_batch(0).copy()
    tensors["Q"] = tensors["Q"].copy()
    tensors["Q"][0, 0] = 3.0  # 超出 int8 档位在 frac=6 下的表示范围
    q8 = quantize(tensors["Q"], 6, 8)
    q16 = quantize(tensors["Q"], 6, 16)
    assert not np.array_equal(q8, q16)  # int8 档位截断真实改变量化整数


def test_accumulator_saturation_records_overflow(cfg, dataset):
    big = cfg.with_overrides(fixed_point_frac_bits=20,
                             accumulator_int_bits=32)
    q, k, v = _blocks(big, dataset)
    result = compute_score_block(q[0], k[0], big, 0, 0)
    assert result.overflows, "应记录溢出事件"
    ev = result.overflows[0]
    assert ev.query_block == 0 and ev.key_block == 0
    assert 0 <= ev.dim < big.head_dim  # 累加位置（键维度升序）
    limit = 1 << 31
    assert all(-limit <= s < limit for row in result.scores for s in row)


def test_fp_exp_deterministic_and_accurate():
    for x in (-3.0, -1.5, -0.25, -0.001):
        x_fp = int(round(x * (1 << 20)))
        got = fp_exp(x_fp, 20) / (1 << 20)
        assert abs(got - np.exp(x)) < 1e-4
        assert fp_exp(x_fp, 20) == fp_exp(x_fp, 20)

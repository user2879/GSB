import numpy as np

from peft_fp.fixedpoint import (
    Ledger,
    dequant,
    matmul_fixed,
    mul_fixed,
    quantize,
    round_div,
    round_shift,
)


def test_round_shift_half_away_from_zero():
    raw = np.array([3, -3, 2, -2, 5, -5], dtype=np.int64)
    res = round_shift(raw, 1)
    assert res.tolist() == [2, -2, 1, -1, 3, -3]


def test_quantize_dequant_roundtrip():
    x = np.array([0.25, -0.5, 1.0 / 16, -2.0])
    q = quantize(x, 4)
    assert np.allclose(dequant(q, 4), x)


def test_quantize_ledger_records_direction():
    ledger = Ledger()
    quantize(np.array([0.1, -0.1, 0.5]), 4, ledger, "op")
    entry = ledger.entries[0]
    assert entry["op"] == "op"
    assert entry["count"] == 3
    assert entry["round_up"] + entry["round_down"] + entry["exact"] == 3
    assert 0.0 <= entry["max_abs_residual_quanta"] <= 0.5


def test_matmul_fixed_deterministic_and_exact():
    a = quantize(np.array([[0.5, 0.25], [1.0, -0.5]]), 8)
    b = quantize(np.array([[1.0, 2.0], [3.0, 4.0]]), 8)
    r1 = matmul_fixed(a, b, 8)
    r2 = matmul_fixed(a, b, 8)
    assert np.array_equal(r1, r2)
    expect = np.array([[0.5, 0.25], [1.0, -0.5]]) @ np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(dequant(r1, 8), expect)


def test_round_div_half_away():
    res = round_div(np.array([5, -5, 3, -3, 7], dtype=np.int64), 2)
    assert res.tolist() == [3, -3, 2, -2, 4]


def test_mul_fixed():
    a = quantize(np.float64(1.5), 8)
    b = quantize(np.float64(2.0), 8)
    assert dequant(mul_fixed(a, b, 8), 8) == 3.0

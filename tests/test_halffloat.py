import numpy as np

from hdr_sdr_workbench import halffloat as hf


def test_scalar_matches_vectorized_and_numpy():
    rng = np.random.default_rng(7)
    vals = (rng.random(200000, dtype=np.float32) * 131072.0 - 100.0)
    vals = np.concatenate([
        vals,
        np.geomspace(1e-10, 1e-3, 50000).astype(np.float32),
        np.array([0.0, -0.0, 65504.0, 65505.0, 1e9, -1e9,
                  float("inf"), float("-inf"), float("nan"),
                  5.9604645e-8, 2.9802322e-8], dtype=np.float32),
    ])
    vec = hf.float32_to_half_bytes(vals)
    scalar = np.array([hf.encode_half_scalar(float(v)) for v in vals], dtype=np.int64)
    assert np.array_equal(vec.astype(np.int64), scalar)

    finite = vals[np.isfinite(vals) & (np.abs(vals) <= 65504.0)]
    assert np.array_equal(hf.float32_to_half_bytes(finite),
                          finite.astype(np.float16).view(np.uint16))


def test_decode_roundtrip_exact():
    bits = np.arange(0, 0x10000, dtype=np.uint16)
    decoded = hf.half_bytes_to_float32(bits)
    scalar = np.array([hf.decode_half_scalar(int(b)) for b in bits], dtype=np.float32)
    assert np.array_equal(decoded.view(np.uint32), scalar.view(np.uint32))


def test_inf_nan_policy():
    inf_bits = np.array([0x7C00, 0xFC00], dtype=np.uint16)
    out = hf.half_bytes_to_float32(inf_bits)
    assert out[0] == np.float32(65504.0) and out[1] == np.float32(-65504.0)
    nan_bits = np.array([0x7E00, 0x7FFF], dtype=np.uint16)
    assert np.all(hf.half_bytes_to_float32(nan_bits) == np.float32(0.0))
    assert hf.float32_to_half_bytes(np.array([float("nan")], dtype=np.float32))[0] == 0
    assert hf.float32_to_half_bytes(np.array([1e9], dtype=np.float32))[0] == 0x7BFF

import numpy as np

from hdr_sdr_workbench.errors import E_TONEMAP_PARAM, WorkbenchError
from hdr_sdr_workbench.pipeline import (M_2020_TO_709, run_pipeline,
                                        validate_state_params)


def _sample():
    rng = np.random.default_rng(3)
    x = rng.random((24, 32, 3), dtype=np.float32) * np.float32(12.0)
    return x


def test_deterministic_and_byte_identical():
    x = _sample()
    a, _ = run_pipeline(x, "bt709", 0.75, "aces", 1.0, "srgb")
    b, _ = run_pipeline(x, "bt709", 0.75, "aces", 1.0, "srgb")
    assert a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert a.tobytes() == b.tobytes()


def test_tonemap_families_and_oetf_distinct():
    x = _sample()
    reinhard, _ = run_pipeline(x, "bt2020", 0.0, "reinhard", 4.0, "bt709")
    aces, _ = run_pipeline(x, "bt2020", 0.0, "aces", 1.0, "bt709")
    assert not np.array_equal(reinhard, aces)
    srgb, _ = run_pipeline(x, "bt709", 0.0, "reinhard", 4.0, "srgb")
    bt709, _ = run_pipeline(x, "bt709", 0.0, "reinhard", 4.0, "bt709")
    assert not np.array_equal(srgb, bt709)


def test_saturation_and_quantization_range():
    x = np.full((8, 8, 3), 1e6, dtype=np.float32)
    u8, mapped = run_pipeline(x, "bt2020", 0.0, "reinhard", 1.0, "bt709")
    assert u8.max() == 255 and u8.min() >= 0
    assert mapped.max() <= 1.0 and mapped.min() >= 0.0


def test_matrix_constants_fixed():
    # 矩阵为固定 float32 常量，行和约等于 1（白点保持）
    assert M_2020_TO_709.dtype == np.float32
    ones = np.ones(3, dtype=np.float32)
    white = M_2020_TO_709 @ ones
    assert np.allclose(white, ones, atol=2e-6)


def test_invalid_params():
    for kwargs in (
        dict(colorspace="xyz", exposure_stops=0, tonemap="aces",
             tonemap_param=1, oetf="srgb"),
        dict(colorspace="bt709", exposure_stops=0, tonemap="filmic",
             tonemap_param=1, oetf="srgb"),
        dict(colorspace="bt709", exposure_stops=0, tonemap="reinhard",
             tonemap_param=0, oetf="srgb"),
        dict(colorspace="bt709", exposure_stops=99, tonemap="aces",
             tonemap_param=1, oetf="srgb"),
    ):
        try:
            validate_state_params(**kwargs)
            assert False, f"应拒绝 {kwargs}"
        except WorkbenchError as exc:
            assert exc.code == E_TONEMAP_PARAM

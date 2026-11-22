from __future__ import annotations

import numpy as np
import pytest

from qcolumn_diag.calibration import linear_quantile


@pytest.mark.parametrize("seed", [0, 7, 20261123])
def test_linear_quantile_matches_numpy_reference(seed: int) -> None:
    rng = np.random.Generator(np.random.PCG64(seed))
    values = rng.standard_normal(37).astype(np.float64)
    values = np.sort(values, kind="stable")
    for permille in (0, 1, 50, 500, 950, 999, 1000):
        expected = float(
            np.quantile(values, permille / 1000.0, method="linear")
        )
        assert linear_quantile(values, permille) == pytest.approx(
            expected, rel=0.0, abs=1e-15
        )


def test_linear_quantile_duplicates_are_deterministic() -> None:
    values = np.asarray([-2.0, -2.0, -2.0, 3.0, 3.0], dtype=np.float64)
    first = linear_quantile(values, 500)
    for _ in range(20):
        assert linear_quantile(values, 500) == first
    assert first == pytest.approx(-2.0)

import numpy as np
import pytest

from probegi import probes, sh, strategies
from probegi.errors import BudgetDeniedError
from probegi.hashing import sha256_json
from probegi.strategies import ProbeField


@pytest.fixture(scope="module")
def field():
    positions, directions, radiance, _ = probes.generate(8, 5, 64)
    coeffs = sh.project(radiance, directions, 3)
    return ProbeField(8, probes.DEFAULT_BBOX, coeffs)


def test_selection_reproducible(field):
    kwargs = dict(grid_n=8, order=3, n_queries=100000,
                  field_scale=field.field_scale,
                  memory_budget_bytes=4 * 1024 * 1024, time_budget_ms=250.0,
                  accuracy_budget=0.05)
    a = strategies.select_strategy(**kwargs)
    b = strategies.select_strategy(**kwargs)
    assert sha256_json(a) == sha256_json(b)
    assert a["strategy"] == "full"


def test_each_strategy_selectable(field):
    base = dict(grid_n=8, order=3, n_queries=100000,
                field_scale=field.field_scale, accuracy_budget=0.05)
    full = strategies.select_strategy(memory_budget_bytes=10 ** 9,
                                      time_budget_ms=1e9, **base)
    assert full["strategy"] == "full"
    nn = strategies.select_strategy(memory_budget_bytes=10 ** 9,
                                    time_budget_ms=100.0, **base)
    assert nn["strategy"] == "nearest_gradient"
    cl = strategies.select_strategy(memory_budget_bytes=50 * 1024,
                                    time_budget_ms=1e9, **base)
    assert cl["strategy"] == "cluster"


def test_selection_denied_when_nothing_fits(field):
    with pytest.raises(BudgetDeniedError) as exc:
        strategies.select_strategy(
            grid_n=32, order=3, n_queries=100000,
            field_scale=field.field_scale, memory_budget_bytes=1024,
            time_budget_ms=1e9, accuracy_budget=0.05)
    assert exc.value.code == "E_BUDGET_DENIED"


def test_estimated_ordering_matches_measured(field):
    rng = np.random.default_rng(0)
    pts = rng.uniform(-4, 4, (16, 3))
    nrm = rng.standard_normal((16, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    ref = strategies.evaluate(field, "full", pts, nrm)
    errs = {}
    for name in ("nearest_gradient", "cluster"):
        got = strategies.evaluate(field, name, pts, nrm)
        errs[name] = float(np.mean(np.linalg.norm(got - ref, axis=1)))
    est = {s: strategies.est_error_bound(s, 8, 3, field.field_scale)
           for s in strategies.STRATEGIES}
    assert est["full"] < est["nearest_gradient"] < est["cluster"]
    assert errs["nearest_gradient"] <= errs["cluster"] * 1.05

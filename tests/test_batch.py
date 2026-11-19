import numpy as np
import pytest

from probegi import probes, queries as qmod, sh
from probegi.strategies import ProbeField


@pytest.fixture(scope="module")
def field():
    positions, directions, radiance, _ = probes.generate(8, 11, 64)
    return ProbeField(8, probes.DEFAULT_BBOX,
                      sh.project(radiance, directions, 2))


@pytest.fixture(scope="module")
def queries():
    return qmod.generate_queries(11, 16)


def test_batch_matches_single(field, queries):
    for strategy in ("full", "nearest_gradient", "cluster"):
        batch = qmod.solve_batch(field, strategy, queries)
        single = qmod.solve_single(field, strategy, queries)
        check = qmod.check_batch_consistency(batch, single)
        assert check["consistent"], (strategy, check)


def test_shuffle_invariant_result_hash(field, queries):
    rng = np.random.default_rng(3)
    baseline = qmod.result_set_hash(qmod.solve_batch(field, "full", queries))
    for _ in range(5):
        shuffled = list(queries)
        rng.shuffle(shuffled)
        got = qmod.result_set_hash(qmod.solve_batch(field, "full", shuffled))
        assert got == baseline


def test_merge_is_key_ordered(field, queries):
    results = qmod.solve_batch(field, "full", queries)
    keys = [k for k, _ in qmod.merge_by_key(results)]
    assert keys == sorted(keys)
    assert keys[0] == "cam:000"


def test_query_set_deterministic():
    a = qmod.generate_queries(11, 16)
    b = qmod.generate_queries(11, 16)
    assert qmod.query_set_hash(a) == qmod.query_set_hash(b)

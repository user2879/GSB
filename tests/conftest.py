from __future__ import annotations

import json

import numpy as np
import pytest

from cqdiag.config import validate_config
from cqdiag.samples import SampleSet, SampleTensor


def make_config_raw(**overrides) -> dict:
    raw = {
        "tensors": [{"name": "t", "shape": [6, 4], "column_axis": 1}],
        "quantile_lo_permille": 0,
        "quantile_hi_permille": 1000,
        "scheme": "asymmetric",
        "quant_bits_candidates": [4, 8],
        "quant_bits": 8,
        "frac_bits": 8,
        "saturation": "clamp",
        "abs_tol": 0.1,
        "rel_tol": 0.1,
        "groups": {"t": [0, 0, 1, 1]},
        "group_oor_limit": 0.5,
        "max_workers": 2,
        "element_budget": 1000,
    }
    raw.update(overrides)
    return raw


def make_config(**overrides):
    return validate_config(make_config_raw(**overrides))


def make_samples(config, values: np.ndarray | None = None,
                 missing: np.ndarray | None = None) -> SampleSet:
    spec = config.tensors[0]
    if values is None:
        rng = np.random.default_rng(7)
        values = rng.standard_normal(spec.shape) * 2.0
    if missing is None:
        missing = np.zeros(spec.shape, dtype=bool)
    st = SampleTensor(name=spec.name, spec=spec,
                      values=np.asarray(values, dtype=np.float64),
                      missing=np.asarray(missing, dtype=bool))
    return SampleSet(tensors={spec.name: st})


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def samples(config):
    return make_samples(config)

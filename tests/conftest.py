import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from qu.config import PipelineConfig
from qu.synthetic import DEFAULT_TIERS, SyntheticGenerator


@pytest.fixture(scope="session")
def config():
    return PipelineConfig(seed=20261120, max_workers=4,
                          latency_budget_ms=25.0)


@pytest.fixture(scope="session")
def generated(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("data")
    generator = SyntheticGenerator(20261120)
    spec = next(spec for spec in DEFAULT_TIERS if spec.tier_id == "tiny")
    return generator.generate_tier(str(out_dir), spec)

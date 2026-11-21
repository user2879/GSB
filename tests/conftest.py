import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pe_diag.config import EffectiveConfig  # noqa: E402
from pe_diag.encoding import PositionEncoder  # noqa: E402
from pe_diag.config import Sample  # noqa: E402


@pytest.fixture
def small_config():
    return EffectiveConfig(
        tier_id="small", base=10000.0, train_len=8, target_len=32, dim=16,
        scale=1.0, chunk_size=8, precision_tiers=("float64", "float32"),
        workers=2, snapshot_interval=4, batch_point_budget=100000,
        memory_limit_mb=64.0,
        tolerances={"position": 1e-6, "score": 1e-6, "weight": 1e-8},
        sensitivity_base_grid=(0.0, 5000.0, 10000.0),
        sensitivity_scale_grid=(0.0, 0.5, 1.0),
    )


@pytest.fixture
def sample():
    rng = np.random.default_rng(42)
    dim = 16
    return Sample(
        sample_id="s-001", tier_id="small", query_pos=2,
        key_positions=(3, 6, 10, 20),
        query_features=tuple(rng.normal(size=dim)),
        key_features=tuple(tuple(rng.normal(size=dim)) for _ in range(4)),
        index=0,
    )

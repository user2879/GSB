import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fablock.config import OperatorConfig
from fablock.dataset import SampleDataset


@pytest.fixture
def cfg():
    return OperatorConfig(burst_query_blocks=(1,))


@pytest.fixture
def dataset(cfg):
    return SampleDataset(seed=20240501, s_q=20, s_k=24,
                         head_dim=cfg.head_dim,
                         query_block_len=cfg.query_block_len,
                         key_block_len=cfg.key_block_len,
                         burst_query_blocks=cfg.burst_query_blocks)

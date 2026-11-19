import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from fwaudit.generator import build_image, build_patch
from fwaudit.patch import BudgetConfig


@pytest.fixture
def partitions():
    return [
        {"name": "bootloader", "part_type": 0, "slot": 0,
         "payload_length": 256},
        {"name": "kernel", "part_type": 1, "slot": 0,
         "payload_length": 512},
        {"name": "appfs", "part_type": 2, "slot": 0,
         "payload_length": 768},
    ]


@pytest.fixture
def image_blob(partitions):
    return build_image(partitions, version=1, base_image_id=100)


@pytest.fixture
def budget():
    return BudgetConfig(
        image_byte_threshold=4096,
        op_count_threshold=16,
        chunk_size=8,
        max_ops_per_patch=4096,
        timeout_seconds=5.0,
        concurrency=1,
    )

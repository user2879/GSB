from __future__ import annotations

from pathlib import Path

import pytest

from routing_workbench.sample_generator import generate_samples


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    root = tmp_path / "samples"
    generate_samples(root)
    return root


import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_diag.samplegen import write_samples, write_preemption_samples  # noqa: E402


@pytest.fixture()
def sample_dir(tmp_path):
    target = tmp_path / "samples"
    write_samples(target)
    return target


@pytest.fixture()
def preempt_dir(tmp_path):
    target = tmp_path / "preempt"
    write_preemption_samples(target)
    return target


@pytest.fixture()
def state_dir(tmp_path):
    target = tmp_path / "state"
    target.mkdir()
    return target

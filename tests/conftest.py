import os
import shutil
import tempfile

import pytest

from hdr_sdr_pipeline.synth import generate_sequence


@pytest.fixture()
def tiny_workdir():
    """Small deterministic dataset: 8 frames, 96x54, keeps tests fast."""
    tmp = tempfile.mkdtemp(prefix="hdr_test_")
    generate_sequence(tmp, frames=8, width=96, height=54, seed=4242)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture()
def empty_workdir():
    tmp = tempfile.mkdtemp(prefix="hdr_test_empty_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)

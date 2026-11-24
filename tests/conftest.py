"""Shared pytest fixtures: deterministic local sample workspace."""
from __future__ import annotations

import os
import shutil

import pytest

from obsflow.sample_data import generate_samples, SEED


@pytest.fixture()
def sample_dir(tmp_path):
    target = tmp_path / "samples"
    result = generate_samples(str(target), seed=SEED, records_per_source=24)
    return result


@pytest.fixture()
def workdir(tmp_path):
    path = tmp_path / "work"
    path.mkdir()
    return str(path)

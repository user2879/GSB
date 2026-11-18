"""Shared pytest fixtures: deterministic synthetic dataset and configs."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voiceprint.config import build_config, load_pair_manifest  # noqa: E402
from voiceprint.synth import generate_dataset  # noqa: E402
from voiceprint.util import fingerprint, read_json  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(ROOT, "configs", "voiceprint_config.json")
PAIR_PATH = os.path.join(ROOT, "configs", "speaker_pairs.json")
INJECTION_PATH = os.path.join(ROOT, "configs", "injection_manifest.json")


@pytest.fixture(scope="session")
def base_config():
    return build_config(read_json(CONFIG_PATH))


@pytest.fixture(scope="session")
def pair_manifest():
    return load_pair_manifest(PAIR_PATH)


@pytest.fixture()
def dataset_dir(tmp_path):
    target = tmp_path / "samples"
    target.mkdir()
    generate_dataset(str(target), INJECTION_PATH)
    return str(target)


@pytest.fixture()
def injection_hash():
    return fingerprint(read_json(INJECTION_PATH))


@pytest.fixture()
def make_config(base_config):
    def _make(**overrides):
        raw = dict(base_config.raw)
        raw.update(overrides)
        return build_config(raw)
    return _make

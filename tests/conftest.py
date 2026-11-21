from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
EXAMPLES = ROOT / "examples"


@pytest.fixture
def sample_paths() -> dict[str, Path]:
    return {
        "weights": SAMPLES / "weights.jsonl",
        "calibration": SAMPLES / "calibration.json",
        "config": SAMPLES / "config.yaml",
    }


@pytest.fixture
def base_config() -> dict:
    return yaml.safe_load((SAMPLES / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def base_calibration() -> dict:
    return json.loads((SAMPLES / "calibration.json").read_text(encoding="utf-8"))


@pytest.fixture
def copy_inputs(tmp_path: Path, sample_paths: dict[str, Path]) -> dict[str, Path]:
    target = tmp_path / "inputs"
    target.mkdir()
    result: dict[str, Path] = {}
    for name, source in sample_paths.items():
        destination = target / source.name
        destination.write_bytes(source.read_bytes())
        result[name] = destination
    return result


def write_config(path: Path, config: dict) -> Path:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def clone(value):
    return copy.deepcopy(value)

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from aedwb.pipeline import run_workbench
from aedwb.samplegen import generate_sample


@pytest.fixture
def make_sample(tmp_path: Path):
    counter = {"value": 0}

    def factory(injected: bool = False) -> Path:
        counter["value"] += 1
        destination = tmp_path / f"sample-{counter['value']}"
        generate_sample(destination, injected=injected)
        return destination

    return factory


@pytest.fixture(scope="session")
def base_run(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("base-run")
    sample = root / "sample"
    state = root / "state"
    generate_sample(sample)
    report = run_workbench(sample, state, state / "report.json")
    return {"sample": sample, "state": state, "report": report}


@pytest.fixture
def copied_sample(base_run, tmp_path: Path) -> Path:
    destination = tmp_path / "sample-copy"
    shutil.copytree(base_run["sample"], destination)
    return destination


def load_config(sample: Path) -> dict:
    return json.loads((sample / "config.json").read_text(encoding="utf-8"))


def save_config(sample: Path, config: dict) -> None:
    (sample / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def config_helpers():
    return load_config, save_config


def run(sample: Path, root: Path) -> dict:
    return run_workbench(sample, root / "state", root / "state" / "report.json")

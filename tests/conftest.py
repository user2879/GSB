import shutil
from pathlib import Path

import pytest

from voicecmd.config import load_config
from voicecmd.samplegen import generate_dataset

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "pipeline.json"


@pytest.fixture()
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture()
def workspace(tmp_path, cfg):
    generate_dataset(cfg, tmp_path)
    shutil.copy(CONFIG_PATH, tmp_path / "config.json")
    return tmp_path


@pytest.fixture()
def loaded_pipeline(workspace, cfg):
    from voicecmd.pipeline import Pipeline

    pipeline = Pipeline(cfg, str(workspace), snapshot=False)
    pipeline.load_all()
    return pipeline

import json
import shutil
from pathlib import Path

import pytest

from workbench import samplegen


@pytest.fixture()
def small_samples(tmp_path):
    """Small deterministic sample set: 2x2 tiles of 32px, overlap 8."""
    out = tmp_path / "samples"
    samplegen.generate_all(out, rows=2, cols=2, tile_size=32, overlap=8)
    return out


@pytest.fixture()
def config_path(small_samples):
    return small_samples / "config.json"


def write_config(path, **overrides):
    cfg = json.loads(Path(path).read_text())
    cfg.update(overrides)
    Path(path).write_text(json.dumps(cfg, indent=2))
    return path

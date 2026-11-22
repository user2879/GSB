import json
from pathlib import Path

import pytest

from edge_lease.cli import generate_samples


@pytest.fixture()
def sample_workspace(tmp_path: Path) -> Path:
    generate_samples(tmp_path, base_port=48000)
    return tmp_path


def write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

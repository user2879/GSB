from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edge_ledger.sample_gen import write_sample_tree
from edge_ledger.spec import load_spec_dir


@pytest.fixture()
def spec_dir(tmp_path: Path) -> Path:
    path = tmp_path / "spec"
    write_sample_tree(path, allocate_ports=True, create_snapshot=True)
    return path


@pytest.fixture()
def spec(spec_dir: Path):
    return load_spec_dir(spec_dir)

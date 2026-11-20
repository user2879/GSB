from __future__ import annotations

from pathlib import Path

import pytest

from ckptmeta.config import load_config
from ckptmeta.ledger import load_ledger
from ckptmeta.samples import generate_fixtures


@pytest.fixture
def fixtures(tmp_path: Path):
    return generate_fixtures(tmp_path / "fixtures")


@pytest.fixture
def loaded(fixtures):
    return load_config(fixtures.config), load_ledger(fixtures.ledger)

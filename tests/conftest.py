from __future__ import annotations

from pathlib import Path

import pytest

from snapdiag.store import Session


@pytest.fixture
def session(tmp_path: Path) -> Session:
    return Session.initialize(
        tmp_path / "session",
        block_size=512,
        block_count=16,
        threshold=0.5,
    )


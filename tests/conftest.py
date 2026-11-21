from __future__ import annotations

from pathlib import Path

import pytest

from timbre_lab.synth import generate_demo


@pytest.fixture(scope="session")
def demo_workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("demo-input")
    return generate_demo(root, seed=20261121)

from __future__ import annotations

import pytest

from fingerprint_workbench.generator import GeneratorConfig, generate_materials


@pytest.fixture(scope="session")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("generated")
    return generate_materials(root / "materials", GeneratorConfig(seed=20261119, track_count=7))

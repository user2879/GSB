from __future__ import annotations

from pathlib import Path

import pytest

from gsb.context import load_project
from gsb.pipeline import PipelineResult, run_pipeline


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def project():
    return load_project(root=ROOT)


@pytest.fixture(scope="session")
def baseline(project) -> PipelineResult:
    return run_pipeline(
        project.dataset,
        config=project.config,
        templates=project.require_templates(),
    )


@pytest.fixture(scope="session")
def flip_project():
    return load_project(
        root=ROOT,
        config_path="configs/quantization-flip.yaml",
        dataset_path="samples/quantization_flip.dataset.yaml",
    )


@pytest.fixture(scope="session")
def flip_q8(flip_project) -> PipelineResult:
    return run_pipeline(
        flip_project.dataset,
        config=flip_project.config.with_quantization("Q8"),
        templates=flip_project.require_templates(),
    )

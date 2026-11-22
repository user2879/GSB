from __future__ import annotations

from pathlib import Path

import pytest

from image_edit_workbench.config import load_config
from image_edit_workbench.image_registry import load_images
from image_edit_workbench.manifest import load_edit_manifest
from image_edit_workbench.samples import generate_sample_workspace


@pytest.fixture(scope="session")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("workbench")
    generated = generate_sample_workspace(root)
    sample_dir = root / "samples"
    config = load_config(root / "config" / "default.json")
    images = load_images(sample_dir / "images", expected_size=config.image_size)
    manifest = load_edit_manifest(sample_dir / "edits.json", images=images)
    assert not manifest.failures
    return {
        "root": root,
        "sample_dir": sample_dir,
        "config": config,
        "images": images,
        "regions": manifest.regions,
        "generated": generated,
    }


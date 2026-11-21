from __future__ import annotations

import json
from pathlib import Path

import pytest

from remote_seg.samplegen import generate_sample_workspace


def write_config(
    path: Path,
    *,
    max_workers: int = 3,
    recompute_budget: int = 50,
    fusion_version: str = "f2",
    min_connected_area: int = 4,
) -> Path:
    base = json.loads((Path(__file__).parents[1] / "configs" / "segmentation.json").read_text())
    base["max_workers"] = max_workers
    base["recompute_budget"] = recompute_budget
    base["overlap_fusion"]["strategy_version"] = fusion_version
    base["postprocess"]["min_connected_area"] = min_connected_area
    path.write_text(json.dumps(base, sort_keys=True, indent=2) + "\n")
    return path


@pytest.fixture()
def sample_workspace(tmp_path: Path) -> dict[str, object]:
    data_dir = tmp_path / "data"
    generate_sample_workspace(
        data_dir,
        width=32,
        height=32,
        rows=2,
        cols=3,
        overlap=6,
        seed=777,
        patch_batches=2,
    )
    config_path = write_config(tmp_path / "segmentation.json", max_workers=3, recompute_budget=50)
    return {
        "root": tmp_path,
        "data": data_dir,
        "index": data_dir / "index.json",
        "fusion": data_dir / "fusion_versions.json",
        "plan": data_dir / "patch_plan.json",
        "config": config_path,
    }

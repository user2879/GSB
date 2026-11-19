from __future__ import annotations

from pathlib import Path

from layout_workbench.cache import CacheStore
from layout_workbench.commands import initial_command_state
from layout_workbench.compare import first_divergence
from layout_workbench.config import RecognitionConfig
from layout_workbench.loading import scan_image_sources
from layout_workbench.pipeline import Pipeline


def _state(report: dict, image: str, stage: str) -> str:
    return next(item for item in report["images"] if item["name"] == image)["stages"][stage]["state"]


def test_stage_dependency_invalidation(sample_data: Path, tmp_path: Path) -> None:
    cache = CacheStore(tmp_path / "cache")
    sources = scan_image_sources(sample_data)
    base = RecognitionConfig()
    first = Pipeline(base, cache).run(sources)
    assert all(_state(first, "sample_a", stage) == "recomputed" for stage in ("detection", "line_aggregation", "segmentation", "recognition"))

    second = Pipeline(base, cache).run(sources)
    assert all(_state(second, "sample_a", stage) == "reused" for stage in ("detection", "line_aggregation", "segmentation", "recognition"))

    metric = Pipeline(base.with_updates(match_metric="ink_balance"), cache).run(sources)
    assert _state(metric, "sample_a", "detection") == "reused"
    assert _state(metric, "sample_a", "line_aggregation") == "reused"
    assert _state(metric, "sample_a", "segmentation") == "reused"
    assert _state(metric, "sample_a", "recognition") == "recomputed"

    width = Pipeline(base.with_updates(char_max_width=22), cache).run(sources)
    assert _state(width, "sample_a", "detection") == "reused"
    assert _state(width, "sample_a", "line_aggregation") == "reused"
    assert _state(width, "sample_a", "segmentation") == "recomputed"
    assert _state(width, "sample_a", "recognition") == "recomputed"

    line = Pipeline(base.with_updates(line_merge_projection_gap=8), cache).run(sources)
    assert _state(line, "sample_a", "detection") == "reused"
    assert _state(line, "sample_a", "line_aggregation") == "recomputed"
    assert _state(line, "sample_a", "segmentation") == "recomputed"
    assert _state(line, "sample_a", "recognition") == "recomputed"

    threshold = Pipeline(base.with_updates(binarization_threshold=168), cache).run(sources)
    assert _state(threshold, "sample_a", "detection") == "recomputed"
    assert _state(threshold, "sample_a", "line_aggregation") == "recomputed"


def test_image_replacement_only_invalidates_that_image(sample_data: Path, tmp_path: Path) -> None:
    cache = CacheStore(tmp_path / "cache")
    state = initial_command_state(sample_data, RecognitionConfig(), sample_data.parent)
    initial = Pipeline(state.config, cache).run(state.ordered_sources())
    replacement = sample_data / "replacements" / "sample_b_v2.png"
    truth = sample_data / "replacements" / "sample_b_v2.truth.json"
    state.apply(
        {
            "id": "replace",
            "type": "replace_image",
            "payload": {
                "name": "sample_b",
                "source_png": str(replacement),
                "source_truth": str(truth),
            },
        }
    )
    replaced = Pipeline(state.config, cache).run(state.ordered_sources())
    for stage in ("detection", "line_aggregation", "segmentation", "recognition"):
        assert _state(replaced, "sample_b", stage) == "recomputed"
        assert _state(replaced, "sample_a", stage) == "reused"
    assert first_divergence(initial, replaced) is not None


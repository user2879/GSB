from __future__ import annotations

import copy
import shutil
from pathlib import Path

from ancient_vlayout.engine import RecognitionEngine
from ancient_vlayout.samples import generate_samples
from ancient_vlayout.util import atomic_write_json, read_json


def _run(samples: Path, state: Path) -> object:
    return RecognitionEngine(samples, state_dir=state).settle()


def _stages(result) -> set[str]:
    return {item["stage"] for item in result.report["cache"]["recomputed"]}


def _all_products(result) -> list[str]:
    return [item["product"] for item in result.report["cache"]["recomputed"]]


def test_edit_sequence_cache_invalidation_and_full_rebuild_equivalence(
    samples_copy: Path, tmp_path: Path
) -> None:
    state = tmp_path / "incremental-state"
    baseline = _run(samples_copy, state)
    assert baseline.recognition["images"] and not baseline.pending

    config = read_json(samples_copy / "config.json")

    metric_config = copy.deepcopy(config)
    metric_config["match_metric"] = "l1"
    atomic_write_json(samples_copy / "config.json", metric_config)
    metric_result = _run(samples_copy, state)
    assert _stages(metric_result) == {"match"}
    assert {item["stage"] for item in metric_result.report["cache"]["hits"]} == {
        "columns",
        "split",
        "border",
    }

    height_config = copy.deepcopy(metric_config)
    height_config["min_char_height"] = 24
    atomic_write_json(samples_copy / "config.json", height_config)
    height_result = _run(samples_copy, state)
    assert _stages(height_result) == {"split", "border", "match"}
    assert {item["stage"] for item in height_result.report["cache"]["hits"]} == {"columns"}

    penalty_config = copy.deepcopy(height_config)
    penalty_config["split_penalty"] = 15
    atomic_write_json(samples_copy / "config.json", penalty_config)
    penalty_result = _run(samples_copy, state)
    assert _stages(penalty_result) == {"border", "match"}
    assert {item["stage"] for item in penalty_result.report["cache"]["hits"]} == {
        "columns",
        "split",
    }

    threshold_config = copy.deepcopy(penalty_config)
    threshold_config["binarization_threshold"] = 155
    atomic_write_json(samples_copy / "config.json", threshold_config)
    threshold_result = _run(samples_copy, state)
    assert _stages(threshold_result) == {"columns", "split", "border", "match"}

    projection_config = copy.deepcopy(threshold_config)
    projection_config["projection_window"] = 2
    atomic_write_json(samples_copy / "config.json", projection_config)
    projection_result = _run(samples_copy, state)
    assert _stages(projection_result) == {"columns", "split", "border", "match"}

    alternate = tmp_path / "alternate-samples"
    generate_samples(alternate, seed=20261125)
    shutil.copy2(alternate / "page_01.png", samples_copy / "page_01.png")
    shutil.copy2(alternate / "page_01.truth.json", samples_copy / "page_01.truth.json")
    replacement_result = _run(samples_copy, state)
    recomputed_products = _all_products(replacement_result)
    assert recomputed_products
    assert all("page_01.png" in product for product in recomputed_products)
    assert not any("page_02.png" in product for product in recomputed_products)
    assert not any("page_03.png" in product for product in recomputed_products)
    assert not any("page_04.png" in product for product in recomputed_products)

    fresh = _run(samples_copy, tmp_path / "fresh-state")
    assert replacement_result.recognition == fresh.recognition
    assert replacement_result.pending == fresh.pending == []


def test_cache_and_ledger_lists_have_deterministic_order(samples_copy: Path, tmp_path: Path) -> None:
    first = _run(samples_copy, tmp_path / "first")
    second = _run(samples_copy, tmp_path / "second")
    assert first.recognition == second.recognition
    for field in ("hits", "recomputed", "invalidated"):
        left = first.report["cache"][field]
        right = second.report["cache"][field]
        assert left == right
        assert left == sorted(left, key=lambda item: (item["product"], item["stage"]))

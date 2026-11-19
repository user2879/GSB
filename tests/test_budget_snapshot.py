from __future__ import annotations

from pathlib import Path

from layout_workbench.cache import CacheStore
from layout_workbench.compare import semantic_projection
from layout_workbench.config import RecognitionConfig
from layout_workbench.loading import scan_image_sources
from layout_workbench.pipeline import Pipeline
from layout_workbench.snapshot import SnapshotStore, snapshot_stage_docs


def test_point_budget_leaves_stable_pending_queue(sample_data: Path, tmp_path: Path) -> None:
    config = RecognitionConfig(char_match_point_budget=195)
    cache = CacheStore(tmp_path / "cache")
    first = Pipeline(config, cache).run(scan_image_sources(sample_data))
    assert first["budget"]["consumed"] == 195
    assert first["budget"]["pending_count"] == 20
    assert all(item["code"] == "E_POINT_BUDGET_PENDING" for item in first["pending_match_queue"])
    assert first["pending_match_queue"] == sorted(
        first["pending_match_queue"],
        key=lambda item: (item["image"], item["line_index"], item["segment_index"]),
    )
    second = Pipeline(config, cache).run(scan_image_sources(sample_data))
    assert semantic_projection(first) == semantic_projection(second)
    assert second["budget"]["consumed"] == first["budget"]["consumed"]
    assert second["budget"]["reused"] == 195
    assert second["budget"]["recomputed"] == 0


def test_snapshot_whole_document_version_replacement(sample_data: Path, tmp_path: Path) -> None:
    config = RecognitionConfig()
    cache = CacheStore(tmp_path / "cache")
    report = Pipeline(config, cache).run(scan_image_sources(sample_data))
    stage_keys = {
        image["name"]: {
            stage: details["cache_key"] for stage, details in image["stages"].items()
        }
        for image in report["images"]
    }
    stages = snapshot_stage_docs(
        report["images"],
        stage_keys,
        consumed_points=report["budget"]["consumed"],
        pending_images=set(),
    )
    store = SnapshotStore(tmp_path / "snapshots", config.config_hash)
    first = store.commit(
        global_version=1,
        command_index=0,
        command_id="initial",
        stages=stages,
        config_hash=config.config_hash,
    )
    assert first["written"] is True
    assert first["reason"] == "initial"
    second = store.commit(
        global_version=2,
        command_index=1,
        command_id="next",
        stages=stages,
        config_hash=config.config_hash,
    )
    assert second["written"] is True
    assert second["reason"] == "version_changed"
    document = store.load()
    assert document["global_version"] == 2
    assert document["previous_global_version"] == 1
    assert document["replacement_mode"] == "whole_document_atomic_replace"
    assert set(document["stages"]) == {
        "detection",
        "line_aggregation",
        "segmentation",
        "recognition",
    }
    assert len(list((tmp_path / "snapshots").glob("current.json"))) == 1
    assert store.verify(config.config_hash)["valid"] is True


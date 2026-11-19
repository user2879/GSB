from __future__ import annotations

import tempfile
from pathlib import Path

from layout_workbench.cache import CacheStore
from layout_workbench.compare import semantic_projection
from layout_workbench.config import RecognitionConfig
from layout_workbench.loading import scan_image_sources
from layout_workbench.pipeline import Pipeline


def test_thread_count_does_not_change_stable_output(sample_data: Path) -> None:
    sources = scan_image_sources(sample_data)
    with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as four:
        serial = Pipeline(
            RecognitionConfig(thread_pool_workers=1), CacheStore(Path(one))
        ).run(sources)
        parallel = Pipeline(
            RecognitionConfig(thread_pool_workers=4), CacheStore(Path(four))
        ).run(sources)
    assert semantic_projection(serial) == semantic_projection(parallel)
    assert [image["name"] for image in serial["images"]] == [
        image["name"] for image in parallel["images"]
    ]
    assert serial["concurrency"]["max_workers_observed"] <= 1
    assert parallel["concurrency"]["max_workers_observed"] <= 4
    assert serial["pending_match_queue"] == parallel["pending_match_queue"]


from __future__ import annotations

import tempfile
from pathlib import Path

from layout_workbench.cache import CacheStore
from layout_workbench.config import RecognitionConfig
from layout_workbench.loading import scan_image_sources
from layout_workbench.matching import match_point_cost, recognize_segment
from layout_workbench.pipeline import Pipeline


def test_metric_switch_changes_score_and_character(sample_data: Path) -> None:
    sources = scan_image_sources(sample_data)
    with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
        l1 = Pipeline(
            RecognitionConfig(match_metric="l1"), CacheStore(Path(one))
        ).run(sources)
        ink = Pipeline(
            RecognitionConfig(match_metric="ink_balance"), CacheStore(Path(two))
        ).run(sources)
    l1_char = l1["images"][0]["lines"][0]["characters"][1]
    ink_char = ink["images"][0]["lines"][0]["characters"][1]
    assert l1_char["bbox"] == ink_char["bbox"]
    assert (l1_char["text"], l1_char["score"]) != (ink_char["text"], ink_char["score"])
    assert l1_char["text"] == "E"
    assert ink_char["text"] == "D"
    assert ink_char["candidates"][0]["score"] == ink_char["candidates"][1]["score"]
    assert ink_char["candidates"][1]["eliminated_reason"] == "tie_lost_lower_codepoint"
    assert ink_char["candidates"][0]["codepoint"] < ink_char["candidates"][1]["codepoint"]
    assert len(ink_char["candidates"]) == RecognitionConfig().top_k_candidates


def test_recognition_keeps_metric_and_candidate_provenance(sample_data: Path) -> None:
    config = RecognitionConfig(match_metric="cosine")
    from layout_workbench.loading import load_image_source

    image = load_image_source(scan_image_sources(sample_data)[0], config)
    result = recognize_segment(image.gray, [56, 42, 71, 63], config)
    assert result["metric"] == "cosine"
    assert result["point_cost"] == match_point_cost()
    assert result["candidates"]
    assert all("eliminated_reason" in item for item in result["candidates"])
    assert result["candidate_count"] >= len(result["candidates"])


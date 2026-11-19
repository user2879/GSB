from __future__ import annotations

from pathlib import Path

from layout_workbench.config import RecognitionConfig
from layout_workbench.detection import aggregate_line_candidates, detect_blocks
from layout_workbench.loading import load_image_source, scan_image_sources
from layout_workbench.segmentation import segment_line


def _image(sample_data: Path, name: str, config: RecognitionConfig | None = None):
    source = next(source for source in scan_image_sources(sample_data) if source.name == name)
    return load_image_source(source, config or RecognitionConfig())


def test_detection_finds_stable_lines_and_suppression(sample_data: Path) -> None:
    config = RecognitionConfig()
    image = _image(sample_data, "sample_a", config)
    detection = detect_blocks(image.gray, config)
    lines = aggregate_line_candidates(image.gray, detection, config)
    assert [line["bbox"] for line in lines["candidate_lines"]] == [
        [35, 42, 134, 63],
        [35, 124, 122, 145],
    ]
    assert lines["suppressed_lines"] == []
    assert all(line["score"] >= config.candidate_score_threshold for line in lines["candidate_lines"])

    strict = config.with_updates(candidate_score_threshold=0.999)
    strict_lines = aggregate_line_candidates(image.gray, detection, strict)
    assert strict_lines["candidate_lines"] == []
    assert len(strict_lines["suppressed_lines"]) == 2
    assert all(
        {"bbox", "score", "scale", "reason"} <= set(line)
        for line in strict_lines["suppressed_lines"]
    )


def test_narrow_merge_and_wide_split_records(sample_data: Path) -> None:
    config = RecognitionConfig()
    image_a = _image(sample_data, "sample_a", config)
    detection_a = detect_blocks(image_a.gray, config)
    lines_a = aggregate_line_candidates(image_a.gray, detection_a, config)["candidate_lines"]
    quote_line = segment_line(image_a.gray, lines_a[1], config)
    merge_records = [r for r in quote_line["records"] if r["event"] == "merge_narrow"]
    assert len(merge_records) == 2
    assert all(r["classification"] == "normal_intra_glyph_gap" for r in merge_records)
    assert all(r["result_width"] >= config.char_min_width for r in merge_records)
    assert [segment["text"] if "text" in segment else segment["width"] for segment in quote_line["segments"]] == [9, 15, 15, 15, 9]

    image_b = _image(sample_data, "sample_b", config)
    detection_b = detect_blocks(image_b.gray, config)
    lines_b = aggregate_line_candidates(image_b.gray, detection_b, config)["candidate_lines"]
    bridge_line = segment_line(image_b.gray, lines_b[0], config)
    split_records = [r for r in bridge_line["records"] if r["event"] == "split_wide"]
    assert len(split_records) == 1
    assert split_records[0]["original_width"] > config.char_max_width
    assert split_records[0]["classification"] in {
        "normal_projection_valley",
        "forced_width_split",
    }
    assert split_records[0]["result_ranges"][0][0] >= bridge_line["line_bbox"][0]


def test_forced_swallow_is_distinguishable(sample_data: Path) -> None:
    config = RecognitionConfig()
    image = _image(sample_data, "sample_c", config)
    detection = detect_blocks(image.gray, config)
    lines = aggregate_line_candidates(image.gray, detection, config)["candidate_lines"]
    segmented = segment_line(image.gray, lines[0], config)
    forced = [r for r in segmented["records"] if r["classification"] == "forced_swallow"]
    assert forced
    assert any(r["event"] == "merge_narrow" for r in forced)
    assert forced[0]["original_width_ranges"]


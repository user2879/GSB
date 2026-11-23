from __future__ import annotations

from pathlib import Path

from ancient_vlayout.config import load_config
from ancient_vlayout.imaging import binarize
from ancient_vlayout.loader import load_dataset
from ancient_vlayout.matching import match_character
from ancient_vlayout.pipeline import make_border_artifact, make_columns_artifact, make_split_artifact
from ancient_vlayout.templates import build_templates, template_version
from ancient_vlayout.util import read_json


def test_columns_borders_split_and_recognition_match_truth(generated_samples: Path) -> None:
    config = load_config(generated_samples / "config.json")
    dataset = load_dataset(generated_samples, config)
    templates = build_templates()
    version = template_version(templates)
    split_count = 0
    char_count = 0

    for image_name, image in sorted(dataset.images.items()):
        truth = read_json(generated_samples / image_name.replace(".png", ".truth.json"))
        expected_chars = [char for column in truth["columns"] for char in column["chars"]]
        ink = binarize(image.array, config["binarization_threshold"])
        columns = make_columns_artifact(image.array, config, image.content_fingerprint, "columns-key")
        assert [column["x0"] for column in columns["columns"]] == sorted(
            [column["x0"] for column in columns["columns"]], reverse=True
        )
        assert [(column["x0"], column["x1"]) for column in columns["columns"]] == [
            (column["bbox"][0], column["bbox"][2]) for column in truth["columns"]
        ]
        actual_chars = []
        for column in columns["columns"]:
            split = make_split_artifact(
                ink,
                config,
                image_name,
                image.content_fingerprint,
                "columns-key",
                f"split-{column['column_index']}",
                column,
            )
            assert all(item["code"] == "E_SEGMENT_TOO_SHORT" for item in split["discarded"])
            char_index = 0
            for segment in split["candidates"]:
                border = make_border_artifact(
                    ink,
                    config,
                    f"border-{segment['segment_id']}",
                    segment,
                    "columns-key",
                    f"split-{column['column_index']}",
                )
                if border["split"]["applied"]:
                    split_count += 1
                    assert border["split"]["tie_break"] == "upper"
                    assert border["split"]["tie"] is True
                    assert border["split"]["selected_cut"] == min(border["split"]["tie_candidates"])
                    assert border["split"]["infeasible_cuts"]
                    assert border["split"]["cut_evidence"]
                    assert all(item["valley_depth"] >= 0 for item in border["split"]["cut_evidence"])
                for piece in border["pieces"]:
                    assert piece["parent_segment_id"] == segment["segment_id"]
                    match = match_character(
                        image.array,
                        piece["bbox"],
                        config,
                        templates,
                        image_name,
                        column["column_index"],
                        char_index,
                        piece["piece_id"],
                        segment["segment_id"],
                        piece["parent_segment_id"],
                        [column["x0"], column["x1"]],
                        version,
                        f"match-{piece['piece_id']}",
                    )
                    actual_chars.append(match)
                    assert 0.0 <= match["confidence_value"] <= 1.0
                    assert match["confidence_fixed"] % match["confidence_quantum"] == 0
                    char_index += 1
        assert len(actual_chars) == len(expected_chars)
        for actual, expected in zip(actual_chars, expected_chars):
            assert actual["bbox"] == expected["bbox"]
            assert actual["recognized_char"] == expected["char"]
        char_count += len(actual_chars)

    assert split_count == 4
    assert char_count == 80


def test_concurrency_does_not_exceed_config(generated_samples: Path, tmp_path: Path) -> None:
    from ancient_vlayout.engine import RecognitionEngine

    result = RecognitionEngine(generated_samples, state_dir=tmp_path / "state").settle()
    concurrency = result.report["concurrency"]
    assert concurrency["observed_peak_tasks"] <= concurrency["configured_max_workers"]


def test_segment_fingerprints_are_stable_and_content_derived(generated_samples: Path) -> None:
    config = load_config(generated_samples / "config.json")
    dataset = load_dataset(generated_samples, config)
    image = dataset.images["page_01.png"]
    ink = binarize(image.array, config["binarization_threshold"])
    columns = make_columns_artifact(image.array, config, image.content_fingerprint, "k")
    first = make_split_artifact(
        ink,
        config,
        image.name,
        image.content_fingerprint,
        "k",
        "s",
        columns["columns"][0],
    )
    second = make_split_artifact(
        ink,
        config,
        image.name,
        image.content_fingerprint,
        "k",
        "s",
        columns["columns"][0],
    )
    assert first["candidates"] == second["candidates"]
    assert all(len(item["segment_fingerprint"]) == 64 for item in first["candidates"])


def test_character_code_is_tie_breaker(generated_samples: Path) -> None:
    config = load_config(generated_samples / "config.json")
    config = dict(config)
    config["fusion_weights"] = [1]
    dataset = load_dataset(generated_samples, config)
    image = dataset.images["page_01.png"]
    ink = binarize(image.array, config["binarization_threshold"])
    columns = make_columns_artifact(image.array, config, image.content_fingerprint, "k")
    split = make_split_artifact(
        ink, config, image.name, image.content_fingerprint, "k", "s", columns["columns"][0]
    )
    border = make_border_artifact(
        ink, config, "b", split["candidates"][0], "k", "s"
    )
    template = build_templates()["口"][0]
    tied_templates = {"乙": [template], "甲": [template.copy()]}
    result = match_character(
        image.array,
        border["pieces"][0]["bbox"],
        config,
        tied_templates,
        image.name,
        0,
        0,
        "piece",
        "segment",
        "segment",
        [columns["columns"][0]["x0"], columns["columns"][0]["x1"]],
        "tie-template-version",
        "tie-key",
    )
    assert result["recognized_char"] == "乙"
    assert result["top_tied_chars"] == ["乙", "甲"]

from __future__ import annotations

from pathlib import Path

import pytest

from timbre_lab.aggregation import aggregate_speaker
from timbre_lab.assignment import adjacent_pairs, decide_assignment
from timbre_lab.audio import scan_wavs
from timbre_lab.config import load_config
from timbre_lab.features import extract_segment_features
from timbre_lab.pipeline import compute_full
from timbre_lab.quantization import build_description
from timbre_lab.segmentation import build_document


def _features_and_descriptions(config, document):
    features = {}
    descriptions = {}
    for segment in document.ordered_segments():
        feature = extract_segment_features(document.files[segment.file], segment, config)
        features[segment.segment_id] = feature
        descriptions[segment.segment_id] = build_description(feature, config, {})
    return features, descriptions


def test_quantization_bit_width_changes_integer_and_vector(demo_workspace: dict[str, Path]) -> None:
    config = load_config(demo_workspace["config"])
    scan = scan_wavs(demo_workspace["samples"])
    document, _, _ = build_document(scan, config)
    segment = document.ordered_segments()[0]
    feature = extract_segment_features(document.files[segment.file], segment, config)
    low = build_description(feature, config, {"f0_mean": 4, "speech_rate": 4})
    high = build_description(feature, config, {"f0_mean": 10, "speech_rate": 8})
    repeat = build_description(feature, config, {"f0_mean": 4, "speech_rate": 4})
    assert low["dimensions"][0]["integer"] != high["dimensions"][0]["integer"]
    assert low["vector"][0] != high["vector"][0]
    assert low["dimensions"][0]["step"] > high["dimensions"][0]["step"]
    assert low == repeat
    assert low["dimensions"][0]["residual"] == pytest.approx(
        low["dimensions"][0]["value"] - low["dimensions"][0]["reconstructed"]
    )


def test_similarity_equal_threshold_is_inclusive_with_dual_references(demo_workspace: dict[str, Path]) -> None:
    config = load_config(demo_workspace["config"])
    scan = scan_wavs(demo_workspace["samples"])
    document, _, _ = build_document(scan, config)
    left, right = adjacent_pairs(document)[0]
    left.speaker = "speaker_a"
    right.speaker = "speaker_b"
    features, descriptions = _features_and_descriptions(config, document)
    from timbre_lab.quantization import similarity

    score, _ = similarity(descriptions[left.segment_id], descriptions[right.segment_id], config)
    config.similarity_threshold = score
    decision = decide_assignment(
        left,
        right,
        descriptions[left.segment_id],
        descriptions[right.segment_id],
        features[left.segment_id],
        features[right.segment_id],
        config,
        {},
    )
    assert decision["boundary"] is True
    assert decision["vector_same"] is True
    assert decision["boundary_reference"]["inclusive_same"] is True
    assert decision["boundary_reference"]["exclusive_same"] is False
    assert decision["boundary_reference"]["ordered_start_frames"] == [left.start_frame, right.start_frame]
    assert decision["conflict"]["code"] == "E_ANNOTATION_VECTOR_CONFLICT"
    assert len(decision["dual_level_reference"]["left"][0]["levels"]) == 2
    assert len(decision["dual_level_reference"]["left"][0]["levels"][0]) >= 4


def test_fixed_point_speaker_aggregation_and_degenerate_flag(demo_workspace: dict[str, Path]) -> None:
    config = load_config(demo_workspace["config"])
    scan = scan_wavs(demo_workspace["samples"])
    document, _, failures = build_document(scan, config)
    result = compute_full(document, config, {}, failures)
    assert {segment["pronunciation_style"] for segment in result["segments"]} >= {"slow", "fast"}
    assert {speaker["speaker"] for speaker in result["speaker_vectors"]} == {"spk_amber", "spk_noah"}
    features, descriptions = _features_and_descriptions(config, document)
    group = result["speaker_groups"][0]
    aggregate = next(item for item in result["speaker_vectors"] if item["speaker"] == group["speaker"])
    first_dimension = aggregate["dimensions"][0]
    manual = sum(
        descriptions[segment_id]["integers"][0] * features[segment_id]["effective_frame_count"]
        for segment_id in group["segment_ids"]
    )
    expected = (2 * manual + first_dimension["weight_denominator"]) // (2 * first_dimension["weight_denominator"])
    assert first_dimension["integer"] == expected
    assert first_dimension["rounding"] == "nearest_half_up_fixed_point"
    assert aggregate["degenerate_single_segment"] is False

    single_segment = group["segment_ids"][0]
    single = aggregate_speaker(
        {"speaker": group["speaker"], "segment_ids": [single_segment]},
        features,
        descriptions,
        config,
        {},
    )
    assert single["degenerate_single_segment"] is True


def test_quantization_out_of_range_is_rejected_without_mutation(demo_workspace: dict[str, Path]) -> None:
    from timbre_lab.commands import apply_command
    from timbre_lab.errors import E_QUANTIZATION_LEVEL, BusinessError

    config = load_config(demo_workspace["config"])
    scan = scan_wavs(demo_workspace["samples"])
    document, _, _ = build_document(scan, config)
    overrides: dict[str, int] = {}
    before = document.state_dict()
    with pytest.raises(BusinessError) as error:
        apply_command(
            document,
            config,
            overrides,
            {"id": "bad", "op": "set_quantization", "dimension": "f0_mean", "bit_width": 99},
            demo_workspace["root"],
        )
    assert error.value.code == E_QUANTIZATION_LEVEL
    assert overrides == {}
    assert document.state_dict() == before

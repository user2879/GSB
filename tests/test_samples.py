from __future__ import annotations

import json
from pathlib import Path

from haptic_diag.generator import bundle_bytes, load_config
from haptic_diag.model import PRIMITIVE_KINDS

ROOT = Path(__file__).resolve().parents[1]


def test_generation_is_byte_identical_and_matches_committed_samples() -> None:
    config = load_config(ROOT / "configs" / "samples.json")
    first_graphs, first_truth = bundle_bytes(config)
    second_graphs, second_truth = bundle_bytes(config)
    assert first_graphs == second_graphs
    assert first_truth == second_truth
    assert first_graphs == (ROOT / "samples" / "effect_graphs.json").read_bytes()
    assert first_truth == (ROOT / "samples" / "truth_manifest.json").read_bytes()


def test_samples_cover_required_scenarios_and_primitive_kinds() -> None:
    config = load_config(ROOT / "configs" / "samples.json")
    graphs, truth = bundle_bytes(config)
    graph_bundle = json.loads(graphs)
    truth_bundle = json.loads(truth)
    scenarios = {item["scenario"] for item in config["samples"]}
    assert scenarios == {
        "single_channel",
        "multi_channel_overlap",
        "cross_slice_boundary",
        "long_duration",
    }
    assert {primitive["kind"] for sample in graph_bundle["samples"] for primitive in sample["primitives"]} == set(PRIMITIVE_KINDS)
    for sample, sample_truth in zip(graph_bundle["samples"], truth_bundle["samples"], strict=True):
        assert sample_truth["graph_content_hash"]
        assert sample_truth["expected_frame_count"] == (sample["duration_us"] + sample["frame_period_us"] - 1) // sample["frame_period_us"]
        for primitive, primitive_truth in zip(sample["primitives"], sample_truth["primitives"], strict=True):
            assert primitive_truth["channel_count"] == len(primitive["channels"])
            assert primitive_truth["start_us"] == primitive["start_us"]
            assert primitive_truth["end_us"] == primitive["start_us"] + primitive["duration_us"]
            assert primitive_truth["duration_us"] > 0


def test_single_channel_sample_uses_only_one_channel() -> None:
    config = load_config(ROOT / "configs" / "samples.json")
    from haptic_diag.generator import generate_sample

    sample_config = next(item for item in config["samples"] if item["scenario"] == "single_channel")
    graph = generate_sample(sample_config)
    assert {channel for primitive in graph.primitives for channel in primitive.channels} == {"left_grip"}


def test_long_duration_sample_exceeds_twenty_thousand_microsecond_span() -> None:
    config = load_config(ROOT / "configs" / "samples.json")
    from haptic_diag.generator import generate_sample

    sample_config = next(item for item in config["samples"] if item["scenario"] == "long_duration")
    graph = generate_sample(sample_config)
    assert graph.duration_us >= 100_000_000
    assert graph.frame_count == 6000


def test_effect_graph_roundtrip_normalizes_channel_bindings() -> None:
    config = load_config(ROOT / "configs" / "samples.json")
    from haptic_diag.generator import generate_sample
    from haptic_diag.model import EffectGraph

    graph = generate_sample(config["samples"][1])
    restored = EffectGraph.from_dict(graph.to_dict())
    assert restored == graph
    assert all(isinstance(primitive.channels, tuple) for primitive in restored.primitives)

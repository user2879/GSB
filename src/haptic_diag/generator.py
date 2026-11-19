"""Deterministic effect sample generation and truth manifests."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json_text, write_canonical_json
from .model import CompositionLayer, EffectGraph, Primitive, TimeSlice


def _number(rng: random.Random, low: float, high: float, digits: int = 3) -> float:
    return round(rng.uniform(low, high), digits)


def _single_channel(config: Mapping[str, Any], rng: random.Random) -> EffectGraph:
    duration = int(config["duration_us"])
    primitives = (
        Primitive(
            id="single_sine",
            kind="sine",
            channels=("left_grip",),
            start_us=100_000,
            duration_us=1_200_000,
            amplitude=_number(rng, 0.45, 0.8),
            frequency_hz=_number(rng, 70.0, 150.0),
            phase_deg=_number(rng, 0.0, 359.0),
        ),
        Primitive(
            id="single_pulse",
            kind="pulse_train",
            channels=("left_grip",),
            start_us=1_000_000,
            duration_us=1_400_000,
            amplitude=_number(rng, 0.35, 0.7),
            frequency_hz=_number(rng, 20.0, 55.0),
            duty_cycle=_number(rng, 0.2, 0.6),
        ),
        Primitive(
            id="single_envelope",
            kind="envelope",
            channels=("left_grip",),
            start_us=2_200_000,
            duration_us=1_500_000,
            amplitude=_number(rng, 0.5, 0.9),
            frequency_hz=_number(rng, 80.0, 180.0),
            attack_us=260_000,
            release_us=340_000,
        ),
    )
    slices = (
        TimeSlice(
            id="single_slice_a",
            start_us=0,
            end_us=2_000_000,
            primitive_ids=("single_sine", "single_pulse"),
            blend="max",
        ),
        TimeSlice(
            id="single_slice_b",
            start_us=2_000_000,
            end_us=duration,
            primitive_ids=("single_pulse", "single_envelope"),
            blend="max",
        ),
    )
    layers = (CompositionLayer(id="single_layer", slice_ids=("single_slice_a", "single_slice_b")),)
    return EffectGraph(
        id=str(config["name"]),
        duration_us=duration,
        frame_period_us=int(config["frame_period_us"]),
        batch_size=int(config["batch_size"]),
        primitives=primitives,
        slices=slices,
        layers=layers,
    )


def _multi_channel_overlap(config: Mapping[str, Any], rng: random.Random) -> EffectGraph:
    duration = int(config["duration_us"])
    primitives = (
        Primitive(
            id="multi_left_grip_sine",
            kind="sine",
            channels=("left_grip",),
            start_us=100_000,
            duration_us=3_000_000,
            amplitude=_number(rng, 0.4, 0.8),
            frequency_hz=_number(rng, 90.0, 180.0),
        ),
        Primitive(
            id="multi_left_grip_pulse",
            kind="pulse_train",
            channels=("left_grip",),
            start_us=900_000,
            duration_us=2_200_000,
            amplitude=_number(rng, 0.3, 0.75),
            frequency_hz=_number(rng, 25.0, 60.0),
            duty_cycle=_number(rng, 0.15, 0.45),
        ),
        Primitive(
            id="multi_right_grip_envelope",
            kind="envelope",
            channels=("right_grip",),
            start_us=500_000,
            duration_us=3_500_000,
            amplitude=_number(rng, 0.5, 0.9),
            frequency_hz=_number(rng, 100.0, 210.0),
            attack_us=300_000,
            release_us=400_000,
        ),
        Primitive(
            id="multi_both_triggers",
            kind="sine",
            channels=("left_trigger", "right_trigger"),
            start_us=1_500_000,
            duration_us=2_800_000,
            amplitude=_number(rng, 0.45, 0.85),
            frequency_hz=_number(rng, 60.0, 140.0),
        ),
        Primitive(
            id="multi_all_channels_pulse",
            kind="pulse_train",
            channels=("left_grip", "right_grip", "left_trigger", "right_trigger"),
            start_us=2_500_000,
            duration_us=2_000_000,
            amplitude=_number(rng, 0.2, 0.5),
            frequency_hz=_number(rng, 18.0, 42.0),
            duty_cycle=_number(rng, 0.15, 0.35),
        ),
    )
    slices = (
        TimeSlice(
            id="multi_slice_early",
            start_us=0,
            end_us=2_600_000,
            primitive_ids=(
                "multi_left_grip_sine",
                "multi_left_grip_pulse",
                "multi_right_grip_envelope",
                "multi_both_triggers",
            ),
            blend="max",
        ),
        TimeSlice(
            id="multi_slice_late",
            start_us=2_400_000,
            end_us=duration,
            primitive_ids=(
                "multi_left_grip_sine",
                "multi_right_grip_envelope",
                "multi_both_triggers",
                "multi_all_channels_pulse",
            ),
            blend="max",
        ),
    )
    layers = (
        CompositionLayer(
            id="multi_layer",
            slice_ids=("multi_slice_early", "multi_slice_late"),
            blend="max",
        ),
    )
    return EffectGraph(
        id=str(config["name"]),
        duration_us=duration,
        frame_period_us=int(config["frame_period_us"]),
        batch_size=int(config["batch_size"]),
        primitives=primitives,
        slices=slices,
        layers=layers,
    )


def _cross_slice_boundary(config: Mapping[str, Any], rng: random.Random) -> EffectGraph:
    duration = int(config["duration_us"])
    primitives = (
        Primitive(
            id="cross_sine",
            kind="sine",
            channels=("right_trigger",),
            start_us=1_800_000,
            duration_us=2_400_000,
            amplitude=_number(rng, 0.5, 0.9),
            frequency_hz=_number(rng, 110.0, 190.0),
        ),
        Primitive(
            id="cross_envelope",
            kind="envelope",
            channels=("left_trigger", "right_trigger"),
            start_us=2_700_000,
            duration_us=1_900_000,
            amplitude=_number(rng, 0.4, 0.8),
            frequency_hz=_number(rng, 70.0, 160.0),
            attack_us=250_000,
            release_us=300_000,
        ),
        Primitive(
            id="cross_pulse",
            kind="pulse_train",
            channels=("left_grip",),
            start_us=4_400_000,
            duration_us=1_200_000,
            amplitude=_number(rng, 0.35, 0.65),
            frequency_hz=_number(rng, 22.0, 50.0),
            duty_cycle=_number(rng, 0.25, 0.55),
        ),
    )
    # The first two primitives deliberately straddle the 3,000,000 us slice
    # boundary.  This exercises slice membership across frame boundaries.
    slices = (
        TimeSlice(
            id="cross_left_of_boundary",
            start_us=0,
            end_us=3_000_000,
            primitive_ids=("cross_sine", "cross_envelope"),
            blend="max",
        ),
        TimeSlice(
            id="cross_right_of_boundary",
            start_us=3_000_000,
            end_us=duration,
            primitive_ids=("cross_sine", "cross_envelope", "cross_pulse"),
            blend="max",
        ),
    )
    layers = (
        CompositionLayer(
            id="cross_layer",
            slice_ids=("cross_left_of_boundary", "cross_right_of_boundary"),
            blend="max",
        ),
    )
    return EffectGraph(
        id=str(config["name"]),
        duration_us=duration,
        frame_period_us=int(config["frame_period_us"]),
        batch_size=int(config["batch_size"]),
        primitives=primitives,
        slices=slices,
        layers=layers,
    )


def _long_duration(config: Mapping[str, Any], rng: random.Random) -> EffectGraph:
    duration = int(config["duration_us"])
    primitives = (
        Primitive(
            id="long_left_grip",
            kind="sine",
            channels=("left_grip",),
            start_us=0,
            duration_us=110_000_000,
            amplitude=_number(rng, 0.45, 0.8),
            frequency_hz=_number(rng, 80.0, 150.0),
            phase_deg=_number(rng, 0.0, 359.0),
        ),
        Primitive(
            id="long_right_grip",
            kind="pulse_train",
            channels=("right_grip",),
            start_us=2_000_000,
            duration_us=116_000_000,
            amplitude=_number(rng, 0.3, 0.7),
            frequency_hz=_number(rng, 20.0, 45.0),
            duty_cycle=_number(rng, 0.2, 0.5),
        ),
        Primitive(
            id="long_triggers",
            kind="envelope",
            channels=("left_trigger", "right_trigger"),
            start_us=8_000_000,
            duration_us=108_000_000,
            amplitude=_number(rng, 0.4, 0.8),
            frequency_hz=_number(rng, 90.0, 170.0),
            attack_us=1_000_000,
            release_us=2_000_000,
        ),
    )
    midpoint = duration // 2
    slices = (
        TimeSlice(
            id="long_slice_a",
            start_us=0,
            end_us=midpoint + 5_000_000,
            primitive_ids=("long_left_grip", "long_right_grip", "long_triggers"),
            blend="max",
        ),
        TimeSlice(
            id="long_slice_b",
            start_us=midpoint - 5_000_000,
            end_us=duration,
            primitive_ids=("long_left_grip", "long_right_grip", "long_triggers"),
            blend="max",
        ),
    )
    layers = (
        CompositionLayer(id="long_layer", slice_ids=("long_slice_a", "long_slice_b"), blend="max"),
    )
    return EffectGraph(
        id=str(config["name"]),
        duration_us=duration,
        frame_period_us=int(config["frame_period_us"]),
        batch_size=int(config["batch_size"]),
        primitives=primitives,
        slices=slices,
        layers=layers,
    )


SCENARIOS = {
    "single_channel": _single_channel,
    "multi_channel_overlap": _multi_channel_overlap,
    "cross_slice_boundary": _cross_slice_boundary,
    "long_duration": _long_duration,
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def generate_sample(config: Mapping[str, Any]) -> EffectGraph:
    scenario = str(config["scenario"])
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown sample scenario: {scenario}")
    rng = random.Random(int(config["seed"]))
    return SCENARIOS[scenario](config, rng)


def truth_entry(graph: EffectGraph, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": graph.id,
        "scenario": config["scenario"],
        "seed": config["seed"],
        "duration_us": graph.duration_us,
        "frame_period_us": graph.frame_period_us,
        "expected_frame_count": graph.frame_count,
        "batch_size": graph.batch_size,
        "graph_content_hash": graph.content_hash,
        "slices": [
            {
                "id": time_slice.id,
                "start_us": time_slice.start_us,
                "end_us": time_slice.end_us,
                "primitive_ids": list(time_slice.primitive_ids),
            }
            for time_slice in graph.slices
        ],
        "primitives": [
            {
                "id": primitive.id,
                "kind": primitive.kind,
                "channels": list(primitive.channels),
                "channel_count": primitive.channel_count,
                "start_us": primitive.start_us,
                "end_us": primitive.end_us,
                "duration_us": primitive.duration_us,
            }
            for primitive in graph.primitives
        ],
    }


def generate_bundle(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    graphs: list[EffectGraph] = []
    truth: list[dict[str, Any]] = []
    for sample_config in config["samples"]:
        graph = generate_sample(sample_config)
        graphs.append(graph)
        truth.append(truth_entry(graph, sample_config))
    graph_bundle = {
        "schema_version": 1,
        "seed_list": list(config["seed_list"]),
        "samples": [graph.to_dict() for graph in graphs],
    }
    truth_bundle = {
        "schema_version": 1,
        "seed_list": list(config["seed_list"]),
        "samples": truth,
    }
    return graph_bundle, truth_bundle


def bundle_bytes(config: Mapping[str, Any]) -> tuple[bytes, bytes]:
    graph_bundle, truth_bundle = generate_bundle(config)
    return (
        canonical_json_text(graph_bundle).encode("utf-8"),
        canonical_json_text(truth_bundle).encode("utf-8"),
    )


def write_bundle(
    config_path: str | Path,
    graphs_path: str | Path,
    truth_path: str | Path,
) -> tuple[bytes, bytes]:
    config = load_config(config_path)
    graph_bytes, truth_bytes = bundle_bytes(config)
    Path(graphs_path).write_bytes(graph_bytes)
    Path(truth_path).write_bytes(truth_bytes)
    return graph_bytes, truth_bytes


def load_bundle(path: str | Path) -> list[EffectGraph]:
    with Path(path).open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    return [EffectGraph.from_dict(item) for item in bundle["samples"]]


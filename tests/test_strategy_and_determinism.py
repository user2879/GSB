from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from motion_pipeline.config import load_config
from motion_pipeline.samples import load_sequences
from motion_pipeline.stages import (
    decode_step,
    encode_frames,
    estimate_motion_pair,
    predict_from_motion,
)
from motion_pipeline.strategy import select_scale_input

from .helpers import SAMPLE_CONFIG, SAMPLE_SEQUENCES, run_pipeline


def _load_sample_inputs():
    config = load_config(SAMPLE_CONFIG).config
    sequences = load_sequences(SAMPLE_SEQUENCES).sequences
    assert config is not None
    return config, sequences


def test_strategy_changes_across_scale_tiers() -> None:
    config, sequences = _load_sample_inputs()
    tier_one = select_scale_input(config, sequences, 1)
    tier_two = select_scale_input(config, sequences, 2)
    tier_four = select_scale_input(config, sequences, 4)

    assert tier_one.selected.candidate_id == "quality-linear"
    assert tier_two.selected.candidate_id == "balanced-weighted"
    assert tier_four.selected.candidate_id == "throughput-weighted"
    assert tier_one.selected.prediction_path == "linear"
    assert tier_two.selected.prediction_path == "weighted"
    assert len({tier_one.selected.encoding_precision, tier_two.selected.encoding_precision, tier_four.selected.encoding_precision}) == 3
    assert len({tier_one.selected.block_size, tier_two.selected.block_size, tier_four.selected.block_size}) == 3
    assert len({tier_one.selected.quantization_level, tier_two.selected.quantization_level, tier_four.selected.quantization_level}) == 3
    assert tier_one.decision["metrics"]["sequence_count"] == 2
    assert tier_two.decision["metrics"]["sequence_count"] == 4
    assert tier_four.decision["metrics"]["sequence_count"] == 8
    assert tier_one.decision["metrics"]["average_trajectory_count"] == 2.0
    assert tier_two.decision["metrics"]["average_trajectory_count"] == 4.0
    assert tier_four.decision["metrics"]["average_trajectory_count"] == 8.0
    assert tier_one.decision["metrics"]["prediction_steps"] == 2
    assert tier_two.decision["metrics"]["prediction_steps"] == 4
    assert tier_four.decision["metrics"]["prediction_steps"] == 8
    assert len(tier_one.decision["candidates"]) == 3
    assert all("score" in row for row in tier_one.decision["candidates"])
    assert tier_one.decision["reason"].startswith("selected quality-linear")


def test_same_inputs_are_byte_for_byte_deterministic(tmp_path: Path) -> None:
    first = run_pipeline(
        config_path=SAMPLE_CONFIG,
        sequence_path=SAMPLE_SEQUENCES,
        tier=2,
        state_dir=tmp_path / "state-a",
        output_dir=tmp_path / "out-a",
        fresh=True,
    )
    second = run_pipeline(
        config_path=SAMPLE_CONFIG,
        sequence_path=SAMPLE_SEQUENCES,
        tier=2,
        state_dir=tmp_path / "state-b",
        output_dir=tmp_path / "out-b",
        fresh=True,
    )

    assert first.predictions == second.predictions
    assert first.decision_record == second.decision_record
    assert first.cache_ledger == second.cache_ledger
    assert first.failures == second.failures


def test_linear_and_weighted_paths_change_vectors_and_pixels() -> None:
    config, sequences = _load_sample_inputs()
    scaled = select_scale_input(config, sequences, 2)
    sequence = scaled.sequences[0]
    history = sequence.history_frames(config.history_frames, scaled.trajectory_limit)
    encoded = encode_frames(
        history,
        width=config.frame_width,
        height=config.frame_height,
        bits=8,
    )
    frames = [np.asarray(frame, dtype=np.int16) for frame in encoded["frame_pixels"]]
    pairs = [
        estimate_motion_pair(
            frames[index],
            frames[index + 1],
            bits=8,
            block_size=8,
            search_radius=config.motion_model.search_radius,
            pair_index=index,
            sequence_id=sequence.sequence_id,
        )
        for index in range(len(frames) - 1)
    ]
    motion = {
        "sequence_id": sequence.sequence_id,
        "pairs": pairs,
        "blocks_y": 3,
        "blocks_x": 4,
    }
    linear = predict_from_motion(
        motion,
        path="linear",
        horizon=4,
        bits=8,
        linear_gain=config.motion_model.linear_gain,
        damping=config.motion_model.damping,
        quantization_level=2,
    )
    weighted = predict_from_motion(
        motion,
        path="weighted",
        horizon=4,
        bits=8,
        linear_gain=config.motion_model.linear_gain,
        damping=config.motion_model.damping,
        quantization_level=2,
    )
    linear_pixels, _ = decode_step(
        frames[-1],
        np.asarray(linear["steps"][0]["vectors"], dtype=np.int64),
        step=1,
        block_size=8,
        bits=8,
        sequence_id=sequence.sequence_id,
        noise_seed=history[-1]["noise_seed"],
        noise_amplitude=config.noise_layer.amplitude,
        noise_salt=config.noise_layer.seed_salt,
    )
    weighted_pixels, _ = decode_step(
        frames[-1],
        np.asarray(weighted["steps"][0]["vectors"], dtype=np.int64),
        step=1,
        block_size=8,
        bits=8,
        sequence_id=sequence.sequence_id,
        noise_seed=history[-1]["noise_seed"],
        noise_amplitude=config.noise_layer.amplitude,
        noise_salt=config.noise_layer.seed_salt,
    )

    assert linear["steps"][0]["vectors"] != weighted["steps"][0]["vectors"]
    assert (
        linear["steps"][0]["motion_summary"]["digest"]
        != weighted["steps"][0]["motion_summary"]["digest"]
    )
    assert not np.array_equal(linear_pixels, weighted_pixels)


def test_scaled_runs_change_intermediate_outputs(tmp_path: Path) -> None:
    one = run_pipeline(
        config_path=SAMPLE_CONFIG,
        sequence_path=SAMPLE_SEQUENCES,
        tier=1,
        state_dir=tmp_path / "state-1",
        output_dir=tmp_path / "out-1",
        fresh=True,
    )
    two = run_pipeline(
        config_path=SAMPLE_CONFIG,
        sequence_path=SAMPLE_SEQUENCES,
        tier=2,
        state_dir=tmp_path / "state-2",
        output_dir=tmp_path / "out-2",
        fresh=True,
    )
    one_sequence = next(item for item in one.predictions["sequences"] if item["sequence_id"] == "seq-0001")
    two_sequence = next(item for item in two.predictions["sequences"] if item["sequence_id"] == "seq-0001")
    assert one_sequence["motion_vectors"] != two_sequence["motion_vectors"]
    assert one_sequence["predictions"][0]["pixels"] != two_sequence["predictions"][0]["pixels"]
    assert one.decision_record["selected"] != two.decision_record["selected"]


from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from fixedvideo.cache import ArtifactCache
from fixedvideo.config import load_render_config
from fixedvideo.renderer import DeterministicRenderer
from fixedvideo.state import CommandEffect, LatentPatch, apply_command, initial_state


def _text_state(config):
    state = initial_state(config)
    state, _ = apply_command(
        config,
        state,
        {"command_id": "text", "op": "set_initial_image", "path": None},
        command_index=0,
    )
    return state


def _effect(state):
    from fixedvideo.state import CommandEffect

    return CommandEffect(
        tuple(range(state.frame_count)),
        ("encoding", "initialization", "denoise", "decode", "composition"),
        "test render",
    )


def _render(config, state, cache=None, command_id="test"):
    renderer = DeterministicRenderer(config, cache or ArtifactCache())
    result = renderer.render(state, _effect(state), command_id=command_id)
    return renderer, result


def test_no_global_rng_dependency(sample_dir: Path) -> None:
    config = load_render_config(sample_dir / "render_config.json")
    state = _text_state(config)
    _, first = _render(config, state)
    np.random.seed(999)
    np.random.random(5000)
    _, second = _render(config, state)
    assert first.prompt_fp == second.prompt_fp
    assert first.frame_fps == second.frame_fps
    assert first.step_traces == second.step_traces
    assert first.trace_fingerprint == second.trace_fingerprint


def test_seed_changes_image_to_video_after_skip(sample_dir: Path) -> None:
    config = load_render_config(sample_dir / "render_config.json")
    state = initial_state(config)
    _, first = _render(config, state)
    state2, _ = apply_command(
        config,
        state,
        {"command_id": "seed", "op": "set_seed", "seed": state.seed + 1},
        command_index=0,
    )
    _, second = _render(config, state2)
    assert first.frame_fps != second.frame_fps
    assert first.trace_fingerprint != second.trace_fingerprint


def test_guidance_reuses_encoding_and_initialization(sample_dir: Path) -> None:
    config = load_render_config(sample_dir / "render_config.json")
    state = _text_state(config)
    renderer, first = _render(config, state)
    state2, effect = apply_command(
        config,
        state,
        {"command_id": "g", "op": "set_guidance", "guidance": state.guidance + 0.5},
        command_index=1,
    )
    second = renderer.render(state2, effect, command_id="g")
    status_by_stage = {}
    for record in second.audit:
        status_by_stage.setdefault(record["stage"], []).append(record["status"])
    assert set(status_by_stage["encoding"]) == {"reuse"}
    assert set(status_by_stage["initialization"]) == {"reuse"}
    assert status_by_stage["denoise"].count("recompute") == state.frame_count * state.steps
    assert status_by_stage["decode"].count("recompute") == state.frame_count
    assert first.frame_fps != second.frame_fps


def test_patch_invalidates_only_its_step_downstream_and_frame(sample_dir: Path) -> None:
    config = load_render_config(sample_dir / "render_config.json")
    state = _text_state(config)
    renderer, _ = _render(config, state)
    patch = LatentPatch(
        patch_id="p",
        step=6,
        frame=1,
        x=1,
        y=1,
        width=2,
        height=2,
        delta=0.25,
    )
    state2 = replace(state, patches=(patch,), version=state.version + 1, command_cursor=2)
    effect = CommandEffect((1,), ("denoise", "decode", "composition"), "patch test")
    second = renderer.render(state2, effect, command_id="patch")
    denoise = [record for record in second.audit if record["stage"] == "denoise"]
    for record in denoise:
        if record["frame"] == 1 and record["step"] >= 6:
            assert record["status"] == "recompute"
        else:
            assert record["status"] == "reuse"
    decode = [record for record in second.audit if record["stage"] == "decode"]
    assert [record["status"] for record in decode] == ["reuse", "recompute"] + ["reuse"] * 4


def test_chunk_rows_change_partition_and_all_denoise_reuses(sample_dir: Path) -> None:
    config = load_render_config(sample_dir / "render_config.json")
    state = _text_state(config)
    renderer, first = _render(config, state)
    state2, effect = apply_command(
        config,
        state,
        {"command_id": "c", "op": "set_chunk_rows", "chunk_rows": 4},
        command_index=1,
    )
    second = renderer.render(state2, effect, command_id="c")
    denoise = [record for record in second.audit if record["stage"] == "denoise"]
    assert set(record["status"] for record in denoise) == {"reuse"}
    assert first.frame_traces[0]["blocks"] != second.frame_traces[0]["blocks"]
    assert first.composition_fp != second.composition_fp


def test_quantization_change_changes_pixels(sample_dir: Path) -> None:
    config = load_render_config(sample_dir / "render_config.json")
    state = _text_state(config)
    renderer, first = _render(config, state)
    state2, effect = apply_command(
        config,
        state,
        {"command_id": "q", "op": "set_quantization_level", "quantization_level": "q4"},
        command_index=1,
    )
    second = renderer.render(state2, effect, command_id="q")
    first_pixels = renderer.cache.decodes[first.frame_keys[0]].pixels
    second_pixels = renderer.cache.decodes[second.frame_keys[0]].pixels
    assert first.frame_fps != second.frame_fps
    assert not np.array_equal(first_pixels, second_pixels)

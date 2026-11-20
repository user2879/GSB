"""Requirement 6: incremental edits must equal a one-shot full decode."""

import dataclasses
import os


from asr_stream_editor.audio import load_wav, plan_chunks
from asr_stream_editor.full_decode import full_decode
from asr_stream_editor.runner import load_resources
from asr_stream_editor.snapshot import SnapshotStore
from asr_stream_editor.engine import StreamingEngine

COMPARED_FIELDS = (
    "transcript", "frame_labels", "frame_owner", "alignment_conflicts",
    "failures", "stable_prefix", "mutable_tail", "stable_until_frame",
    "boundary_runs", "text", "stable_prefix_text",
)


def _engine(samples_dir, work_dir, fault=None, budget_override=None):
    pm, ac, sc, clips, _ = load_resources(samples_dir)
    if budget_override is not None:
        sc = dataclasses.replace(sc, redecode_budget_ms=budget_override)
    clip = clips[0]
    store = SnapshotStore(os.path.join(work_dir, "snap", clip.source))
    return StreamingEngine(clip, pm, ac, sc, store, fault or {}), pm, ac, sc, clip


def _drain(engine, commands, replacements=None):
    responses = []
    pending = True
    while pending:
        pending = False
        for command in commands:
            replacement = None
            if command.get("command") == "replace_chunk":
                replacement = replacements[command["chunk_index"]]
            response = engine.process_command(command, replacement)
            responses.append(response.to_dict())
            if response.code is not None:
                pending = True
        commands = [{"command": "resume"}]
    return responses


def _final_engine_state(engine):
    frame_shift_ms = engine.acoustic.frame_shift_ms
    from asr_stream_editor.stability import split_tokens
    prefix, tail = split_tokens(engine.merged.tokens,
                                engine.stability.stable_until_frame)
    return {
        "transcript": [token.to_dict(frame_shift_ms) for token in engine.merged.tokens],
        "text": "".join(token.word for token in engine.merged.tokens),
        "frame_labels": list(engine.merged.frame_labels),
        "frame_owner": list(engine.merged.frame_owner),
        "alignment_conflicts": list(engine.merged.conflicts),
        "failures": [record.to_dict() for record in engine.failures],
        "stable_prefix": [token.to_dict(frame_shift_ms) for token in prefix],
        "stable_prefix_text": "".join(token.word for token in prefix),
        "mutable_tail": [token.to_dict(frame_shift_ms) for token in tail],
        "stable_until_frame": engine.stability.stable_until_frame,
        "boundary_runs": engine.stability.boundary_runs,
    }


def _assert_reference_equal(engine_state, reference):
    for field in COMPARED_FIELDS:
        assert engine_state[field] == reference[field], f"field differs: {field}"
    # word-by-word coverage including every per-word frame and score field
    assert len(engine_state["transcript"]) == len(reference["transcript"])
    for index, (left, right) in enumerate(zip(engine_state["transcript"],
                                              reference["transcript"])):
        assert left == right, f"word {index} differs"


def test_full_edit_sequence_matches_one_shot_decode(samples_dir, work_dir):
    engine, pm, ac, sc, clip = _engine(samples_dir, work_dir, budget_override=100000)
    chunks = plan_chunks(clip.n_samples, sc.chunk_ms, sc.overlap_ms)
    target = chunks[2]
    region_len = target.own_end_sample - target.own_start_sample
    replacement_clip = load_wav(os.path.join(
        samples_dir, "replacements", "replacement_clip01_chunk02.wav"))
    replacement = replacement_clip.samples[:region_len]
    edited_samples = clip.samples.copy()
    edited_samples[target.own_start_sample:target.own_end_sample] = replacement

    commands = [
        {"command": "initial_decode"},
        {"command": "replace_chunk", "chunk_index": 2},
        {"command": "set_overlap", "overlap_ms": 120},
        {"command": "set_stable_frames", "min_consistent_frames": 4},
        {"command": "set_backoff", "backoff_scale": 0.5},
        {"command": "finalize"},
    ]
    responses = _drain(engine, commands, replacements={2: replacement})

    reference = full_decode(
        clip, edited_samples, pm, ac, sc, overlap_ms=120,
        min_consistent_frames=4, backoff_scale=0.5)
    _assert_reference_equal(_final_engine_state(engine), reference)

    # every response exposes the required budget-bounded summary fields
    for response in responses:
        assert "stable_prefix_text" in response
        assert "mutable_tail_frames" in response
        assert "affected_chunks" in response
        assert "progress" in response


def test_each_individual_edit_alone_matches_reference(samples_dir, work_dir):
    scenarios = [
        [{"command": "set_overlap", "overlap_ms": 120}],
        [{"command": "set_stable_frames", "min_consistent_frames": 5}],
        [{"command": "set_backoff", "backoff_scale": 0.25}],
    ]
    for scenario_index, commands in enumerate(scenarios):
        engine, pm, ac, sc, clip = _engine(
            samples_dir, os.path.join(work_dir, f"case{scenario_index}"),
            budget_override=100000)
        _drain(engine, [{"command": "initial_decode"}] + commands)
        overlap = commands[0].get("overlap_ms", sc.overlap_ms)
        stable = commands[0].get("min_consistent_frames", sc.min_consistent_frames)
        scale = commands[0].get("backoff_scale", 1.0)
        reference = full_decode(clip, clip.samples, pm, ac, sc,
                                overlap_ms=overlap,
                                min_consistent_frames=stable,
                                backoff_scale=scale)
        _assert_reference_equal(_final_engine_state(engine), reference)


def test_reuse_and_redecode_ranges_are_reported(samples_dir, work_dir):
    engine, pm, ac, sc, clip = _engine(samples_dir, work_dir, budget_override=100000)
    engine.process_command({"command": "initial_decode"})
    response = engine.process_command(
        {"command": "replace_chunk", "chunk_index": 2},
        replacement_samples=_replacement(samples_dir, clip, sc, 2))
    payload = response.to_dict()
    assert 2 in payload["affected_chunks"]
    assert payload["reused_sample_ranges"]
    assert payload["redecoded_sample_ranges"]
    reused = {item["chunk_index"] for item in payload["reused_sample_ranges"]}
    redecoded = {item["chunk_index"] for item in payload["redecoded_sample_ranges"]}
    assert reused.isdisjoint(redecoded)
    assert reused | redecoded == set(range(len(engine.chunks)))


def _replacement(samples_dir, clip, sc, chunk_index):
    chunks = plan_chunks(clip.n_samples, sc.chunk_ms, sc.overlap_ms)
    target = chunks[chunk_index]
    region_len = target.own_end_sample - target.own_start_sample
    replacement_clip = load_wav(os.path.join(
        samples_dir, "replacements", "replacement_clip01_chunk02.wav"))
    return replacement_clip.samples[:region_len]

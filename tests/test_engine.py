from __future__ import annotations

from pathlib import Path

from haptic_diag.compiler import compile_graph
from haptic_diag.dependency import PROTOCOL_TRIGGER
from haptic_diag.engine import apply_update, initial_state
from haptic_diag.generator import load_config, generate_sample
from haptic_diag.quantization import DEFAULT_QUANT_PROFILE

ROOT = Path(__file__).resolve().parents[1]


def _state(sample_index: int = 1):
    config = load_config(ROOT / "configs" / "samples.json")
    graph = generate_sample(config["samples"][sample_index])
    return initial_state(graph, 2)


def test_quantization_equivalent_frequency_and_amplitude_changes_reuse_everything() -> None:
    state = _state()
    primitive = state.graph.primitives[0]
    changed, result = apply_update(
        state,
        {
            primitive.id: {
                "frequency_hz": primitive.frequency_hz + 1e-7,
                "amplitude": primitive.amplitude + 1e-9,
            }
        },
    )
    assert result.recompiled_frames == ()
    assert result.invalidated_slices == ()
    assert result.invalidated_digests == ()
    assert {change.parameter for change in result.ignored_changes} == {
        "frequency_hz",
        "amplitude",
    }
    assert [frame.content_hash for frame in changed.frames] == [
        frame.content_hash for frame in state.frames
    ]


def test_effective_change_recomputes_only_hash_changed_frames_and_matches_full_compile() -> None:
    state = _state()
    primitive = state.graph.primitives[0]
    changed, update = apply_update(
        state,
        {primitive.id: {"frequency_hz": primitive.frequency_hz + 10.0}},
    )
    assert update.recompiled_frames
    assert update.invalidated_frames == update.recompiled_frames
    assert len(update.recompiled_frames) < state.graph.frame_count
    assert update.invalidated_digests
    assert update.reused_nodes
    full = compile_graph(state.graph.with_primitive_changes(
        {primitive.id: {"frequency_hz": primitive.frequency_hz + 10.0}}
    ), 2)
    assert [frame.content_hash for frame in changed.frames] == [
        frame.content_hash for frame in full.frames
    ]


def test_invalidated_frames_are_limited_to_primitive_interval_and_bound_channels() -> None:
    state = _state()
    primitive = state.graph.primitive_by_id["multi_left_grip_sine"]
    changed, update = apply_update(
        state,
        {primitive.id: {"frequency_hz": primitive.frequency_hz + 10.0}},
    )
    changed_set = set(update.recompiled_frames)
    for frame in state.frames:
        inside = primitive.start_us <= frame.timestamp_us < primitive.end_us
        if not inside:
            assert frame.seq not in changed_set
            assert frame.content_hash == changed.frames_by_seq[frame.seq].content_hash
        elif frame.seq in changed_set:
            assert "left_grip" in frame.channel_by_name
    for record in update.reused_nodes:
        if record.node_kind == "frame":
            assert record.source == state.reuse_source
            assert record.version_stamp.startswith("v2:")


def test_dependency_edges_declare_parameter_and_protocol_keys() -> None:
    state = _state()
    parameter_edges = [
        edge for edge in state.dependency_graph.edges if "frequency_hz" in edge.trigger_keys
    ]
    protocol_edges = [
        edge for edge in state.dependency_graph.edges if edge.protocol_key == PROTOCOL_TRIGGER
    ]
    assert parameter_edges
    assert protocol_edges
    assert all(edge.trigger_keys for edge in parameter_edges + protocol_edges)


def test_protocol_switch_reencodes_all_frames_and_archives_old_ledger() -> None:
    state = _state()
    primitive = state.graph.primitives[0]
    changed, update = apply_update(
        state,
        {primitive.id: {"amplitude": min(1.0, primitive.amplitude + 0.05)}},
    )
    assert changed.ledger.entries
    migrated, migration = apply_update(changed, {}, new_protocol_version=1)
    assert migration.protocol_changed
    assert migration.invalidated_frames == tuple(range(migrated.graph.frame_count))
    assert migration.ledger_archived
    assert migrated.protocol_version == 1
    assert migrated.ledger.protocol_version == 1
    assert migrated.ledger_history
    assert all(not ledger.reusable for ledger in migrated.ledger_history)
    assert migrated.ledger_history[-1].archived_reason
    assert all(entry.protocol_version == 1 for entry in migrated.ledger.entries)
    full = compile_graph(migrated.graph, 1)
    assert [frame.content_hash for frame in migrated.frames] == [
        frame.content_hash for frame in full.frames
    ]


def test_duration_change_invalidates_only_affected_timeline_and_matches_full_compile() -> None:
    state = _state(2)
    primitive = state.graph.primitive_by_id["cross_pulse"]
    changed, update = apply_update(
        state,
        {primitive.id: {"duration_us": primitive.duration_us + 500_000}},
    )
    assert update.recompiled_frames
    full = compile_graph(changed.graph, 2)
    assert [frame.content_hash for frame in changed.frames] == [
        frame.content_hash for frame in full.frames
    ]
    assert max(update.invalidated_frames) < changed.graph.frame_count


def test_declared_quant_profile_codes_are_exact() -> None:
    profile = DEFAULT_QUANT_PROFILE
    value = 0.123456789
    assert profile.same_amplitude_code(value, value + 1e-9)
    frequency = 123.456789
    assert profile.same_frequency_code(frequency, frequency + 1e-9)


def test_v1_frequency_change_is_ignored_until_protocol_switch_reencodes() -> None:
    config = load_config(ROOT / "configs" / "samples.json")
    graph = generate_sample(config["samples"][2])
    state = initial_state(graph, 1)
    primitive = state.graph.primitives[0]
    changed, result = apply_update(
        state,
        {primitive.id: {"frequency_hz": primitive.frequency_hz + 25.0}},
    )
    assert result.recompiled_frames == ()
    assert result.ignored_changes[0].reason == "legacy_protocol_has_no_frequency_field"
    assert [frame.encode() for frame in changed.frames] == [frame.encode() for frame in state.frames]
    upgraded, migration = apply_update(changed, {}, new_protocol_version=2)
    assert migration.invalidated_frames == tuple(range(upgraded.graph.frame_count))
    full = compile_graph(upgraded.graph, 2)
    assert [frame.content_hash for frame in upgraded.frames] == [
        frame.content_hash for frame in full.frames
    ]

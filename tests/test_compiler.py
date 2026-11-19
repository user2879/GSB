from __future__ import annotations

from pathlib import Path

from haptic_diag.compiler import compile_graph
from haptic_diag.generator import load_config, generate_sample

ROOT = Path(__file__).resolve().parents[1]


def _graph(index: int = 2):
    config = load_config(ROOT / "configs" / "samples.json")
    return generate_sample(config["samples"][index])


def test_compilation_is_byte_deterministic_and_error_report_is_identical() -> None:
    graph = _graph()
    first = compile_graph(graph, 2)
    second = compile_graph(graph, 2)
    assert [frame.encode() for frame in first.frames] == [frame.encode() for frame in second.frames]
    assert first.report.to_dict() == second.report.to_dict()
    assert first.digests == second.digests


def test_declared_quantization_bounds_hold_for_all_samples() -> None:
    config = load_config(ROOT / "configs" / "samples.json")
    for sample_config in config["samples"]:
        graph = generate_sample(sample_config)
        result = compile_graph(graph, 2)
        assert result.report.max_amplitude_error <= result.report.amplitude_error_bound + 1e-15
        assert result.report.max_frequency_error <= result.report.frequency_error_bound + 1e-15
        assert result.report.profile["amplitude_step"] > 0
        assert result.report.profile["frequency_step"] > 0
        assert result.report.error_bound_satisfied


def test_frames_have_contiguous_sequences_and_microsecond_timestamps() -> None:
    graph = _graph()
    result = compile_graph(graph, 2)
    assert [frame.seq for frame in result.frames] == list(range(graph.frame_count))
    for frame in result.frames:
        assert frame.timestamp_us == frame.seq * graph.frame_period_us
    assert len(result.digests) == (graph.frame_count + graph.batch_size - 1) // graph.batch_size


def test_v1_and_v2_have_distinct_but_deterministic_encodings() -> None:
    graph = _graph()
    v1 = compile_graph(graph, 1)
    v2 = compile_graph(graph, 2)
    assert len(v1.frames) == len(v2.frames)
    assert any(a.encode() != b.encode() for a, b in zip(v1.frames, v2.frames, strict=True))
    assert all(frame.protocol_version == 1 for frame in v1.frames)
    assert all(frame.protocol_version == 2 for frame in v2.frames)


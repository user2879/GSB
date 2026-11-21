from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from timbre_lab.audio import scan_wavs
from timbre_lab.config import load_config
from timbre_lab.engine import IncrementalEngine
from timbre_lab.errors import E_PREVIEW_BUDGET_PENDING
from timbre_lab.pipeline import compare_incremental_with_full
from timbre_lab.segmentation import build_document
from timbre_lab.snapshot import SnapshotManager


def _run_one(engine: IncrementalEngine, command: dict) -> dict:
    response = engine.submit(command)
    if response["status"] == "pending":
        assert response["code"] == E_PREVIEW_BUDGET_PENDING
        engine.wait_for_idle()
        completed = engine.jobs[-1].result
        assert completed is not None
        return completed
    return response


def test_incremental_matches_full_and_reuses_unaffected_work(
    demo_workspace: dict[str, Path], tmp_path: Path
) -> None:
    config = load_config(demo_workspace["config"])
    commands = json.loads(demo_workspace["commands"].read_text(encoding="utf-8"))
    scan = scan_wavs(demo_workspace["samples"])
    document, _, failures = build_document(scan, config)
    engine = IncrementalEngine(
        config,
        document,
        commands,
        tmp_path / "run",
        failures=failures,
        commands_base_dir=demo_workspace["root"],
    )
    try:
        assert engine.last_semantic is not None
        assert engine.last_semantic["full_rebuild_comparison"]["equal"] is True
        initial_speaker_segments = {
            item["speaker"]: len(item["segment_ids"]) for item in engine.last_semantic["speaker_vectors"]
        }
        initial_speaker_keys = {
            item["speaker"]: item["cache_key"] for item in engine.last_semantic["speaker_vectors"]
        }
        audits = []
        split_speaker_segments: dict[str, int] = {}
        split_speaker_keys: dict[str, str] = {}
        for command in commands:
            result = _run_one(engine, command)
            assert result["status"] == "complete"
            assert result["incremental_vs_full_equal"] is True
            audits.append((command["op"], engine.last_semantic["cache_audit"]))
            if len(audits) == 1:
                split_speaker_segments = {
                    item["speaker"]: len(item["segment_ids"]) for item in engine.last_semantic["speaker_vectors"]
                }
                split_speaker_keys = {
                    item["speaker"]: item["cache_key"] for item in engine.last_semantic["speaker_vectors"]
                }

        # Splitting a segment must change the speaker contributor set and aggregate key;
        # the subsequent explicit merge restores the original segment cardinality.
        assert sum(split_speaker_segments.values()) == sum(initial_speaker_segments.values()) + 1
        assert any(split_speaker_keys[speaker] != initial_speaker_keys[speaker] for speaker in initial_speaker_keys)
        assert len(engine.document.segments) == sum(initial_speaker_segments.values())

        split_audit = audits[0][1]
        assert len(split_audit["feature"]["recomputed"]) == 2
        assert len(split_audit["feature"]["reused"]) == 7

        label_audit = audits[2][1]
        assert len(label_audit["feature"]["recomputed"]) == 0
        assert len(label_audit["description"]["recomputed"]) == 0
        assert len(label_audit["assignment"]["recomputed"]) >= 2
        assert len(label_audit["assignment"]["reused"]) >= 3

        quant_audit = audits[-1][1]
        assert len(quant_audit["feature"]["reused"]) == 8
        assert len(quant_audit["description"]["recomputed"]) == 8
        assert engine.command_cursor == len(commands)
        comparison = compare_incremental_with_full(
            engine.last_semantic,
            engine.document,
            config,
            engine.overrides,
            engine.failures,
        )
        assert comparison == {"equal": True, "differences": []}
    finally:
        engine.close()


def test_preview_budget_returns_pending_without_blocking(demo_workspace: dict[str, Path], tmp_path: Path) -> None:
    config = load_config(demo_workspace["config"])
    config.preview_budget_ms = 0.0
    commands = json.loads(demo_workspace["commands"].read_text(encoding="utf-8"))
    scan = scan_wavs(demo_workspace["samples"])
    document, _, failures = build_document(scan, config)
    engine = IncrementalEngine(
        config,
        document,
        commands,
        tmp_path / "run",
        failures=failures,
        commands_base_dir=demo_workspace["root"],
    )
    try:
        started = time.perf_counter()
        response = engine.submit(commands[0])
        elapsed = time.perf_counter() - started
        assert response["status"] == "pending"
        assert response["code"] == E_PREVIEW_BUDGET_PENDING
        assert response["preview"]["progress"]["phase"] in {"queued", "applying", "feature", "description", "assignment", "speaker"}
        assert response["preview"]["affected_segments"]
        assert response["preview"]["affected_segments_source"] in {"forecast", "applied_plan"}
        assert elapsed < 0.1
        engine.wait_for_idle()
        assert engine.jobs[-1].result["status"] == "complete"
    finally:
        engine.close()


def test_invalid_command_is_isolated_and_next_command_still_runs(
    demo_workspace: dict[str, Path], tmp_path: Path
) -> None:
    config = load_config(demo_workspace["config"])
    normal_commands = json.loads(demo_workspace["commands"].read_text(encoding="utf-8"))
    commands = [
        {"id": "bad-quant", "op": "set_quantization", "dimension": "f0_mean", "bit_width": 99},
        normal_commands[0],
    ]
    scan = scan_wavs(demo_workspace["samples"])
    document, _, failures = build_document(scan, config)
    initial_ids = [segment.segment_id for segment in document.ordered_segments()]
    engine = IncrementalEngine(
        config,
        document,
        commands,
        tmp_path / "run",
        failures=failures,
        commands_base_dir=demo_workspace["root"],
    )
    try:
        failed = _run_one(engine, commands[0])
        assert failed["status"] == "failed"
        assert failed["code"] == "E_QUANTIZATION_LEVEL"
        assert failed["preview"]["command_cursor"] == 1
        assert [segment.segment_id for segment in engine.document.ordered_segments()] == initial_ids
        assert engine.failures[-1]["code"] == "E_QUANTIZATION_LEVEL"
        completed = _run_one(engine, commands[1])
        assert completed["status"] == "complete"
        assert completed["incremental_vs_full_equal"] is True
        assert len(engine.document.segments) == len(initial_ids) + 1
    finally:
        engine.close()


def _cli_run(demo_workspace: dict[str, Path], output: Path, *, config: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "timbre_lab",
            "run",
            "--samples",
            str(demo_workspace["samples"]),
            "--config",
            str(config or demo_workspace["config"]),
            "--commands",
            str(demo_workspace["commands"]),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_snapshot_resume_and_stale_rejection(demo_workspace: dict[str, Path], tmp_path: Path) -> None:
    output = tmp_path / "run"
    first = _cli_run(demo_workspace, output)
    assert first.returncode == 0, first.stderr
    config = load_config(demo_workspace["config"])
    commands = json.loads(demo_workspace["commands"].read_text(encoding="utf-8"))
    restored = SnapshotManager(output).load(config, commands)
    assert restored is not None
    assert restored.command_cursor == len(commands)
    snapshot_dirs_before = sorted((output / "snapshots").glob("v*"))

    second = _cli_run(demo_workspace, output)
    assert second.returncode == 0, second.stderr
    summary = json.loads(second.stdout)
    assert summary["command_cursor"] == len(commands)
    snapshot_dirs_after = sorted((output / "snapshots").glob("v*"))
    assert len(snapshot_dirs_after) == len(snapshot_dirs_before)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["resume"]["restored"] is True
    assert report["incremental_full_equal"] is True

    stale_config_value = json.loads(demo_workspace["config"].read_text(encoding="utf-8"))
    stale_config_value["preview_budget_ms"] = 201.0
    stale_config = tmp_path / "stale-config.json"
    stale_config.write_text(json.dumps(stale_config_value), encoding="utf-8")
    third = _cli_run(demo_workspace, output, config=stale_config)
    assert third.returncode == 2
    fatal = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert fatal["failure_codes"] == ["E_SNAPSHOT_STALE"]


def test_sigterm_finishes_current_command_and_exits_zero(demo_workspace: dict[str, Path], tmp_path: Path) -> None:
    output = tmp_path / "signal-run"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "timbre_lab",
            "run",
            "--samples",
            str(demo_workspace["samples"]),
            "--config",
            str(demo_workspace["config"]),
            "--commands",
            str(demo_workspace["commands"]),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    current = output / "snapshots" / "CURRENT"
    deadline = time.perf_counter() + 15.0
    while not current.exists() and process.poll() is None and time.perf_counter() < deadline:
        time.sleep(0.05)
    assert current.exists(), "initial snapshot did not become visible before signal"
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=60)
    assert process.returncode == 0, (stdout, stderr)
    report_path = output / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] in {"stopped_by_signal", "complete"}
    assert (output / "snapshots" / "CURRENT").exists()


def test_bad_audio_in_run_is_isolated(demo_workspace: dict[str, Path], tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "voice_dialogue.wav").write_bytes((demo_workspace["samples"] / "voice_dialogue.wav").read_bytes())
    (samples / "bad.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEbroken")
    config = load_config(demo_workspace["config"])
    scan = scan_wavs(samples)
    document, _, failures = build_document(scan, config)
    assert len(document.segments) > 0
    assert [failure["code"] for failure in failures] == ["E_AUDIO_FORMAT"]

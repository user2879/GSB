from __future__ import annotations

import json
from pathlib import Path

from voice_diag.engine import DiagnosticEngine, run_diagnostics
from voice_diag.fingerprint import sha256_json
from voice_diag.scenario import run_acceptance_scenario
from voice_diag.synth import WORD_PROFILES, synthesize_word_audio

from .conftest import update_config


def test_single_template_replacement_reuses_unaffected_artifacts_and_matches_full(demo: Path) -> None:
    engine = DiagnosticEngine(demo / "config.json", demo / "state-incremental")
    baseline, _ = engine.run(mode="auto")
    profile = next(item for item in WORD_PROFILES if item.command_word == "MA")
    synthesize_word_audio(
        demo / "templates" / "MA" / "T02.wav",
        profile,
        seed=20261126,
        variant_key="single-template-replacement",
    )
    incremental, incremental_report = engine.run(mode="auto")
    full, _ = DiagnosticEngine(demo / "config.json", demo / "state-full").run(mode="full")
    assert incremental == full
    assert sha256_json(incremental) == sha256_json(full)
    assert incremental != baseline
    events = incremental_report["cache_events"]
    acoustic_reuse = [
        event for event in events if event["stage"] == "template-acoustic" and event["status"] == "reuse"
    ]
    acoustic_compute = [
        event for event in events if event["stage"] == "template-acoustic" and event["status"] == "computed"
    ]
    assert len(acoustic_reuse) >= 17
    assert len(acoustic_compute) == 1
    assert any(event["stage"] == "confusion-matrix-cell" and event["status"] == "reuse" for event in events)


def test_complete_edit_sequence_has_exact_incremental_full_equality(demo: Path) -> None:
    root = demo.parent / "scenario"
    summary = run_acceptance_scenario(root, seed=20261126)
    assert summary["canonical_results_exactly_equal"] is True
    assert summary["final_incremental_result_sha256"] == summary["final_full_result_sha256"]
    assert summary["degenerate_words"] == ["NA"]
    assert summary["incremental_reuse_hit_count"] > 0
    assert summary["failure_count_incremental"] == 0
    assert summary["matrix_consistency"]["consistent"] is True


def test_budget_pending_is_stable_and_full_matches_incremental(demo: Path) -> None:
    engine = DiagnosticEngine(demo / "config.json", demo / "state-incremental")
    engine.run(mode="auto")
    update_config(
        demo,
        lambda config: (
            config["registration"].__setitem__("comparison_budget", 100),
            config["registration"].__setitem__("max_workers", 2),
        ),
    )
    incremental, report = engine.run(mode="auto")
    full, full_report = DiagnosticEngine(demo / "config.json", demo / "state-full").run(mode="full")
    assert incremental == full
    assert incremental["pending_queue"]
    assert [item["stable_sort_key"] for item in incremental["pending_queue"]] == sorted(
        item["stable_sort_key"] for item in incremental["pending_queue"]
    )
    assert all(item["code"] == "E_BUDGET_PENDING" for item in incremental["pending_queue"])
    assert any(item["code"] == "E_BUDGET_PENDING" for item in incremental["failures"])
    assert incremental["budget"]["stable_task_order_preserved"] is True
    assert all(item["peak_concurrency"] <= 2 for item in report["phase_peak_concurrency"])
    assert all(item["peak_concurrency"] <= 2 for item in full_report["phase_peak_concurrency"])
    assert incremental["budget"]["consumed_points"] <= 100


def test_stage_snapshot_is_atomically_pointed_to_and_contains_pending_cursor(demo: Path) -> None:
    result, report = run_diagnostics(
        demo / "config.json",
        demo / "state",
        result_path=demo.parent / "result.json",
        report_path=demo.parent / "report.json",
        mode="auto",
    )
    state = demo / "state"
    current = json.loads((state / "current.json").read_text(encoding="utf-8"))
    snapshot_dir = state / current["snapshot_dir"]
    assert snapshot_dir.is_dir()
    assert (snapshot_dir / "result.json").is_file()
    assert (snapshot_dir / "pending-queue.json").is_file()
    assert (snapshot_dir / "completed-cursors.json").is_file()
    assert (snapshot_dir / "stage-conclusions.json").is_file()
    assert not any(path.name.startswith(".staging-") for path in (state / "snapshots").iterdir())
    assert current["snapshot_schema"] == 1
    assert current["config_version"] == "demo-v1"
    assert current["input_digest"] == result["corpus"]["input_digest"]
    assert report["snapshot"]["state_version"] == current["state_version"]


def test_repeat_run_is_pure_persistent_cache_reuse(demo: Path) -> None:
    engine = DiagnosticEngine(demo / "config.json", demo / "state")
    first, _ = engine.run(mode="auto")
    second, report = engine.run(mode="auto")
    assert first == second
    assert report["cache"]["miss_count"] == 0
    assert report["cache"]["computed_count"] == 0
    assert report["cache"]["reuse_hit_count"] > 0


def test_label_correction_invalidates_recognition_even_when_audio_is_unchanged(demo: Path) -> None:
    engine = DiagnosticEngine(demo / "config.json", demo / "state-incremental")
    engine.run(mode="auto")
    labels_path = demo / "labels.jsonl"
    rows = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row["clip_id"] == "pos_lin_01":
            row["true_command"] = "LING"
            row["confusion_with"] = "LIN"
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    incremental, _ = engine.run(mode="auto")
    full, _ = DiagnosticEngine(demo / "config.json", demo / "state-full").run(mode="full")
    assert incremental == full
    corrected = next(item for item in incremental["conclusions"] if item["clip_id"] == "pos_lin_01")
    assert corrected["true_command"] == "LING"
    assert corrected["label_confusion_with"] == "LIN"

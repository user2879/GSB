from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from voice_diag.audio import read_wav, write_pcm16_wav
from voice_diag.pipeline import DiagnosticPipeline, PipelineOptions
from voice_diag.synthesize import generate_demo


ROOT = Path(__file__).resolve().parents[1]


def _run(sample: Path, out: Path, **options) -> dict:
    options.setdefault("max_batches", 32)
    return DiagnosticPipeline(
        sample / "config.json",
        sample / "manifest.json",
        out,
        options=PipelineOptions(**options),
    ).run()


def _state(out: Path) -> dict:
    pointer = json.loads((out / "CURRENT.json").read_text(encoding="utf-8"))
    return json.loads((out / pointer["snapshot"] / "state.json").read_text(encoding="utf-8"))


def _copy_sample(tmp_path: Path, name: str = "sample") -> Path:
    target = tmp_path / name
    generate_demo(target)
    return target


def _mutate_audio(path: Path) -> None:
    wav = read_wav(path)
    pcm = wav.pcm.astype(np.int32)
    pcm = np.clip(pcm + ((np.arange(pcm.size) % 19) - 9) * 2, -32768, 32767).astype("<i2")
    write_pcm16_wav(path, pcm)


def test_end_to_end_dual_receipts_overlap_and_worker_determinism(tmp_path: Path) -> None:
    sample = _copy_sample(tmp_path)
    one_worker = _run(sample, tmp_path / "out-1", max_workers=1)
    four_workers = _run(sample, tmp_path / "out-4", max_workers=4)
    assert one_worker["status"] == "complete"
    assert four_workers["status"] == "complete"
    assert one_worker["final_digest"] == four_workers["final_digest"]
    report = one_worker["report"]
    assert report["receipts"]["input_a1_count"] == 4
    assert report["receipts"]["input_a2_count"] == 4
    assert report["receipts"]["input_normalized_equal"] is True
    assert report["receipts"]["generated_a1_count"] == 4
    assert report["receipts"]["generated_a2_count"] == 4
    assert report["receipts"]["generated_roundtrip_ok"] is True
    assert report["overlap_frame_count"] > 0
    assert report["boundary_frames"] == []
    assert report["active_frame_count"] > 0
    assert report["command_conclusions"]["accepted_count"] >= 4
    state = _state(tmp_path / "out-4")
    attribution = state["ledger"]["entries"]["attribution:W000010"]["payload"]
    assert {source["interval_id"] for source in attribution["sources"]} >= {"I001", "I002"}
    assert len(attribution["sources"]) >= 2


def test_second_run_reuses_windows_and_keeps_same_conclusion(tmp_path: Path) -> None:
    sample = _copy_sample(tmp_path)
    out = tmp_path / "out"
    first = _run(sample, out)
    second = _run(sample, out)
    assert first["final_digest"] == second["final_digest"]
    assert second["report"]["cache"]["reuse_count"] >= 120
    assert second["report"]["cache"]["invalidation_count"] == 0


def test_single_interval_audio_replacement_invalidates_only_referencing_windows(tmp_path: Path) -> None:
    sample = _copy_sample(tmp_path)
    incremental_out = tmp_path / "out-incremental"
    baseline = _run(sample, incremental_out)
    _mutate_audio(sample / "audio" / "F002.wav")
    incremental = _run(sample, incremental_out)
    full = _run(sample, tmp_path / "out-full-modified")
    assert incremental["final_digest"] == full["final_digest"] != baseline["final_digest"]
    report = incremental["report"]
    assert report["cache"]["reuse_count"] > 0
    reasons = {
        reason
        for entry in report["cache"]["invalidation_entries"]
        for reason in entry["reasons"]
    }
    assert any(reason.startswith("audio_content_changed:") for reason in reasons)
    feature_invalidations = [
        entry
        for entry in report["cache"]["invalidation_entries"]
        if entry["stage"] == "features"
    ]
    assert feature_invalidations
    referenced = {ref for entry in feature_invalidations for ref in entry["new_refs"]["interval_ids"]}
    assert "I002" in referenced
    assert all("I002" in entry["new_refs"]["interval_ids"] for entry in feature_invalidations)
    assert any(
        "I002" not in entry["refs"]["interval_ids"]
        for entry in report["cache"]["reuse_entries"]
        if entry["stage"] == "features"
    )


def test_transcript_offset_correction_invalidates_dependent_windows_and_matches_full(tmp_path: Path) -> None:
    sample = _copy_sample(tmp_path)
    out = tmp_path / "out-transcript"
    baseline = _run(sample, out)
    manifest_path = sample / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    interval = next(item for item in raw["intervals"] if item["id"] == "I002")
    interval["chars"][1]["start_sample"] += 37
    interval["chars"][1]["end_sample"] += 37
    manifest_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    incremental = _run(sample, out)
    full = _run(sample, tmp_path / "out-transcript-full")
    assert incremental["final_digest"] == full["final_digest"] != baseline["final_digest"]
    reasons = {
        reason
        for entry in incremental["report"]["cache"]["invalidation_entries"]
        for reason in entry["reasons"]
    }
    assert "transcript_offsets_changed:I002" in reasons


def test_truth_correction_invalidates_conclusion_but_reuses_windows(tmp_path: Path) -> None:
    sample = _copy_sample(tmp_path)
    out = tmp_path / "out-truth"
    baseline = _run(sample, out)
    manifest_path = sample / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    interval = next(item for item in raw["intervals"] if item["id"] == "I001")
    interval["truth_command_code"] = 4
    command = next(item for item in raw["commands"] if item["code"] == 4)
    command["allowed_chars"] = sorted(set(command["allowed_chars"]) | set(interval["text"]))
    manifest_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    incremental = _run(sample, out)
    full = _run(sample, tmp_path / "out-truth-full")
    assert incremental["final_digest"] == full["final_digest"] != baseline["final_digest"]
    assert incremental["report"]["interval_truth_match_count"] == 3
    evidence = {item["interval_id"]: item for item in incremental["report"]["interval_evidence"]}
    assert evidence["I001"]["truth_command_code"] == 4
    assert evidence["I001"]["truth_matches"] is False
    reasons = {
        reason
        for entry in incremental["report"]["cache"]["invalidation_entries"]
        for reason in entry["reasons"]
    }
    assert "receipt_parameters_changed" in reasons
    assert not any(
        entry["stage"] == "features" and "I001" in entry["old_refs"].get("interval_ids", [])
        for entry in incremental["report"]["cache"]["invalidation_entries"]
    )


def test_quantization_profile_changes_intermediate_features_posteriors_and_frame_result(tmp_path: Path) -> None:
    sample = _copy_sample(tmp_path)
    baseline = _run(sample, tmp_path / "out-q6")
    config_path = sample / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["analysis"]["frac_bits"] = 8
    config["global_version"] = 8
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    q8 = _run(sample, tmp_path / "out-q8")
    assert baseline["status"] == q8["status"] == "complete"
    assert baseline["report"]["frames_digest"] != q8["report"]["frames_digest"]
    assert baseline["final_digest"] != q8["final_digest"]
    state_q6 = _state(tmp_path / "out-q6")
    state_q8 = _state(tmp_path / "out-q8")
    feature_q6 = state_q6["ledger"]["entries"]["feature:W000000"]["payload"]["global"]
    feature_q8 = state_q8["ledger"]["entries"]["feature:W000000"]["payload"]["global"]
    assert feature_q6["quant_profile"] == "q6_s1000"
    assert feature_q8["quant_profile"] == "q8_s1000"
    assert feature_q6["feature_mean_q"] != feature_q8["feature_mean_q"]
    posterior_q6 = state_q6["ledger"]["entries"]["posterior:W000000"]["payload"]
    posterior_q8 = state_q8["ledger"]["entries"]["posterior:W000000"]["payload"]
    assert posterior_q6["sources"]["I001"]["posterior_ppm"] != posterior_q8["sources"]["I001"]["posterior_ppm"]


def test_forced_termination_recovers_to_same_full_result(tmp_path: Path) -> None:
    sample = _copy_sample(tmp_path)
    baseline = _run(sample, tmp_path / "out-baseline")
    crash_out = tmp_path / "out-crash"
    script = (
        "from voice_diag.pipeline import DiagnosticPipeline, PipelineOptions\n"
        f"DiagnosticPipeline({str(sample / 'config.json')!r}, {str(sample / 'manifest.json')!r}, "
        f"{str(crash_out)!r}, options=PipelineOptions(max_batches=1, crash_at='after_batch')).run()\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    crashed = subprocess.run([sys.executable, "-c", script], env=env, cwd=ROOT, check=False)
    assert crashed.returncode == 137
    pointer = json.loads((crash_out / "CURRENT.json").read_text(encoding="utf-8"))
    state = _state(crash_out)
    assert state["stages"]["parse"] == "complete"
    assert state["stages"]["endpoint"] == "complete"
    assert state["final"] is None
    recovered = _run(sample, crash_out)
    assert recovered["status"] == "complete"
    assert recovered["final_digest"] == baseline["final_digest"]
    history = list((crash_out / "snapshots").iterdir())
    assert len([item for item in history if item.is_dir() and not item.name.startswith(".")]) >= 2


def test_rolling_arrival_cursor_does_not_finalize_until_all_fragments_arrive(tmp_path: Path) -> None:
    sample = _copy_sample(tmp_path)
    out = tmp_path / "out-arrival"
    partial = _run(sample, out, arrival_limit=2)
    assert partial["status"] == "waiting_arrival"
    assert partial["complete"] is False
    assert partial["fragment_cursor"] == 2
    assert partial["all_fragments_arrived"] is False
    complete = _run(sample, out)
    baseline = _run(sample, tmp_path / "out-arrival-baseline")
    assert complete["status"] == "complete"
    assert complete["fragment_cursor"] == 4
    assert complete["all_fragments_arrived"] is True
    assert complete["final_digest"] == baseline["final_digest"]
    assert complete["report"]["cache"]["reuse_count"] > 0


def test_audio_format_failure_is_skipped_and_reported_without_aborting_session(tmp_path: Path) -> None:
    sample = _copy_sample(tmp_path)
    audio_path = sample / "audio" / "F002.wav"
    data = bytearray(audio_path.read_bytes())
    struct.pack_into("<I", data, 24, 8000)
    audio_path.write_bytes(data)
    result = _run(sample, tmp_path / "out-bad-audio")
    assert result["status"] == "complete"
    assert result["report"]["business_code_counts"]["E_AUDIO_FORMAT"] == 1
    assert result["report"]["receipts"]["generated_a1_count"] == 3
    assert result["report"]["receipts"]["generated_a2_count"] == 3
    assert result["report"]["interval_evidence_count"] == 3


def test_pipeline_quota_abort_records_code_and_commits_no_window_products(tmp_path: Path) -> None:
    sample = _copy_sample(tmp_path)
    config_path = sample / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["scheduler"]["per_interval_quota"] = 1
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    out = tmp_path / "out-quota"
    result = _run(sample, out, max_batches=3)
    assert result["status"] == "quota_exceeded"
    assert result["complete"] is False
    assert result["business_code_counts"]["E_QUOTA_EXCEEDED"] == 1
    state = _state(out)
    stages = {entry["stage"] for entry in state["ledger"]["entries"].values()}
    assert "features" not in stages
    assert state["final"] is None

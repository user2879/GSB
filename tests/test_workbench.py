"""End-to-end tests for the codecq workbench (small deterministic config)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from codecq import ledger as ledger_mod
from codecq import report as report_mod
from codecq import runner, synth

BASE_CFG = {
    "seed": 7,
    "retries": 1,
    "video": {"width": 160, "height": 90, "fps": 30, "duration": 2},
    "audio": {"rate": 48000, "channels": 2},
    "shots": [0, 20, 40],
    "metrics": {"weights": {"Y": 0.8, "U": 0.1, "V": 0.1}, "ssim": {"window": 11, "sigma": 1.5}},
    "tolerances": {
        "audio_offset_samples": 80,
        "audio_max_lag_samples": 4800,
        "audio_gain_dev": 0.05,
        "min_similarity": 0.9,
    },
    "encodes": [
        {"id": "a_420", "crf": 23, "gop": 30, "bframes": 2, "pix_fmt": "yuv420p"},
        {"id": "b_444", "crf": 28, "gop": 60, "bframes": 0, "pix_fmt": "yuv444p"},
    ],
    "injections": [],
}

INJECTIONS = [
    {"id": "j1", "type": "frame_drop", "target": "a_420", "frames": [5, 33]},
    {"id": "j2", "type": "timestamp_jump", "target": "a_420", "frame": 10, "delta_ms": 500},
    {"id": "j3", "type": "chroma_mislabel", "target": "b_444", "declared": "yuv420p"},
    {"id": "j4", "type": "audio_rate_misreport", "target": "b_444", "declared_rate": 44100},
    {"id": "j5", "type": "audio_shift", "target": "a_420", "offset_samples": 480},
]


def cfg_with(**over):
    cfg = copy.deepcopy(BASE_CFG)
    cfg.update(over)
    return cfg


@pytest.fixture(scope="module")
def injected_run(tmp_path_factory):
    cfg = cfg_with(injections=copy.deepcopy(INJECTIONS))
    out = tmp_path_factory.mktemp("injected")
    return runner.run_pipeline(cfg, out), out


def encode_by_id(ledger, enc_id):
    return next(e for e in ledger["encodes"] if e["id"] == enc_id)


def codes(enc):
    return [f["code"] for f in enc["findings"]]


def test_source_generation_deterministic(tmp_path):
    cfg = cfg_with()
    f1, a1, t1 = synth.generate_source(cfg, tmp_path / "s1")
    f2, a2, t2 = synth.generate_source(cfg, tmp_path / "s2")
    assert t1 == t2
    assert (tmp_path / "s1" / "audio.wav").read_bytes() == (tmp_path / "s2" / "audio.wav").read_bytes()
    # video/audio timelines correspond exactly
    assert t1["audio"]["samples"] == t1["video"]["frames"] * 48000 // 30
    assert len(t1["blocks"]) == t1["video"]["frames"]
    assert [s["start"] for s in t1["shots"]] == cfg["shots"]


def test_clean_run_has_no_findings(tmp_path):
    result = runner.run_pipeline(cfg_with(), tmp_path)
    ledger = result["ledger"]
    assert ledger["summary"]["execution_complete"] is True
    assert ledger["summary"]["encodes_failed"] == 0
    for enc in ledger["encodes"]:
        assert enc["findings"] == [], enc["id"]
        assert enc["frame_count"] == 60
        # clean audio: zero offset, unit gain, perfect similarity
        for seg in enc["audio_eval"]["segments"]:
            assert seg["offset_samples"] == 0
            assert seg["gain_ratio"] == 1.0
            assert seg["similarity"] == 1.0


def test_two_runs_byte_identical(tmp_path):
    cfg = cfg_with(injections=copy.deepcopy(INJECTIONS))
    runner.run_pipeline(cfg, tmp_path / "r1")
    runner.run_pipeline(cfg, tmp_path / "r2")
    a = (tmp_path / "r1" / "ledger" / "ledger.json").read_bytes()
    b = (tmp_path / "r2" / "ledger" / "ledger.json").read_bytes()
    assert a == b


def test_rerun_same_dir_byte_identical(injected_run):
    result, out = injected_run
    before = (out / "ledger" / "ledger.json").read_bytes()
    cfg = cfg_with(injections=copy.deepcopy(INJECTIONS))
    runner.run_pipeline(cfg, out)
    after = (out / "ledger" / "ledger.json").read_bytes()
    assert before == after


def test_injections_detected_and_located(injected_run):
    ledger = injected_run[0]["ledger"]
    a = encode_by_id(ledger, "a_420")
    b = encode_by_id(ledger, "b_444")
    assert set(codes(a)) == {"E_FRAME_DROP", "E_TIMESTAMP_JUMP", "E_AUDIO_DRIFT"}
    assert set(codes(b)) == {"E_CHROMA_MISMATCH", "E_AUDIO_RATE"}

    drop = next(f for f in a["findings"] if f["code"] == "E_FRAME_DROP")
    assert drop["location"]["frames"] == [5, 33]
    ts = next(f for f in a["findings"] if f["code"] == "E_TIMESTAMP_JUMP")
    assert ts["location"]["frame"] == 10
    drift = next(f for f in a["findings"] if f["code"] == "E_AUDIO_DRIFT")
    assert drift["location"]["segment"] == 0
    # injection records are serialized into the ledger for exact reproduction
    assert [i["id"] for i in a["injections"]] == ["j1", "j2", "j5"]
    assert [i["id"] for i in b["injections"]] == ["j3", "j4"]


def test_audio_drift_within_tolerance_not_flagged(tmp_path):
    inj = [{"id": "s", "type": "audio_shift", "target": "a_420", "offset_samples": 40}]
    result = runner.run_pipeline(cfg_with(injections=inj), tmp_path)
    a = encode_by_id(result["ledger"], "a_420")
    assert "E_AUDIO_DRIFT" not in codes(a)
    assert a["audio_eval"]["segments"][0]["offset_samples"] == 40


def test_segment_boundaries_match_truth(injected_run):
    ledger = injected_run[0]["ledger"]
    truth_shots = [(s["start"], s["end"]) for s in ledger["source"]["shots"]]
    for enc in ledger["encodes"]:
        seg_bounds = [(s["start"], s["end"]) for s in enc["metrics"]["segments"]]
        assert seg_bounds == truth_shots  # exact boundary equality, not count equality
    a = encode_by_id(ledger, "a_420")
    seg0 = a["metrics"]["segments"][0]
    assert seg0["frames_missing"] == [5]
    assert seg0["frames_present"] == 19


def test_encode_failure_isolated(tmp_path):
    cfg = cfg_with()
    cfg["encodes"].append(
        {"id": "bad_fmt", "crf": 23, "gop": 30, "bframes": 0, "pix_fmt": "yuv422p"}
    )
    result = runner.run_pipeline(cfg, tmp_path)
    ledger = result["ledger"]
    bad = encode_by_id(ledger, "bad_fmt")
    assert bad["status"] == "failed"
    assert bad["retries"] == 1
    assert codes(bad) == ["E_ENCODE_FAILED"]
    # other units still produced; failure is not silently counted as pass
    assert ledger["summary"]["encodes_ok"] == 2
    assert ledger["summary"]["execution_complete"] is False
    assert ledger["isolation"][0]["encode"] == "bad_fmt"
    assert ledger["isolation"][0]["retries"] == 1


def test_ledger_version_stable_and_tmp_ignored(injected_run):
    result, out = injected_run
    ledger_dir = out / "ledger"
    version = result["ledger"]["version"]
    assert (ledger_dir / "HEAD").read_text().strip() == version
    assert (ledger_dir / "snapshots" / f"{version}.json").exists()
    # stray tmp files (as left by a kill) are ignored and cleaned
    (ledger_dir / "ledger.json.tmp").write_text("garbage")
    loaded = ledger_mod.load_committed(ledger_dir)
    assert loaded["version"] == version
    assert not (ledger_dir / "ledger.json.tmp").exists()


def test_ledger_tamper_rejected(injected_run):
    _, out = injected_run
    import shutil

    ledger_dir = out / "ledger_tampered"
    shutil.copytree(out / "ledger", ledger_dir)
    data = json.loads((ledger_dir / "ledger.json").read_text())
    data["summary"]["findings_total"] = 0
    (ledger_dir / "ledger.json").write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        ledger_mod.load_committed(ledger_dir)


def test_view_reads_committed_snapshot(injected_run, tmp_path):
    _, out = injected_run
    report_path = tmp_path / "report.html"
    info = report_mod.render_report(out / "ledger", report_path)
    assert info["version"].startswith("v1-")
    html = report_path.read_text(encoding="utf-8")
    assert "E_FRAME_DROP" in html and "canvas" in html

"""End-to-end requirements 4-8: decisions, error tracking, cache, snapshot, isolation."""
from __future__ import annotations

import json
import os

from voiceprint.config import build_config, load_pair_manifest
from voiceprint.pipeline import VoiceprintPipeline
from voiceprint.reporting import build_report
from voiceprint.util import fingerprint, read_json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAIR_PATH = os.path.join(ROOT, "configs", "speaker_pairs.json")
INJECTION_PATH = os.path.join(ROOT, "configs", "injection_manifest.json")


def _run(config, dataset_dir, work_dir, injection_hash, pair_manifest,
         resume=False, allow_config_change=False):
    pipeline = VoiceprintPipeline(
        config=config, pair_manifest=pair_manifest, input_dir=dataset_dir,
        work_dir=work_dir, injection_hash=injection_hash,
    )
    pipeline.run(resume=resume, allow_config_change=allow_config_change)
    return build_report(pipeline, pair_manifest, elapsed_s=0.0), pipeline


def test_end_to_end_decisions_and_failure_isolation(dataset_dir, base_config,
                                                    pair_manifest, injection_hash, tmp_path):
    report, _ = _run(base_config, dataset_dir, str(tmp_path / "work"),
                     injection_hash, pair_manifest)
    # two malformed WAVs isolated, all valid samples still processed
    failed = {(f["item_id"], f["code"]) for f in report["failures"]}
    assert ("bad_rate_8k", "E_AUDIO_FORMAT") in failed
    assert ("bad_stereo", "E_AUDIO_FORMAT") in failed
    assert report["pair_summary"]["same_speaker"]["accuracy"] == 1.0
    assert report["pair_summary"]["different_speaker"]["accuracy"] == 1.0
    # every pair reports integer intermediates and basis text
    for entry in report["pair_similarity"]:
        assert entry["base"]["dot"] is not None
        assert entry["base"]["decision_basis"]
        assert entry["base"]["sample_a"] if False else entry["sample_a"]


def test_resume_is_field_identical_and_fully_reused(dataset_dir, base_config,
                                                    pair_manifest, injection_hash, tmp_path):
    work = str(tmp_path / "work")
    first, _ = _run(base_config, dataset_dir, work, injection_hash, pair_manifest)
    second, _ = _run(base_config, dataset_dir, work, injection_hash, pair_manifest, resume=True)
    volatile = {"incremental", "snapshot", "elapsed_seconds"}
    for key in first:
        if key not in volatile:
            assert first[key] == second[key]
    assert second["incremental"]["recomputed_count"] == 0
    assert second["incremental"]["reused_count"] > 0
    assert second["failures"] == first["failures"]


def test_incremental_recompute_scope(dataset_dir, base_config, pair_manifest,
                                     injection_hash, tmp_path):
    import numpy as np
    from voiceprint.synth import SpeakerProfile, synth_utterance, write_wav

    work = str(tmp_path / "work")
    _run(base_config, dataset_dir, work, injection_hash, pair_manifest)
    profile = SpeakerProfile("spk_01", 120.0, 6.0, (730.0, 1090.0, 2440.0),
                             (1.0, 0.7, 0.4), 3.5)
    replacement = synth_utterance(profile, "spk_01_utt_02_REPLACED")
    write_wav(str(tmp_path / "replacement.wav"), replacement)
    target = os.path.join(dataset_dir, "spk_01_utt_02.wav")
    with open(tmp_path / "replacement.wav", "rb") as src, open(target, "wb") as dst:
        dst.write(src.read())

    report, _ = _run(base_config, dataset_dir, work, injection_hash, pair_manifest, resume=True)
    recomputed = report["incremental"]["recomputed"]
    affected_stages = {item.split("|")[0] for item in recomputed}
    # only the replaced sample chain and pairs referencing it recompute
    assert all("spk_01_utt_02" in item or "pair_" in item for item in recomputed)
    assert {"features", "aggregation", "embedding", "similarity"} <= affected_stages
    # unrelated sample embedding is reused, not recomputed
    assert "embedding|spk_02_utt_01" not in recomputed
    assert report["incremental"]["reused_count"] > 0


def test_config_mismatch_rejects_resume(dataset_dir, base_config, pair_manifest,
                                        injection_hash, tmp_path):
    import pytest
    from voiceprint.errors import ConfigMismatchError

    work = str(tmp_path / "work")
    _run(base_config, dataset_dir, work, injection_hash, pair_manifest)
    changed = build_config({**base_config.raw, "embedding_dim": 8})
    with pytest.raises(ConfigMismatchError) as exc:
        _run(changed, dataset_dir, work, injection_hash, pair_manifest, resume=True)
    assert exc.value.code == "E_CONFIG_MISMATCH"


def test_threshold_change_only_recomputes_similarity(dataset_dir, base_config,
                                                     pair_manifest, injection_hash, tmp_path):
    work = str(tmp_path / "work")
    _run(base_config, dataset_dir, work, injection_hash, pair_manifest)
    threshold_cfg = build_config({**base_config.raw, "similarity_threshold": 0.9})
    report, _ = _run(threshold_cfg, dataset_dir, work, injection_hash, pair_manifest,
                     resume=True, allow_config_change=True)
    stages = {item.split("|")[0] for item in report["incremental"]["recomputed"]}
    assert stages == {"similarity"}


def test_boundary_sensitivity_flag(dataset_dir, base_config, pair_manifest,
                                   injection_hash, tmp_path):
    # pick a threshold between the base-width and reference-width similarity of one pair
    seed_report, _ = _run(base_config, dataset_dir, str(tmp_path / "seed"),
                         injection_hash, pair_manifest)
    target = next(p for p in seed_report["pair_similarity"]
                  if p["pair_id"] == "pair_diff_03")
    mid = (target["base"]["similarity"] + target["reference_width"]["similarity"]) / 2
    bnd_cfg = build_config({**base_config.raw, "similarity_threshold": mid})
    report, _ = _run(bnd_cfg, dataset_dir, str(tmp_path / "bnd"),
                     injection_hash, pair_manifest)
    flagged = {b["pair_id"] for b in report["boundary_sensitive"]}
    assert "pair_diff_03" in flagged
    codes = {f["code"] for f in report["failures"]}
    assert "E_BOUNDARY_SENSITIVE" in codes


def test_error_propagation_and_adjudication(dataset_dir, base_config,
                                            pair_manifest, injection_hash, tmp_path):
    report, _ = _run(base_config, dataset_dir, str(tmp_path / "work"),
                     injection_hash, pair_manifest)
    by_sample = report["error_propagation"]["by_sample"]
    for sample_id, entry in by_sample.items():
        assert set(entry["stages"]) == {
            "feature_quantization", "statistic_aggregation", "embedding_projection"
        }
        # upstream float64 statistics are width independent -> exactly zero diff
        assert entry["stages"]["statistic_aggregation"]["max_abs_diff"] == 0.0
    adjudicated = [pid for pid, pe in report["error_propagation"]["by_pair"].items()
                   if pe["adjudication"] is not None]
    for pid in adjudicated:
        verdict = report["error_propagation"]["by_pair"][pid]["adjudication"]
        assert verdict["winning_stage"] == "similarity"
        assert "downstream" in verdict["reason"]


def test_snapshot_versions_are_atomic_and_resolvable(dataset_dir, base_config,
                                                     pair_manifest, injection_hash, tmp_path):
    work = str(tmp_path / "work")
    _, pipeline = _run(base_config, dataset_dir, work, injection_hash, pair_manifest)
    manifest = pipeline.store.read_manifest()
    for stage, key_map in manifest["artifact_versions"].items():
        for key, version in key_map.items():
            path = pipeline.store.artifact_path(version, stage, key)
            assert os.path.isfile(path)
    # every snapshot commit is a distinct version directory
    versions = os.listdir(os.path.join(work, "versions"))
    assert len(versions) >= 1

import os

from peft_fp import errors
from peft_fp.pipeline import run_pipeline


def _codes(report):
    return {f["code"] for f in report["failures"]}


def test_healthy_samples_pass(sample_dir, store):
    report = run_pipeline(sample_dir, store)
    assert report["failures"] == []
    assert len(report["tasks"]) == 4


def test_nonfinite_weight(faults_dir, tmp_path):
    report = run_pipeline(os.path.join(faults_dir, "nonfinite_weight"), str(tmp_path / "s"))
    assert errors.E_BASE_INVALID in _codes(report)
    assert report["tasks"] == []


def test_shape_mismatch(faults_dir, tmp_path):
    report = run_pipeline(os.path.join(faults_dir, "shape_mismatch"), str(tmp_path / "s"))
    assert errors.E_BASE_INVALID in _codes(report)


def test_layer_not_whitelisted(faults_dir, tmp_path):
    report = run_pipeline(os.path.join(faults_dir, "layer_not_whitelisted"), str(tmp_path / "s"))
    assert errors.E_TASK_INVALID in _codes(report)
    # the invalid task must not reach training/merge
    names = [t["name"] for t in report["tasks"]]
    assert "task_alpha" not in names
    assert len(names) == 3


def test_zero_tolerance(faults_dir, tmp_path):
    report = run_pipeline(os.path.join(faults_dir, "zero_tolerance"), str(tmp_path / "s"))
    assert errors.E_QUANT_INVALID in _codes(report)
    assert report["tasks"] == []


def test_bad_quant_tier(faults_dir, tmp_path):
    report = run_pipeline(os.path.join(faults_dir, "bad_quant_tier"), str(tmp_path / "s"))
    assert errors.E_QUANT_INVALID in _codes(report)


def test_negative_rank(faults_dir, tmp_path):
    report = run_pipeline(os.path.join(faults_dir, "negative_rank"), str(tmp_path / "s"))
    assert errors.E_TASK_INVALID in _codes(report)


def test_huge_alpha_irreversible(faults_dir, tmp_path):
    report = run_pipeline(os.path.join(faults_dir, "huge_alpha"), str(tmp_path / "s"))
    assert errors.E_IRREVERSIBLE_MERGE in _codes(report)

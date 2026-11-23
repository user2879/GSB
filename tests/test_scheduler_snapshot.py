import json
import shutil

from voicecalib import load_inputs, run_pipeline, build_report
from voicecalib.errors import ErrorCode
from voicecalib.hashing import fingerprint


def test_budget_pending_retained(samples_dir, tmp_path):
    inputs = load_inputs(samples_dir)
    inputs.config.batch_segment_budget = 2
    result = run_pipeline(inputs, tmp_path / "out", resume=False)
    batches = result.batches
    assert len(batches) == 3
    assert batches[0]["consumed"] == 2
    assert batches[0]["pending_rounds"] == [2, 3, 4, 5]
    # 不允许跳过失效轮次：最终全部完成。
    assert len(result.calibrations) == 6
    assert ErrorCode.E_CALIB_BUDGET_PENDING in result.errors.codes()
    # 每批消耗不得超过预算。
    assert all(b["consumed"] <= b["budget"] for b in batches)


def test_snapshot_restore_identical(full_samples_dir, tmp_path):
    out = tmp_path / "out"
    inputs = load_inputs(full_samples_dir)
    first = run_pipeline(inputs, out, resume=False)
    first_report = build_report(first, inputs.config, full_samples_dir, "first")

    inputs2 = load_inputs(full_samples_dir)
    second = run_pipeline(inputs2, out, resume=True)
    second_report = build_report(second, inputs2.config, full_samples_dir, "second")
    assert second.snapshot_restored is True
    assert second_report["incremental"]["reuse_hits"] == 12
    assert fingerprint(first_report["emotion_trace"]) == fingerprint(
        second_report["emotion_trace"])
    assert fingerprint(first_report["speaker_trace"]) == fingerprint(
        second_report["speaker_trace"])
    # 快照清单字段齐全。
    manifest = second.snapshot_manifest
    for key in ("artifact_version", "stage", "config_hash", "injection_hash",
                "cursor_round", "frac_bits", "quant_gear"):
        assert key in manifest


def test_snapshot_stale_on_config_change(full_samples_dir, tmp_path):
    out = tmp_path / "out"
    inputs = load_inputs(full_samples_dir)
    run_pipeline(inputs, out, resume=False)

    stale_out = tmp_path / "stale"
    stale_out.mkdir()
    shutil.copytree(out / "snapshots", stale_out / "snapshots")
    shutil.copy(out / "ledger.json", stale_out / "ledger.json")

    inputs2 = load_inputs(full_samples_dir)
    inputs2.config.source["calibration_gain"] = 0.65
    inputs2.config.config_hash = fingerprint(inputs2.config.source)
    result = run_pipeline(inputs2, stale_out, resume=True)
    assert result.errors.has(ErrorCode.E_SNAPSHOT_STALE)
    # 陈旧快照被拒绝后冷启动仍可完成全量计算。
    assert len(result.calibrations) == 12


def test_snapshot_atomic_layout(output_dir, samples_dir):
    inputs = load_inputs(samples_dir)
    run_pipeline(inputs, output_dir, resume=False)
    snap = output_dir / "snapshots" / "trajectory"
    assert (snap / "manifest.json").exists()
    assert (snap / "artifacts.json").exists()
    assert (snap / "ledger.json").exists()
    # 临时暂存目录必须已被整体替换清理。
    assert not list(output_dir.glob("snapshots/.stage-*"))

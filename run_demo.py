"""验收演示：覆盖证据链校准、增量失效、预算待办、并发一致、快照恢复与故障注入。

每个场景使用独立输出目录，全部产物（报告 JSON/Markdown、快照、账本）落盘。
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from voicecalib import (build_report, load_inputs, render_markdown,  # noqa: E402
                        run_pipeline, write_report)
from voicecalib.audio import write_wav  # noqa: E402
from voicecalib.errors import ErrorCode  # noqa: E402
from voicecalib.hashing import fingerprint  # noqa: E402

import numpy as np  # noqa: E402


def fresh_dir(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def scenario_baseline(base: Path) -> dict:
    out = fresh_dir(base / "01_baseline")
    inputs = load_inputs(ROOT / "samples")
    result = run_pipeline(inputs, out, resume=False)
    report = build_report(result, inputs.config, ROOT / "samples", "01_baseline")
    write_report(report, out)
    (out / "diagnostic_report.md").write_text(
        render_markdown(report), encoding="utf-8")
    return report


def scenario_determinism(base: Path) -> dict:
    """单线程 vs 多线程逐字段一致 + 重复运行逐位一致。"""
    summary: dict = {"scenario": "02_determinism"}
    reports = []
    for label, workers in (("single", 1), ("multi", 4), ("multi_again", 4)):
        sample_dir = ROOT / "samples"
        out = fresh_dir(base / f"02_{label}")
        inputs = load_inputs(sample_dir)
        inputs.config.max_workers = workers
        result = run_pipeline(inputs, out, resume=False)
        report = build_report(result, inputs.config, sample_dir, label)
        write_report(report, out)
        reports.append((label, report))
    fields = ("emotion_trace", "speaker_trace", "trajectories", "failures")
    base_report = reports[0][1]
    diffs = {}
    for label, report in reports[1:]:
        diffs[label] = {
            field: fingerprint(base_report[field]) != fingerprint(report[field])
            for field in fields
        }
    summary["field_diffs_vs_single"] = diffs
    summary["identical"] = all(
        not any(flag for flag in flags.values()) for flags in diffs.values())
    return summary


def scenario_incremental(base: Path) -> dict:
    """替换第 6 轮音频：仅 R6 及其下游重算，R0-R5 全部命中复用。"""
    sample_copy = ROOT / "output" / "samples_incr"
    if sample_copy.exists():
        shutil.rmtree(sample_copy)
    shutil.copytree(ROOT / "samples", sample_copy)
    out = fresh_dir(base / "03_incremental_first")
    inputs = load_inputs(sample_copy)
    first = run_pipeline(inputs, out, resume=False)
    first_report = build_report(first, inputs.config, sample_copy,
                                "03_first")
    write_report(first_report, out)

    # 替换 R6 音频内容（固定确定性新波形）。
    rng = np.random.default_rng(777)
    replacement = np.sin(
        2 * np.pi * 180.0 * np.arange(16000) / 16000) * 0.5 + rng.normal(
            0, 0.005, 16000)
    write_wav(sample_copy / "round_006.wav", replacement)

    out2 = base / "03_incremental_second"
    fresh_dir(out2)
    # 复用首轮账本与快照：把首轮输出迁移到第二轮输出目录。
    shutil.copytree(out / "snapshots", out2 / "snapshots")
    shutil.copy(out / "ledger.json", out2 / "ledger.json")
    inputs2 = load_inputs(sample_copy)
    second = run_pipeline(inputs2, out2, resume=True)
    second_report = build_report(second, inputs2.config, sample_copy,
                                 "03_incremental")
    write_report(second_report, out2)
    return {
        "scenario": "03_incremental_audio_replace",
        "reuse_hits": second_report["incremental"]["reuse_hits"],
        "recompute_rounds": second_report["incremental"]["recompute_rounds"],
        "upstream_preserved": second_report["incremental"]["reuse_hits"] >= 6,
        "decisions": second_report["incremental"]["decisions"],
        "report": second_report,
    }


def scenario_budget(base: Path) -> dict:
    """预算=2：每批最多处理 2 个失效片段，待办轮次保留到下批优先处理。"""
    out = fresh_dir(base / "04_budget")
    inputs = load_inputs(ROOT / "samples")
    inputs.config.batch_segment_budget = 2
    result = run_pipeline(inputs, out, resume=False)
    report = build_report(result, inputs.config, ROOT / "samples", "04_budget")
    write_report(report, out)
    return {
        "scenario": "04_budget_pending",
        "batches": report["budget"]["batches"],
        "pending_seen": ErrorCode.E_CALIB_BUDGET_PENDING in report["failure_summary"],
        "completed": report["rounds_completed"],
        "report": report,
    }


def scenario_delete(base: Path) -> dict:
    """删除 R4 音频：R4 及下游全部失效，R0-R3 保持命中且不重算。"""
    sample_copy = ROOT / "output" / "samples_delete"
    if sample_copy.exists():
        shutil.rmtree(sample_copy)
    shutil.copytree(ROOT / "samples", sample_copy)
    out = fresh_dir(base / "05_delete_first")
    inputs = load_inputs(sample_copy)
    first = run_pipeline(inputs, out, resume=False)
    write_report(build_report(first, inputs.config, sample_copy, "05_first"), out)

    (sample_copy / "round_004.wav").unlink()
    out2 = fresh_dir(base / "05_delete_second")
    shutil.copytree(out / "snapshots", out2 / "snapshots")
    shutil.copy(out / "ledger.json", out2 / "ledger.json")
    inputs2 = load_inputs(sample_copy)
    missing_codes = inputs2.errors.codes()
    second = run_pipeline(inputs2, out2, resume=True)
    report = build_report(second, inputs2.config, sample_copy, "05_delete")
    write_report(report, out2)
    return {
        "scenario": "05_audio_delete",
        "load_error_codes": missing_codes,
        "reuse_hits": report["incremental"]["reuse_hits"],
        "recompute_rounds": report["incremental"]["recompute_rounds"],
        "valid_rounds": report["rounds_total"],
        "report": report,
    }


def scenario_snapshot(base: Path) -> dict:
    """快照恢复一致性 + 陈旧快照 E_SNAPSHOT_STALE 判定。"""
    out = fresh_dir(base / "06_snapshot")
    inputs = load_inputs(ROOT / "samples")
    first = run_pipeline(inputs, out, resume=False)
    first_report = build_report(first, inputs.config, ROOT / "samples",
                                "06_first")

    inputs2 = load_inputs(ROOT / "samples")
    second = run_pipeline(inputs2, out, resume=True)
    second_report = build_report(second, inputs2.config, ROOT / "samples",
                                 "06_resume")
    write_report(second_report, out)
    resumed_identical = (
        fingerprint(first_report["emotion_trace"])
        == fingerprint(second_report["emotion_trace"])
        and fingerprint(first_report["speaker_trace"])
        == fingerprint(second_report["speaker_trace"])
        and fingerprint(first_report["trajectories"])
        == fingerprint(second_report["trajectories"])
    )

    # 篡改配置哈希触发陈旧快照。
    stale_out = fresh_dir(base / "06_snapshot_stale")
    shutil.copytree(out / "snapshots", stale_out / "snapshots")
    shutil.copy(out / "ledger.json", stale_out / "ledger.json")
    inputs3 = load_inputs(ROOT / "samples")
    inputs3.config.source["calibration_gain"] = 0.65
    from voicecalib.hashing import fingerprint as fp
    inputs3.config.config_hash = fp(inputs3.config.source)
    stale_result = run_pipeline(inputs3, stale_out, resume=True)
    return {
        "scenario": "06_snapshot_restore",
        "restored": second_report["snapshot_restored"],
        "resume_identical": resumed_identical,
        "reuse_hits_resume": second_report["incremental"]["reuse_hits"],
        "stale_detected": stale_result.errors.has(ErrorCode.E_SNAPSHOT_STALE),
        "stale_codes": stale_result.errors.codes(),
    }


def scenario_injection(base: Path) -> dict:
    """独立注入清单：四类真实故障命中且单轮隔离。"""
    out = fresh_dir(base / "07_injection")
    inputs = load_inputs(ROOT / "samples", injection_file="injections.json")
    result = run_pipeline(inputs, out, resume=False)
    report = build_report(result, inputs.config, ROOT / "samples",
                          "07_injection")
    write_report(report, out)
    (out / "diagnostic_report.md").write_text(
        render_markdown(report), encoding="utf-8")
    return {
        "scenario": "07_fault_injection",
        "injections_hit": report["injections_hit"],
        "failure_summary": report["failure_summary"],
        "rounds_completed": report["rounds_completed"],
        "report": report,
    }


def scenario_validation(base: Path) -> dict:
    """畸形 WAV / 非有限证据 / 越界注入的加载校验业务码。"""
    sample_copy = ROOT / "output" / "samples_bad"
    if sample_copy.exists():
        shutil.rmtree(sample_copy)
    shutil.copytree(ROOT / "samples", sample_copy)
    # 构造非 RIFF 文件。
    (sample_copy / "round_002.wav").write_bytes(b"NOT_A_RIFF_FILE" * 20)
    # 证据 JSONL 注入 NaN 与维度缺失。
    lines = (sample_copy / "acoustic_evidence.jsonl").read_text(
        encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    for entry in entries:
        if entry["round"] == 1:
            entry["features"]["f0_mean_hz"] = float("nan")
        if entry["round"] == 10:
            entry["features"].pop("rhythm_rate")
    with (sample_copy / "acoustic_evidence.jsonl").open(
            "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    out = fresh_dir(base / "08_validation")
    inputs = load_inputs(sample_copy, injection_file="injections.json")
    return {
        "scenario": "08_validation_codes",
        "error_codes": sorted(set(inputs.errors.codes())),
        "valid_rounds": inputs.valid_rounds,
        "failures": inputs.errors.sorted_dict_list(),
    }


def main() -> None:
    base = fresh_dir(ROOT / "output" / "demo")
    summaries = {}
    summaries["baseline"] = scenario_baseline(base)
    summaries["determinism"] = scenario_determinism(base)
    summaries["incremental"] = scenario_incremental(base)
    summaries["budget"] = scenario_budget(base)
    summaries["delete"] = scenario_delete(base)
    summaries["snapshot"] = scenario_snapshot(base)
    summaries["injection"] = scenario_injection(base)
    summaries["validation"] = scenario_validation(base)

    compact = {}
    for key, value in summaries.items():
        compact[key] = {k: v for k, v in value.items() if k != "report"}
    out_path = base / "acceptance_summary.json"
    out_path.write_text(json.dumps(compact, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False, default=str))
    print(f"\nsummary written to {out_path}")


if __name__ == "__main__":
    main()

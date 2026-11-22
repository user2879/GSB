"""命令行入口: gen / run (含编辑序列增量执行与全量重建一致性验证)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .config import load_config
from .edits import apply_batch, config_to_raw, load_edit_plan
from .errors import CqError, FailureList
from .pipeline import Pipeline, RunResult
from .report import build_report, write_report
from .samples import (
    generate_samples,
    load_samples,
    samples_to_records,
    write_samples_jsonl,
)
from .store import SnapshotStore


def _clipped_totals(result: RunResult) -> dict[str, int]:
    totals: dict[str, int] = {}
    for col_key, entry in result.columns.items():
        tensor = col_key.split(":")[0]
        totals[tensor] = totals.get(tensor, 0) + entry["quant"]["clipped_count"]
    return totals


def _normalize(obj):
    return json.loads(json.dumps(obj, sort_keys=True))


def compare_results(a: RunResult, b: RunResult) -> list[str]:
    """逐字段对比两次运行的列结果/分层/失败清单, 返回差异描述列表."""
    diffs = []
    if _normalize(a.columns) != _normalize(b.columns):
        diffs.append("columns")
    if _normalize(a.stratify) != _normalize(b.stratify):
        diffs.append("stratify")
    if _normalize(a.failures.to_list()) != _normalize(b.failures.to_list()):
        diffs.append("failures")
    return diffs


def cmd_run(args) -> int:
    failures = FailureList()
    try:
        config = load_config(args.config)
    except CqError as exc:
        failures.add_error(exc)
        report = {"failures": failures.to_list(), "aborted": True}
        write_report(report, args.report)
        print(json.dumps(report, ensure_ascii=False))
        return 2
    try:
        samples = load_samples(args.samples, config)
    except CqError as exc:
        failures.add_error(exc)
        report = {"failures": failures.to_list(), "aborted": True}
        write_report(report, args.report)
        print(json.dumps(report, ensure_ascii=False))
        return 2

    store = SnapshotStore(args.workdir)
    pipeline = Pipeline(config, samples, store)
    result = pipeline.run()
    failures.extend(pipeline.failures)

    bitwidth_comparisons: list[dict] = []
    prev_bits = (config.quant_bits, config.frac_bits)
    prev_clipped = _clipped_totals(result)
    batch_reports = []

    if args.edits:
        batches = load_edit_plan(args.edits)
        for b_idx, batch in enumerate(batches):
            config, events = apply_batch(config, samples, batch, failures)
            pipeline = Pipeline(config, samples, store)
            result = pipeline.run()
            failures.extend(pipeline.failures)
            new_bits = (config.quant_bits, config.frac_bits)
            new_clipped = _clipped_totals(result)
            if new_bits != prev_bits:
                for tensor in sorted(set(prev_clipped) | set(new_clipped)):
                    bitwidth_comparisons.append({
                        "batch": b_idx,
                        "tensor": tensor,
                        "old": {"quant_bits": prev_bits[0], "frac_bits": prev_bits[1],
                                "clipped_count": prev_clipped.get(tensor, 0)},
                        "new": {"quant_bits": new_bits[0], "frac_bits": new_bits[1],
                                "clipped_count": new_clipped.get(tensor, 0)},
                    })
            prev_bits = new_bits
            prev_clipped = new_clipped
            batch_reports.append({"batch": b_idx, "events": events,
                                  "global_version": result.global_version})

    extra = {
        "bitwidth_comparisons": bitwidth_comparisons,
        "batches": batch_reports,
    }

    if args.verify_rebuild:
        # 物化最终输入, 从空白工作区全量重建并逐字段对比
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            final_config = tmp_path / "config.json"
            final_samples = tmp_path / "samples.jsonl"
            final_config.write_text(json.dumps(config_to_raw(config), ensure_ascii=False))
            write_samples_jsonl(samples_to_records(samples, config), str(final_samples))
            fresh_config = load_config(str(final_config))
            fresh_samples = load_samples(str(final_samples), fresh_config)
            fresh_store = SnapshotStore(tmp_path / "workdir")
            fresh_result = Pipeline(fresh_config, fresh_samples, fresh_store).run()
            diffs = compare_results(result, fresh_result)
            extra["rebuild_verification"] = {
                "consistent": not diffs,
                "diff_fields": diffs,
            }

    report = build_report(result, extra)
    report["failures"] = failures.to_list()
    write_report(report, args.report)
    print(json.dumps({"report": args.report,
                      "global_version": result.global_version,
                      "failures": len(report["failures"]),
                      "rebuild": extra.get("rebuild_verification")}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cqdiag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--samples", required=True)
    run_p.add_argument("--workdir", required=True)
    run_p.add_argument("--report", required=True)
    run_p.add_argument("--edits")
    run_p.add_argument("--verify-rebuild", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    return 1

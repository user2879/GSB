"""命令行入口：generate / run / verify。"""
import argparse
import json
import os
import sys

from .config import load_config
from .errors import EGWError
from .layout_probe import run_probe
from .requests import load_requests
from .samplegen import generate_samples
from .scenario import (
    apply_agg_overrides,
    apply_state_edits,
    build_gear_comparisons,
    build_report,
    load_edits,
    run_full,
    run_incremental,
)
from .storage import load_rows


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, ensure_ascii=False, indent=2)


def _load_inputs(config_path, rows_path, requests_path):
    config = load_config(config_path)
    table, row_failures = load_rows(rows_path, config["dim"])
    requests, request_failures = load_requests(requests_path, config["agg_modes"])
    return config, table, requests, row_failures + request_failures


def cmd_generate(args):
    info = generate_samples(args.out_dir)
    _write_json(os.path.join(args.out_dir, "sample_manifest.json"), info)
    print(json.dumps(info, indent=2, ensure_ascii=False))


def _execute(config_path, rows_path, requests_path, edits_path, snapshot_dir,
             report_path, concurrency_override, force_rebuild):
    config, table, requests, load_failures = _load_inputs(
        config_path, rows_path, requests_path
    )
    edits = load_edits(edits_path)
    final_table, final_cfg, agg_overrides, step_meta = apply_state_edits(table, edits, config)
    requests = apply_agg_overrides(requests, agg_overrides)

    engine, outcome = run_full(
        final_cfg,
        final_table,
        requests,
        snapshot_dir=snapshot_dir,
        concurrency_override=concurrency_override,
        force_rebuild=force_rebuild,
    )
    probe = run_probe(final_cfg)
    comparisons = build_gear_comparisons(engine, final_table, requests)
    report = build_report(final_cfg, outcome, load_failures, probe, comparisons)
    report["edit_steps"] = step_meta
    _write_json(report_path, report)
    return report


def cmd_run(args):
    report = _execute(
        args.config,
        args.rows,
        args.requests,
        args.edits,
        args.snapshot_dir,
        args.report,
        args.concurrency,
        force_rebuild=not args.resume,
    )
    print(
        json.dumps(
            {
                "report": args.report,
                "requests": len(report["requests"]),
                "failures": len(report["failures"]),
                "reused": report["reuse_ledger"]["reused"],
                "recomputed": report["reuse_ledger"]["recomputed"],
                "layout_order_effect_observed": report["layout_probe"]["order_effect_observed"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def cmd_verify(args):
    from .scenario import run_incremental

    config, table, requests, load_failures = _load_inputs(
        args.config, args.rows, args.requests
    )
    edits = load_edits(args.edits)
    final_table, final_cfg, agg_overrides, step_meta = apply_state_edits(
        table, edits, config
    )
    final_requests = apply_agg_overrides(
        [__import__("copy").copy(req) for req in requests], agg_overrides
    )
    probe = run_probe(final_cfg)

    summaries = {}
    reports = {}

    for label, concurrency in (("single", 1), ("concurrent", final_cfg["concurrency"])):
        inc_engine, inc_outcome, ledgers, _, _, _ = run_incremental(
            config,
            table,
            [__import__("copy").copy(req) for req in requests],
            edits,
            snapshot_dir=os.path.join(args.work_dir, f"snapshot_inc_{label}"),
            concurrency_override=concurrency,
        )
        inc_report = build_report(
            final_cfg,
            inc_outcome,
            load_failures,
            probe,
            build_gear_comparisons(inc_engine, final_table, final_requests),
        )
        inc_report["edit_steps"] = step_meta
        inc_report["incremental_ledgers"] = ledgers

        full_engine, full_outcome = run_full(
            final_cfg,
            final_table,
            [__import__("copy").copy(req) for req in final_requests],
            snapshot_dir=os.path.join(args.work_dir, f"snapshot_full_{label}"),
            concurrency_override=concurrency,
            force_rebuild=True,
            expected_version=inc_outcome["global_version"],
        )
        full_report = build_report(
            final_cfg,
            full_outcome,
            load_failures,
            probe,
            build_gear_comparisons(full_engine, final_table, final_requests),
        )
        full_report["edit_steps"] = step_meta

        comparable_inc = {k: v for k, v in inc_report.items() if k != "incremental_ledgers"}
        equivalent = comparable_inc == full_report
        reports[f"{label}_incremental"] = inc_report
        reports[f"{label}_full"] = full_report
        summaries[label] = {
            "concurrency": concurrency,
            "incremental_full_equivalent": equivalent,
            "incremental_versions": [item["global_version"] for item in ledgers],
        }
        _write_json(
            os.path.join(args.work_dir, f"report_incremental_{label}.json"), inc_report
        )
        _write_json(
            os.path.join(args.work_dir, f"report_full_{label}.json"), full_report
        )

    def _strip_runtime(report):
        clone = dict(report)
        manifest = dict(clone["manifest"])
        manifest.pop("concurrency_used", None)
        clone["manifest"] = manifest
        return clone

    cross_equal = _strip_runtime(reports["single_incremental"]) == _strip_runtime(
        reports["concurrent_incremental"]
    )
    summaries["order_consistency"] = {
        "single_vs_concurrent_equal": cross_equal,
    }
    _write_json(os.path.join(args.work_dir, "verify_summary.json"), summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    if not all(item["incremental_full_equivalent"] for item in summaries.values() if "incremental_full_equivalent" in item):
        sys.exit(1)
    if not cross_equal:
        sys.exit(2)



def build_parser():
    parser = argparse.ArgumentParser(prog="egw", description="Embedding/Gather 定点诊断工作台")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="生成确定性本地样本")
    gen.add_argument("--out-dir", required=True)
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="执行一次诊断并输出报告")
    run.add_argument("--config", required=True)
    run.add_argument("--rows", required=True)
    run.add_argument("--requests", required=True)
    run.add_argument("--edits")
    run.add_argument("--snapshot-dir")
    run.add_argument("--report", required=True)
    run.add_argument("--concurrency", type=int, default=None)
    run.add_argument("--resume", action="store_true", help="允许基于快照续跑")
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="增量/全量等价与并发顺序一致性验证")
    verify.add_argument("--config", required=True)
    verify.add_argument("--rows", required=True)
    verify.add_argument("--requests", required=True)
    verify.add_argument("--edits")
    verify.add_argument("--work-dir", required=True)
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except EGWError as exc:
        print(json.dumps({"code": exc.code, "message": exc.message, "context": exc.context},
                         ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

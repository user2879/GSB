"""命令行入口：样本生成、整链运行（含注入与预算）、快照恢复判定。"""

import argparse
import json
from pathlib import Path

from .config import load_config
from .errors import BusinessError
from .injection import load_injection_config
from .pipeline import Pipeline
from .samplegen import generate_dataset


def _write_report(base_dir: str, cfg, name: str, payload: dict) -> str:
    report_dir = Path(base_dir) / cfg.paths.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def cmd_generate(args) -> None:
    cfg = load_config(args.config)
    result = generate_dataset(cfg, args.base_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_run(args) -> None:
    cfg = load_config(args.config)
    injection = None
    if args.injection:
        injection = load_injection_config(
            args.injection if Path(args.injection).is_absolute()
            else f"{args.base_dir}/{args.injection}"
        )
    pipeline = Pipeline(cfg, args.base_dir, injection_config=injection,
                        snapshot=not args.no_snapshot)
    load_errors = []
    try:
        pipeline.load_all()
    except BusinessError as error:
        load_errors.append(error.to_record())
        if args.strict:
            raise
    report = {"load_errors": load_errors, "batches": []}
    budget = args.budget if args.budget is not None else cfg.runtime.default_budget
    while True:
        batch = pipeline.run_batch(budget=budget)
        report["batches"].append({
            "batch_id": batch.batch_id,
            "recomputed": batch.recomputed,
            "reused": batch.reused,
            "pending": batch.pending,
            "failures": batch.failures,
            "budget_remaining": batch.budget_remaining,
            "budget_pending": batch.budget_pending_records,
        })
        if not pipeline.pending and not pipeline.invalid_windows:
            break
        if not pipeline.pending:
            break
        if args.max_batches and len(report["batches"]) >= args.max_batches:
            break
    report["injection"] = pipeline.engine.report.as_dict()
    report["failures"] = pipeline.failures
    report["conclusions"] = [
        {
            "conclusion_id": item.conclusion_id,
            "code": item.code,
            "word": item.word,
            "window_indices": list(item.window_indices),
            "sample_span": list(item.sample_span),
            "boundary": item.boundary,
            "cache_key": item.cache_key,
        }
        for item in pipeline.conclusions
    ]
    report["windows"] = [
        {
            "window_index": window.window_index,
            "span": [window.start_sample, window.end_sample],
            "fragment_seqs": list(window.fragment_seqs),
            "fixed_point": list(pipeline.features[window.window_index].fixed_point)
            if window.window_index in pipeline.features else None,
            "posterior": pipeline.posteriors[window.window_index].posteriors
            if window.window_index in pipeline.posteriors else None,
            "boundary": pipeline.posteriors[window.window_index].boundary
            if window.window_index in pipeline.posteriors else None,
        }
        for window in sorted(pipeline.windows.values(), key=lambda w: w.window_index)
    ]
    path = _write_report(args.base_dir, cfg, args.report, report)
    print(f"诊断报告: {path}")
    print(json.dumps({
        "batches": len(report["batches"]),
        "windows": len(pipeline.windows),
        "conclusions": len(pipeline.conclusions),
        "pending": pipeline.pending,
        "failures": len(pipeline.failures),
        "injection_hits": len(report["injection"]["hits"]),
    }, indent=2, ensure_ascii=False))


def cmd_resume(args) -> None:
    cfg = load_config(args.config)
    pipeline = Pipeline(cfg, args.base_dir)
    decision = pipeline.resume_if_possible()
    print(json.dumps({"decision": decision}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voicecmd")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="生成确定性样本")
    gen.add_argument("--config", default="configs/pipeline.json")
    gen.add_argument("--base-dir", default=".")
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="运行识别链路")
    run.add_argument("--config", default="configs/pipeline.json")
    run.add_argument("--base-dir", default=".")
    run.add_argument("--injection", default=None)
    run.add_argument("--budget", type=int, default=None)
    run.add_argument("--max-batches", type=int, default=0)
    run.add_argument("--no-snapshot", action="store_true")
    run.add_argument("--strict", action="store_true")
    run.add_argument("--report", default="diagnostic_report.json")
    run.set_defaults(func=cmd_run)

    resume = sub.add_parser("resume-check", help="快照恢复判定")
    resume.add_argument("--config", default="configs/pipeline.json")
    resume.add_argument("--base-dir", default=".")
    resume.set_defaults(func=cmd_resume)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

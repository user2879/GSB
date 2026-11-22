from __future__ import annotations

import argparse
import json
import platform
import signal
import sys
from pathlib import Path
from typing import Any

from .config import AppConfig, config_from_mapping, load_config
from .demo import DEFAULT_SEED, generate_sample
from .engine import Engine
from .interactive import InteractiveSession, run_repl
from .loader import discover_inputs, load_inputs
from .util import failure, read_json, sort_failures
from .verify import run_verification


class GracefulInterrupt(Exception):
    pass


def _parse_switch_points(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default="sample", help="sample data directory (default: sample)")
    parser.add_argument("--state", default=".iidx", help="snapshot/cache state directory (default: .iidx)")


def _add_config_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tier", choices=["auto", "1", "2"], help="override compression tier policy")
    parser.add_argument("--chunk-size", type=int, help="terms per inverted block")
    parser.add_argument("--batch-size", type=int, help="maximum documents per scheduling batch")
    parser.add_argument("--threads", type=int, help="thread pool size; must not exceed quota")
    parser.add_argument("--global-budget", type=int, help="global term-processing token budget")
    parser.add_argument("--per-shard-quota", type=int, help="per-shard token quota")
    parser.add_argument("--snapshot-interval", type=int, help="commit a checkpoint every N batches")
    parser.add_argument("--latency", type=int, help="interactive latency budget in milliseconds")
    parser.add_argument("--fixed-bits", type=int, help="fixed-point score decimal bits")
    parser.add_argument("--quantization-tier", type=int, help="index into quantization_tiers")
    parser.add_argument("--switch-points", help="comma-separated auto-tier switch points")
    parser.add_argument("--low-frequency-threshold", type=int, help="tier-2 document-frequency threshold")
    parser.add_argument("--top-k", type=int, help="default query result limit")


def _collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "compression_tier": args.tier,
        "chunk_size": args.chunk_size,
        "batch_max_docs": args.batch_size,
        "thread_pool_size": args.threads,
        "global_token_budget": args.global_budget,
        "per_shard_token_quota": args.per_shard_quota,
        "snapshot_commit_interval_batches": args.snapshot_interval,
        "interactive_latency_budget_ms": args.latency,
        "fixed_decimal_bits": args.fixed_bits,
        "quantization_tier": args.quantization_tier,
        "low_frequency_threshold": args.low_frequency_threshold,
        "query_top_k": args.top_k,
    }
    if args.switch_points is not None:
        values["compression_switch_points"] = _parse_switch_points(args.switch_points)
    return {key: value for key, value in values.items() if value is not None}


def _runtime_overrides(state_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = state_dir / "runtime_config.json"
    if not path.exists():
        return {}, []
    try:
        value = read_json(path)
    except Exception as exc:
        return {}, [
            failure(
                "config",
                "E_RUNTIME_CONFIG_INVALID",
                "config:runtime",
                f"runtime configuration could not be read: {exc}",
                path=str(path),
            )
        ]
    if not isinstance(value, dict):
        return {}, [
            failure(
                "config",
                "E_RUNTIME_CONFIG_INVALID",
                "config:runtime",
                "runtime configuration root must be an object",
                path=str(path),
            )
        ]
    return value, []


def load_effective_config(
    data_dir: Path,
    state_dir: Path,
    overrides: dict[str, Any],
    *,
    use_runtime: bool = True,
) -> tuple[AppConfig, list[dict[str, Any]]]:
    paths = discover_inputs(data_dir)
    loaded = load_config(paths.config)
    merged = loaded.config.to_dict()
    failures = list(loaded.failures)
    if use_runtime:
        runtime, runtime_failures = _runtime_overrides(state_dir)
        failures.extend(runtime_failures)
        merged.update(runtime)
    merged.update(overrides)
    config, validation_failures = config_from_mapping(merged)
    failures.extend(validation_failures)
    return config, sort_failures(failures)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _run_once(args: argparse.Namespace, *, generated: dict[str, Any] | None = None) -> int:
    data_dir = Path(args.data).resolve()
    state_dir = Path(args.state).resolve()
    overrides = _collect_overrides(args)
    config, failures = load_effective_config(
        data_dir, state_dir, overrides, use_runtime=not getattr(args, "ignore_runtime", False)
    )
    stop_event = None
    if not getattr(args, "no_signal_handler", False):
        import threading

        stop_event = threading.Event()

        def request_stop(signum, frame):  # noqa: ARG001
            stop_event.set()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
    try:
        engine = Engine(
            data_dir=data_dir,
            state_dir=state_dir,
            config=config,
            config_failures=failures,
            stop_event=stop_event,
            expected_version=getattr(args, "expected_version", None),
        )
        result = engine.run()
    except Exception as exc:
        print(f"iidx: fatal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _print_json(result.report)
    else:
        report = result.report
        summary = report.get("summary", {})
        print(
            f"status={report.get('run', {}).get('status')} "
            f"version={report.get('run', {}).get('global_version')} "
            f"blocks={summary.get('block_count')} queries={summary.get('query_count')} "
            f"completed_shards={len(summary.get('completed_shards', []))} "
            f"budget={summary.get('consumed_token_budget')}/{config.global_token_budget} "
            f"tier={summary.get('compression_tier')}"
        )
        if summary.get("affected_shards"):
            print(f"affected_shards={','.join(summary['affected_shards'])}")
        if report.get("failures"):
            codes = ",".join(item["code"] for item in report["failures"])
            print(f"failures={len(report['failures'])} codes={codes}")
        for event in report.get("diagnostics", {}).get("snapshot_events", []):
            print(f"snapshot_warning={event.get('code')} {event.get('message')}")
        if result.interrupted:
            print("graceful_stop=checkpoint_committed")
        print(f"report={result.report_path}")
        if result.snapshot_path:
            print(f"snapshot={result.snapshot_path}")
        if generated is not None:
            print(f"generated={generated.get('generated')} seed={generated.get('seed')}")
    return result.exit_code


def cmd_demo(args: argparse.Namespace) -> int:
    generated = generate_sample(
        Path(args.data), seed=args.seed, force=args.force, include_invalid=args.include_invalid
    )
    return _run_once(args, generated=generated)


def cmd_run(args: argparse.Namespace) -> int:
    return _run_once(args)


def cmd_validate(args: argparse.Namespace) -> int:
    data_dir = Path(args.data).resolve()
    paths = discover_inputs(data_dir)
    loaded = load_config(paths.config)
    load_result = load_inputs(data_dir, loaded.config, None)
    failures = sort_failures(loaded.failures + load_result.failures)
    payload = {
        "ok": not any(item.get("code", "").startswith("E_INPUT_FILE") for item in failures),
        "data_dir": str(data_dir),
        "config": loaded.config.to_dict(),
        "stats": load_result.stats,
        "source_fingerprint": load_result.source_fingerprint,
        "failure_count": len(failures),
        "failures": failures,
        "shard_fingerprints": load_result.shard_fingerprints,
        "documents_by_shard": {
            shard: [doc.doc_id for doc in docs]
            for shard, docs in load_result.documents_by_shard.items()
        },
    }
    if args.json:
        _print_json(payload)
    else:
        print(
            f"documents={load_result.stats['accepted_documents']} "
            f"queries={load_result.stats['accepted_queries']} "
            f"vocabulary={load_result.stats['vocabulary_size']} "
            f"shards={load_result.stats['shards']} failures={len(failures)}"
        )
        for item in failures:
            print(f"{item['code']} {item['stable_key']} {item['message']}")
    return 0 if payload["ok"] else 2


def cmd_interactive(args: argparse.Namespace) -> int:
    data_dir = Path(args.data).resolve()
    state_dir = Path(args.state).resolve()
    overrides = _collect_overrides(args)
    config, failures = load_effective_config(
        data_dir, state_dir, overrides, use_runtime=not args.ignore_runtime
    )
    session = InteractiveSession(
        data_dir=data_dir,
        state_dir=state_dir,
        config=config,
        config_failures=failures,
    )

    def interrupt_handler(signum, frame):  # noqa: ARG001
        session.stop_event.set()
        raise GracefulInterrupt()

    signal.signal(signal.SIGTERM, interrupt_handler)
    signal.signal(signal.SIGINT, interrupt_handler)
    try:
        if args.command:
            for command in args.command:
                response = session.execute_command(command)
                _print_json(response)
                if session.quit_requested:
                    break
            session.close()
            return 0
        return run_repl(session)
    except GracefulInterrupt:
        session.close()
        return 0
    except Exception as exc:
        print(f"iidx: interactive error: {type(exc).__name__}: {exc}", file=sys.stderr)
        session.close()
        return 2


def cmd_verify(args: argparse.Namespace) -> int:
    work_root = Path(args.work_dir).resolve()
    result = run_verification(work_root, seed=args.seed)
    if args.json:
        _print_json(result)
    else:
        print(f"verification_ok={result['ok']} seed={result['seed']}")
        print(f"steps={len(result['steps'])}")
        for step in result["steps"]:
            comparison = step["comparison"]
            print(
                f"  {step['step']}: equivalent={comparison['equivalent']} "
                f"digest={comparison['incremental_digest']}"
            )
            if not comparison["equivalent"]:
                print(f"    first_difference={comparison['first_difference']}")
        final = result["final"]
        print(
            f"blocks={final['block_count']} tiers={final['selected_tiers_present']} "
            f"two_tier_lengths_differ={final['all_two_tier_bytes_differ']} "
            f"decode_ok={final['all_decode_checksums_ok']}"
        )
        print(f"resume_ok={result['resume_verification']['ok']}")
        print(f"budget_scheduler_ok={result['budget_verification']['ok']}")
        print(f"validation_ok={result['validation_verification']['ok']}")
        print(f"report={work_root / 'verification-report.json'}")
    return 0 if result["ok"] else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_ok": sys.version_info >= (3, 10),
    }
    try:
        import numpy

        payload["numpy_version"] = numpy.__version__
        payload["numpy_ok"] = True
    except Exception as exc:  # pragma: no cover - environment dependent
        payload["numpy_ok"] = False
        payload["numpy_error"] = f"{type(exc).__name__}: {exc}"
    payload["ok"] = bool(payload["python_ok"] and payload["numpy_ok"])
    if args.json:
        _print_json(payload)
    else:
        print(f"python={payload['python_version']} ok={payload['python_ok']}")
        print(f"numpy={payload.get('numpy_version', 'missing')} ok={payload['numpy_ok']}")
    return 0 if payload["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iidx",
        description="Deterministic inverted-index builder, encoder and budget scheduler",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="generate a deterministic sample and run it")
    _add_common_paths(demo)
    _add_config_overrides(demo)
    demo.add_argument("--seed", type=int, default=DEFAULT_SEED)
    demo.add_argument("--force", action="store_true", help="regenerate the sample files")
    demo.add_argument("--include-invalid", action="store_true", help="include deliberate validation failures")
    demo.add_argument("--json", action="store_true")
    demo.add_argument("--ignore-runtime", action="store_true")
    demo.add_argument("--expected-version", type=int)
    demo.set_defaults(func=cmd_demo)

    init_demo = subparsers.add_parser("init-demo", help="generate the deterministic sample only")
    init_demo.add_argument("--data", default="sample")
    init_demo.add_argument("--seed", type=int, default=DEFAULT_SEED)
    init_demo.add_argument("--force", action="store_true")
    init_demo.add_argument("--include-invalid", action="store_true")
    init_demo.add_argument("--json", action="store_true")

    def init_demo_command(args: argparse.Namespace) -> int:
        generated = generate_sample(
            Path(args.data), seed=args.seed, force=args.force, include_invalid=args.include_invalid
        )
        if args.json:
            _print_json(generated)
        else:
            print(f"generated={generated['generated']} workspace={generated['workspace']} seed={generated['seed']}")
        return 0

    init_demo.set_defaults(func=init_demo_command)

    run = subparsers.add_parser("run", help="validate inputs, build/cache index phases and emit a report")
    _add_common_paths(run)
    _add_config_overrides(run)
    run.add_argument("--json", action="store_true")
    run.add_argument("--ignore-runtime", action="store_true")
    run.add_argument("--expected-version", type=int)
    run.set_defaults(func=cmd_run)

    validate = subparsers.add_parser("validate", help="validate documents, queries, vocabulary and config")
    validate.add_argument("--data", default="sample")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=cmd_validate)

    interactive = subparsers.add_parser("interactive", help="launch the latency-budgeted interactive shell")
    _add_common_paths(interactive)
    _add_config_overrides(interactive)
    interactive.add_argument("--command", action="append", help="execute one command and print JSON; repeatable")
    interactive.add_argument("--ignore-runtime", action="store_true")
    interactive.set_defaults(func=cmd_interactive)

    verify = subparsers.add_parser("verify", help="run incremental/full, resume, budget and validation acceptance suite")
    verify.add_argument("--work-dir", default=".iidx/verification")
    verify.add_argument("--seed", type=int, default=DEFAULT_SEED)
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=cmd_verify)

    doctor = subparsers.add_parser("doctor", help="check Python and NumPy environment prerequisites")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))

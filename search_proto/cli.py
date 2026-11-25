from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import PrototypeConfig
from .engine import SearchEngine
from .errors import Failure, InputValidationError
from .generator import generate_samples
from .loader import LoadedInputs, load_inputs
from .model import StopwordTable
from .report import compare_engines, write_run_outputs
from .snapshot import SnapshotStore
from .util import read_json, write_json


def _resume_summary(info: dict[str, Any]) -> dict[str, Any]:
    keys = ("decision", "reason", "status", "role", "version_name", "validation")
    return {key: info.get(key) for key in keys if key in info}


def _configured_edit_target(
    inputs: LoadedInputs,
) -> tuple[PrototypeConfig, StopwordTable]:
    config = inputs.config
    stopwords = inputs.stopwords
    for command in inputs.edits:
        if command.op == "set_field_weights":
            config = config.with_field_weights(command.payload["weights"])
        elif command.op == "set_bm25":
            config = config.with_bm25(
                command.payload["k1"], command.payload["b"]
            )
        elif command.op == "set_fixed":
            config = config.with_fixed(
                command.payload["decimal_places"], command.payload["quantum"]
            )
        elif command.op == "set_top_k":
            config = config.with_top_k(command.payload["top_k"])
        elif command.op == "add_stopword":
            stopwords = stopwords.add(
                command.payload["field"], command.payload["term"]
            )
        elif command.op == "remove_stopword":
            stopwords = stopwords.remove(
                command.payload["field"], command.payload["term"]
            )
    return config, stopwords


def _within_workspace(path: Path, workspace: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"路径必须位于工作目录内: {resolved}") from exc
    return resolved


def _probe_temp(workspace: Path) -> None:
    probe_dir = workspace / ".runtime-tmp"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_dir / "write-probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()


def _write_bootstrap_failure(out_dir: Path, failures: list[Failure]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    failure_dicts = [failure.to_dict() for failure in sorted(failures, key=Failure.sort_key)]
    counts: dict[str, int] = {}
    for failure in failures:
        counts[failure.code] = counts.get(failure.code, 0) + 1
    write_json(out_dir / "failures.json", failure_dicts)
    write_json(
        out_dir / "report.json",
        {
            "schema": "search-prototype-report-v1",
            "status": "input_failed",
            "business_code_counts": {key: counts[key] for key in sorted(counts)},
            "failures": failure_dicts,
            "queries": [],
        },
    )


def _fresh_engine(
    inputs: LoadedInputs,
    *,
    serial: bool,
    failures: list[Failure] | None = None,
    use_edit_final_state: bool = False,
) -> SearchEngine:
    if use_edit_final_state:
        pre_engine = SearchEngine(
            inputs.config,
            inputs.stopwords,
            inputs.vocabulary,
            inputs.documents,
            inputs.queries,
            failures=inputs.failures,
            max_workers_override=1 if serial else None,
        )
        pre_engine.apply_edits(inputs.edits)
        engine = SearchEngine(
            pre_engine.config,
            pre_engine.stopwords,
            pre_engine.vocabulary,
            pre_engine.ordered_documents,
            pre_engine.ordered_queries,
            failures=pre_engine.sorted_failures(),
            max_workers_override=1 if serial else None,
        )
    else:
        engine = SearchEngine(
            inputs.config,
            inputs.stopwords,
            inputs.vocabulary,
            inputs.documents,
            inputs.queries,
            failures=inputs.failures if failures is None else failures,
            max_workers_override=1 if serial else None,
        )
    engine.input_fingerprint = inputs.input_fingerprint
    return engine


def _run_full(
    inputs: LoadedInputs,
    *,
    serial: bool,
    write_output: bool,
    out_dir: Path,
) -> SearchEngine:
    engine = _fresh_engine(inputs, serial=serial, use_edit_final_state=True)
    engine.materialize()
    engine.global_version = max(engine.global_version, 1)
    engine.run_batches()
    if write_output:
        write_run_outputs(engine, out_dir)
    return engine


def _run_incremental(
    inputs: LoadedInputs,
    *,
    out_dir: Path,
    snapshot_dir: Path,
    serial: bool,
    resume: bool,
    no_snapshot: bool,
    expected_global_version: int | None,
    crash_after_batch: int | None,
    warm_cache: bool,
) -> tuple[SearchEngine, dict[str, Any]]:
    store = SnapshotStore(snapshot_dir)
    target_config, target_stopwords = _configured_edit_target(inputs)
    resume_info: dict[str, Any] = {"decision": "disabled" if no_snapshot else "fresh"}
    engine: SearchEngine | None = None
    if resume and not no_snapshot:
        resume_info = store.load_for_resume(
            source_fingerprint=inputs.input_fingerprint,
            target_config_hash=target_config.full_fingerprint,
            target_stopword_version=target_stopwords.effective_version,
            expected_global_version=expected_global_version,
        )
        if resume_info.get("decision") == "resume":
            engine = SearchEngine.from_state_dict(resume_info["state"])
            engine.input_fingerprint = inputs.input_fingerprint
            engine.max_workers_override = 1 if serial else None

    if engine is None:
        engine = _fresh_engine(inputs, serial=serial)
        engine.materialize()
        engine.global_version = max(engine.global_version, 1)

        if not no_snapshot:
            store.commit(engine)

        if inputs.edits and warm_cache:
            engine.run_batches()
            engine.apply_edits(inputs.edits)
            engine.materialize()
            engine.global_version += 1
            if not no_snapshot:
                store.commit(engine)
        elif inputs.edits:
            engine.apply_edits(inputs.edits)
            engine.materialize()
            engine.global_version += 1
            if not no_snapshot:
                store.commit(engine)

    def snapshot_callback(current: SearchEngine) -> None:
        if not no_snapshot:
            store.commit(current)

    engine.run_batches(
        snapshot_callback=None if no_snapshot else snapshot_callback,
        crash_after_batch=crash_after_batch,
    )
    if not no_snapshot:
        store.commit(engine)
    write_run_outputs(engine, out_dir)
    write_json(
        out_dir / "run_meta.json",
        {
            "mode": "incremental",
            "source_fingerprint": inputs.input_fingerprint,
            "resume": resume_info,
            "snapshot_dir": str(snapshot_dir),
            "warm_cache": warm_cache,
            "global_version": engine.global_version,
        },
    )
    return engine, resume_info


def _run_both(
    inputs: LoadedInputs,
    *,
    out_dir: Path,
    snapshot_dir: Path,
    serial: bool,
    resume: bool,
    no_snapshot: bool,
    expected_global_version: int | None,
    crash_after_batch: int | None,
    warm_cache: bool,
) -> dict[str, Any]:
    incremental_dir = out_dir / "incremental"
    full_dir = out_dir / "full"
    incremental, resume_info = _run_incremental(
        inputs,
        out_dir=incremental_dir,
        snapshot_dir=snapshot_dir,
        serial=serial,
        resume=resume,
        no_snapshot=no_snapshot,
        expected_global_version=expected_global_version,
        crash_after_batch=crash_after_batch,
        warm_cache=warm_cache,
    )
    full = _run_full(inputs, serial=serial, write_output=True, out_dir=full_dir)
    comparison = compare_engines(incremental, full)
    comparison["resume"] = resume_info
    write_json(out_dir / "comparison.json", comparison)
    write_json(
        out_dir / "report.json",
        {
            "schema": "search-prototype-combined-report-v1",
            "comparison": comparison,
            "incremental_report": read_json(incremental_dir / "report.json"),
            "full_report": read_json(full_dir / "report.json"),
        },
    )
    return comparison


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--stopwords", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--edits", type=Path)


def _load(inputs_args: argparse.Namespace) -> LoadedInputs:
    return load_inputs(
        inputs_args.config,
        inputs_args.vocabulary,
        inputs_args.stopwords,
        inputs_args.documents,
        inputs_args.queries,
        inputs_args.edits,
    )


def _command_generate(args: argparse.Namespace, workspace: Path) -> int:
    out_dir = _within_workspace(args.out_dir, workspace)
    manifest = generate_samples(
        out_dir,
        seed=args.seed,
        document_count=args.documents,
        query_count=args.queries,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _command_validate(args: argparse.Namespace) -> int:
    try:
        inputs = _load(args)
    except InputValidationError as exc:
        payload = [failure.to_dict() for failure in exc.failures]
        print(json.dumps({"status": "invalid", "failures": payload}, ensure_ascii=False, indent=2))
        return 2
    payload = {
        "status": "ok" if not inputs.failures else "invalid_rows",
        "document_count": len(inputs.documents),
        "query_count": len(inputs.queries),
        "edit_count": len(inputs.edits),
        "input_fingerprint": inputs.input_fingerprint,
        "failures": [failure.to_dict() for failure in inputs.failures],
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if not inputs.failures else 2


def _command_run(args: argparse.Namespace, workspace: Path) -> int:
    _probe_temp(workspace)
    out_dir = _within_workspace(args.out_dir, workspace)
    snapshot_dir = _within_workspace(args.snapshot_dir, workspace)
    try:
        inputs = _load(args)
    except InputValidationError as exc:
        _write_bootstrap_failure(out_dir, exc.failures)
        print(json.dumps({"status": "input_failed", "failures": [f.to_dict() for f in exc.failures]}, ensure_ascii=False, indent=2))
        return 2

    if args.mode == "both":
        comparison = _run_both(
            inputs,
            out_dir=out_dir,
            snapshot_dir=snapshot_dir,
            serial=args.serial,
            resume=args.resume,
            no_snapshot=args.no_snapshot,
            expected_global_version=args.expected_global_version,
            crash_after_batch=args.crash_after_batch,
            warm_cache=not args.no_warm_cache,
        )
        print(
            json.dumps(
                {
                    "status": "ok" if comparison["equal"] else "mismatch",
                    "comparison": comparison,
                    "out_dir": str(out_dir),
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0 if comparison["equal"] else 1

    if args.mode == "full":
        engine = _run_full(inputs, serial=args.serial, write_output=True, out_dir=out_dir)
        report = engine.report()
        print(
            json.dumps(
                {
                    "status": "ok",
                    "mode": "full",
                    "out_dir": str(out_dir),
                    "summary": {
                        "queries": len(report["queries"]),
                        "business_code_counts": report["business_code_counts"],
                        "order_flip_count": report["order_flip_count"],
                        "boundary_document_count": report["boundary_document_count"],
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    engine, resume_info = _run_incremental(
        inputs,
        out_dir=out_dir,
        snapshot_dir=snapshot_dir,
        serial=args.serial,
        resume=args.resume,
        no_snapshot=args.no_snapshot,
        expected_global_version=args.expected_global_version,
        crash_after_batch=args.crash_after_batch,
        warm_cache=not args.no_warm_cache,
    )
    report = engine.report()
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": "incremental",
                "resume": _resume_summary(resume_info),
                "out_dir": str(out_dir),
                "summary": {
                    "queries": len(report["queries"]),
                    "business_code_counts": report["business_code_counts"],
                    "cache_entry_counts": report["cache_entry_counts"],
                    "order_flip_count": report["order_flip_count"],
                    "boundary_document_count": report["boundary_document_count"],
                    "concurrency": report["concurrency"],
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def _command_index(args: argparse.Namespace, workspace: Path) -> int:
    _probe_temp(workspace)
    out_dir = _within_workspace(args.out_dir, workspace)
    try:
        inputs = _load(args)
    except InputValidationError as exc:
        _write_bootstrap_failure(out_dir, exc.failures)
        return 2
    engine = _fresh_engine(inputs, serial=args.serial, use_edit_final_state=True)
    engine.materialize()
    write_run_outputs(engine, out_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "index_fingerprint": engine.index_fingerprint,
                "content_fingerprint": engine.content_fingerprint,
                "document_count": len(engine.documents),
                "term_count": len(engine.term_stats),
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search-proto",
        description="Deterministic local inverted-index / fixed-point ranking prototype",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local samples")
    generate.add_argument("--out-dir", type=Path, default=Path("samples"))
    generate.add_argument("--seed", type=int, default=20261126)
    generate.add_argument("--documents", type=int, default=24)
    generate.add_argument("--queries", type=int, default=8)

    validate = subparsers.add_parser("validate", help="validate all input files")
    _add_input_arguments(validate)

    run = subparsers.add_parser("run", help="build, score, rank and write auditable outputs")
    _add_input_arguments(run)
    run.add_argument("--out-dir", type=Path, default=Path("artifacts/run"))
    run.add_argument("--snapshot-dir", type=Path, default=Path("artifacts/state"))
    run.add_argument("--mode", choices=("incremental", "full", "both"), default="incremental")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--expected-global-version", type=int)
    run.add_argument("--serial", action="store_true", help="force max_workers=1")
    run.add_argument("--no-snapshot", action="store_true")
    run.add_argument("--no-warm-cache", action="store_true", help="do not pre-populate caches before edits")
    run.add_argument("--crash-after-batch", type=int)

    index = subparsers.add_parser("index", help="build final index without running queries")
    _add_input_arguments(index)
    index.add_argument("--out-dir", type=Path, default=Path("artifacts/index"))
    index.add_argument("--serial", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = Path.cwd().resolve()
    if args.command == "generate":
        return _command_generate(args, workspace)
    if args.command == "validate":
        return _command_validate(args)
    if args.command == "run":
        return _command_run(args, workspace)
    if args.command == "index":
        return _command_index(args, workspace)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

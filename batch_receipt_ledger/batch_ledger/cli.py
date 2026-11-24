from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .engine import LedgerEngine, build_report
from .engine import config_fingerprint
from .samples import write_sample_workspace
from .snapshot import SnapshotStore
from .snapshot import fingerprint_receipt_files
from .state import apply_command, clone_state
from .view import interactive_view
from .worker import infer_over_loopback, launch_worker_subprocess
from .workspace import load_workspace_dir


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cmd_init_samples(args: argparse.Namespace) -> int:
    path = write_sample_workspace(args.dir)
    print(json.dumps({"ok": True, "sample_dir": str(path)}, ensure_ascii=False))
    return 0


def _worker_demo(out: Path) -> dict[str, Any]:
    project = _project_root()
    processes = []
    try:
        w1, port1 = launch_worker_subprocess("W1-demo", preferred=29181, project_root=project)
        processes.append(w1)
        w2, port2 = launch_worker_subprocess("W2-demo", preferred=29182, project_root=project)
        processes.append(w2)
        started = time.perf_counter()
        response1 = infer_over_loopback(
            port1,
            {
                "batch_id": "B-DEMO-1",
                "requests": [
                    {"request_id": "Q0001", "request_number": 1},
                    {"request_id": "Q0002", "request_number": 2},
                ],
            },
        )
        response2 = infer_over_loopback(
            port2,
            {
                "batch_id": "B-DEMO-2",
                "requests": [{"request_id": "Q0005", "request_number": 5}],
            },
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        result = {
            "ok": response1["ok"] and response2["ok"],
            "bind_host": "127.0.0.1",
            "workers": [
                {"worker_id": "W1-demo", "port": port1, "in_range": 20000 <= port1 <= 39999},
                {"worker_id": "W2-demo", "port": port2, "in_range": 20000 <= port2 <= 39999},
            ],
            "roundtrip_elapsed_ms": elapsed_ms,
            "response_digests": {
                "W1-demo": response1["output_digests"],
                "W2-demo": response2["output_digests"],
            },
            "numpy_backend": response1.get("compute_backend"),
        }
        write_json(out / "worker_demo.json", result)
        return result
    finally:
        for process in processes:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
            finally:
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out)
    sample_dir = Path(args.sample_dir)
    write_sample_workspace(sample_dir)
    state = load_workspace_dir(sample_dir)

    initial = build_report(state, parallel=False)
    write_json(out / "initial_report.json", initial)

    engine = LedgerEngine(clone_state(state))
    engine.run(parallel=True)
    command_results = []
    cache_trace = []
    for command in state.get("command_stream", []):
        result = engine.apply(command, parallel=True)
        command_results.append(result)
        cache_trace.append({"command": command.get("type"), **engine.last_cache_stats})
    final = engine.current_report
    assert final is not None
    full = engine.full_rebuild(parallel=False)
    parallel = build_report(clone_state(engine.state), parallel=True)
    consistency = {
        "incremental_vs_blank_full_equal": engine.incremental_matches_full(parallel=True),
        "parallel_vs_serial_equal": final == parallel,
        "report_hash": final["report_hash"],
        "cache_trace": cache_trace,
    }

    negative_state = clone_state(state)
    negative_result = apply_command(negative_state, {"type": "set_batch_size", "batch_size": 0})
    # Simulate loading a workspace whose persisted config was edited to zero.
    negative_state["config"]["batch_size"] = 0
    negative_report = build_report(negative_state)
    negative = {
        "command_result": negative_result,
        "failure_codes": negative_report["summary"]["failure_code_counts"],
        "failures": [item for item in negative_report["failures"] if item["code"] == "E_BATCH_SIZE_INVALID"],
    }

    request_ids = [row["request_id"] for row in final["decisions"]]
    partial_ids = request_ids[: len(request_ids) // 2]
    view = interactive_view(final, completed_request_ids=partial_ids, budget_ms=args.view_budget_ms)
    divergence_demo_state = clone_state(final)
    divergent_view = interactive_view(
        final,
        completed_request_ids=request_ids[:1],
        previous_summary=view["summary"],
        mutate_fast_row=lambda row: {**row, "basis": "INJECTED_DIVERGENCE"},
    )

    final_input_dir = out / "final_inputs"
    final_receipts_path = final_input_dir / "receipts" / "final.jsonl"
    final_receipts_path.parent.mkdir(parents=True, exist_ok=True)
    with final_receipts_path.open("w", encoding="utf-8") as handle:
        for receipt in engine.state.get("receipts", []):
            public = {key: value for key, value in receipt.items() if not key.startswith("_")}
            handle.write(json.dumps(public, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(final_input_dir / "config.json", engine.state["config"])

    store = SnapshotStore(out / "snapshot")
    manifest = store.commit(
        final,
        engine.state,
        input_dir=final_input_dir,
        command_results=command_results,
    )
    recovery = store.recovery_decision(
        expected_global_version=manifest["global_version"],
        expected_config_hash=manifest["config_hash"],
        expected_receipt_fingerprints=manifest["receipt_fingerprints"],
    )

    worker_demo = None if args.skip_worker else _worker_demo(out)
    artifacts = {
        "ok": True,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sample_dir": str(sample_dir),
        "out_dir": str(out),
        "summary": final["summary"],
        "consistency": consistency,
        "negative_batch_size": negative,
        "view": {
            "consistent_partial": {k: v for k, v in view.items() if k != "summary"},
            "injected_divergence": divergent_view,
        },
        "snapshot": {"manifest": manifest, "recovery": recovery},
        "worker_demo": worker_demo,
        "key_invalidation": final["invalidation"],
    }
    write_json(out / "command_results.json", command_results)
    write_json(out / "cache_trace.json", cache_trace)
    write_json(out / "final_report.json", final)
    write_json(out / "view_result.json", {"normal": view, "divergence_demo": divergent_view})
    write_json(out / "snapshot_manifest.json", manifest)
    write_json(out / "run_report.json", artifacts)
    print(json.dumps({"ok": True, "run_report": str(out / "run_report.json"), "summary": final["summary"], "consistency": consistency}, ensure_ascii=False, indent=2))
    return 0


def cmd_run_workspace(args: argparse.Namespace) -> int:
    state = load_workspace_dir(args.dir)
    report = build_report(state, parallel=args.parallel)
    if args.out:
        write_json(Path(args.out), report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui import launch_ui

    return launch_ui(args.report)


def cmd_recover(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    config_path = input_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    expected_version = int(config.get("global_version", 1))
    decision = SnapshotStore(args.snapshot_dir).recovery_decision(
        expected_global_version=expected_version,
        expected_config_hash=config_fingerprint(config),
        expected_receipt_fingerprints=fingerprint_receipt_files(input_dir),
    )
    public = {key: value for key, value in decision.items() if key != "manifest"}
    if args.show_manifest:
        public["manifest"] = decision.get("manifest")
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if decision["resume"] else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="batch-ledger", description="Deterministic batch receipt ledger")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init-samples")
    init_parser.add_argument("--dir", default="samples/local_demo")
    init_parser.set_defaults(func=cmd_init_samples)

    demo_parser = sub.add_parser("demo")
    demo_parser.add_argument("--sample-dir", default="samples/local_demo")
    demo_parser.add_argument("--out", default="artifacts/demo_run")
    demo_parser.add_argument("--view-budget-ms", type=int, default=250)
    demo_parser.add_argument("--skip-worker", action="store_true")
    demo_parser.set_defaults(func=cmd_demo)

    run_parser = sub.add_parser("run-workspace")
    run_parser.add_argument("--dir", required=True)
    run_parser.add_argument("--out")
    run_parser.add_argument("--parallel", action="store_true")
    run_parser.set_defaults(func=cmd_run_workspace)

    ui_parser = sub.add_parser("ui")
    ui_parser.add_argument("--report", default="artifacts/demo_run/final_report.json")
    ui_parser.set_defaults(func=cmd_ui)

    recover_parser = sub.add_parser("recover")
    recover_parser.add_argument("--snapshot-dir", required=True)
    recover_parser.add_argument("--input-dir", required=True)
    recover_parser.add_argument("--show-manifest", action="store_true")
    recover_parser.set_defaults(func=cmd_recover)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""batchdiag command line interface."""

import argparse
import socket
import subprocess
import sys
import time

from . import samples as samples_mod
from .canonical import dumps
from .engine import (recover, run_full, run_incremental, simulate_force_terminate,
                     verify_byte_identical)
from .invalidation import recompute_all, results_to_bytes
from .ledger import open_ledger, replay_check
from .planner import build_plan
from .receipts import decode_receipts_file
from .scheduler import schedule_all
from .spec import load_spec
from .worker import ensure_port_free


def _print(payload):
    print(dumps(payload))


def _wait_port(host, port, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.2)
        try:
            probe.connect((host, port))
            probe.close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


def start_workers(spec, stall_ms=0):
    """Start one real subprocess per worker and wait for its listen socket."""
    processes = []
    for worker in sorted(spec.workers, key=lambda w: w.worker_id):
        ensure_port_free(worker.host, worker.port)
    counts = {w.worker_id: 0 for w in spec.workers}
    plan = build_plan(spec)
    for batch in plan.batches:
        counts[batch.worker_id] += 1
    for worker in sorted(spec.workers, key=lambda w: w.worker_id):
        cmd = [sys.executable, "-m", "batchdiag.__main_worker__",
               "--host", worker.host, "--port", str(worker.port),
               "--exit-after", str(max(counts[worker.worker_id], 1))]
        if stall_ms and worker.worker_id == spec.workers[0].worker_id:
            cmd += ["--stall-ms", str(stall_ms)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        processes.append(proc)
        if not _wait_port(worker.host, worker.port):
            for other in processes:
                other.kill()
            raise RuntimeError(
                f"worker {worker.worker_id} did not start on "
                f"{worker.host}:{worker.port}")
    return processes


def stop_workers(processes):
    for proc in processes:
        proc.terminate()
    for proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def cmd_gen_samples(args):
    summary = samples_mod.generate(args.samples_dir)
    _print(summary)


def cmd_validate(args):
    spec = load_spec(args.requests, args.workers, args.rules)
    _print({"ok": True,
            "requests": len(spec.requests),
            "workers": len(spec.workers),
            "model_dim": spec.model_dim,
            "global_concurrency": spec.merge_rules.global_concurrency})


def cmd_normalize(args):
    spec = load_spec(args.requests, args.workers, args.rules)
    plan = build_plan(spec)
    records = decode_receipts_file(
        args.receipts,
        known_request_ids={r.request_id for r in spec.requests},
        batch_sizes=plan.batch_sizes)
    _print({
        "source": str(args.receipts),
        "count": len(records),
        "formats": sorted({r.source_format for r in records}),
        "records": [r.to_dict() for r in records],
    })


def cmd_plan(args):
    spec = load_spec(args.requests, args.workers, args.rules)
    plan = build_plan(spec)
    _print({
        "batches": [
            {"batch_id": b.batch_id, "worker_id": b.worker_id,
             "batch_seq": b.batch_seq, "arrivals": list(b.arrivals),
             "rank_order": list(b.requests)}
            for b in plan.batches
        ],
        "rule": "sort(arrival,request_id); round-robin workers; "
                "batch_id=B:<worker>:<seq>:sha256(canonical(arrivals,seq))[:12]",
    })


def cmd_run(args):
    spec = load_spec(args.requests, args.workers, args.rules)
    plan, results, receipts, _ = run_full(
        spec, args.ledger, receipt_fmt=args.receipt_format)
    _print({"ok": True, "ledger": args.ledger,
            "batches": len(results),
            "batch_ids": [b.batch_id for b in plan.batches],
            "output_digests": {k: v.output_digest for k, v in
                                sorted(results.items())}})


def cmd_incremental(args):
    old = load_spec(args.requests, args.workers, args.rules)
    new = load_spec(args.requests, args.new_workers, args.rules)
    run_full(old, args.ledger)
    plan, fresh, report, status = run_incremental(
        old, new, args.ledger)
    full = recompute_all(new, plan)
    incremental_bytes = results_to_bytes(fresh)
    full_bytes = results_to_bytes(full)
    import hashlib
    _print({
        "snapshot": status.__dict__,
        "field_diffs": report.field_diffs,
        "invalid_batches": report.invalid_batches,
        "invalid_requests": report.invalid_requests,
        "incremental_equals_full": incremental_bytes == full_bytes,
        "incremental_byte_len": len(incremental_bytes),
        "full_byte_len": len(full_bytes),
        "byte_digest": hashlib.sha256(full_bytes).hexdigest(),
    })


def cmd_verify(args):
    from .spec import Spec
    spec_a = load_spec(args.requests, args.workers, args.rules)
    shuffled_requests = list(reversed(spec_a.requests))
    spec_b = Spec(requests=shuffled_requests, workers=spec_a.workers,
                  merge_rules=spec_a.merge_rules, model_dim=spec_a.model_dim,
                  sources=spec_a.sources, line_maps=spec_a.line_maps)
    result = verify_byte_identical(spec_a, spec_b, args.ledger_a,
                                   args.ledger_b)
    _print(result)


def cmd_scheduler(args):
    spec = load_spec(args.requests, args.workers, args.rules)
    plan = build_plan(spec)
    stall = args.stall_ms if args.timeout_demo else 0
    processes = start_workers(spec, stall_ms=stall)
    try:
        results, trace = schedule_all(spec, plan)
    finally:
        stop_workers(processes)
    _print({
        "global_limit": trace.global_limit,
        "worker_limits": trace.worker_limits,
        "max_global_in_flight": trace.max_global_in_flight,
        "max_per_worker_in_flight": trace.max_per_worker_in_flight,
        "records": [r.__dict__ for r in trace.records],
        "errors": [r.__dict__ for r in trace.records if r.status == "error"],
        "ok_batches": len(results),
    })


def cmd_snapshot(args):
    spec = load_spec(args.requests, args.workers, args.rules)
    run_full(spec, args.ledger)
    ledger, status = open_ledger(args.ledger)
    before = status.__dict__
    ledger.close()
    if args.tamper:
        conn = __import__("sqlite3").connect(args.ledger)
        with conn:
            conn.execute(
                "UPDATE batches SET output_digest=? WHERE batch_id="
                "(SELECT batch_id FROM batches LIMIT 1)",
                ("tampered-digest",))
        conn.close()
    ledger, status = open_ledger(args.ledger)
    ledger.close()
    _print({"first_open": before, "reopen": status.__dict__})


def cmd_crash(args):
    spec = load_spec(args.requests, args.workers, args.rules)
    simulate_force_terminate(spec, args.ledger, crash_after=args.after)


def cmd_recover(args):
    spec = load_spec(args.requests, args.workers, args.rules)
    outcome = recover(spec, args.ledger)
    _print({
        "snapshot": outcome["status"].__dict__,
        "unknown_batches": outcome["unknown_batches"],
        "committed_batch_ids": outcome["committed_batch_ids"],
        "replayed_ok": outcome["replayed_ok"],
    })


def cmd_replay(args):
    spec = load_spec(args.requests, args.workers, args.rules)
    plan = build_plan(spec)
    results = recompute_all(spec, plan)
    ledger, status = open_ledger(args.ledger)
    try:
        replay_check(ledger, plan, results)
        _print({"replay_ok": True, "snapshot": status.__dict__})
    finally:
        ledger.close()


def cmd_gui(args):
    try:
        from .gui import launch
        launch(args.requests, args.workers, args.rules, args.ledger)
    except Exception as exc:
        _print({"gui_available": False, "reason": str(exc),
                "fallback": "use CLI commands; this headless host has no Tk"})


def build_parser():
    parser = argparse.ArgumentParser(prog="batchdiag")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gen-samples")
    p.add_argument("--samples-dir", default="samples")
    p.set_defaults(func=cmd_gen_samples)

    p = sub.add_parser("validate")
    p.add_argument("--requests", required=True)
    p.add_argument("--workers", required=True)
    p.add_argument("--rules", required=True)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("normalize")
    p.add_argument("--requests", required=True)
    p.add_argument("--workers", required=True)
    p.add_argument("--rules", required=True)
    p.add_argument("--receipts", required=True)
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("plan")
    p.add_argument("--requests", required=True)
    p.add_argument("--workers", required=True)
    p.add_argument("--rules", required=True)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run")
    p.add_argument("--requests", required=True)
    p.add_argument("--workers", required=True)
    p.add_argument("--rules", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--receipt-format", default="r3",
                   choices=("r1", "r2", "r3"))
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("incremental")
    p.add_argument("--requests", required=True)
    p.add_argument("--workers", required=True)
    p.add_argument("--new-workers", required=True)
    p.add_argument("--rules", required=True)
    p.add_argument("--ledger", required=True)
    p.set_defaults(func=cmd_incremental)

    p = sub.add_parser("verify")
    p.add_argument("--requests", required=True)
    p.add_argument("--workers", required=True)
    p.add_argument("--rules", required=True)
    p.add_argument("--ledger-a", required=True)
    p.add_argument("--ledger-b", required=True)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("scheduler")
    p.add_argument("--requests", required=True)
    p.add_argument("--workers", required=True)
    p.add_argument("--rules", required=True)
    p.add_argument("--timeout-demo", action="store_true")
    p.add_argument("--stall-ms", type=int, default=3000)
    p.set_defaults(func=cmd_scheduler)

    p = sub.add_parser("snapshot")
    p.add_argument("--requests", required=True)
    p.add_argument("--workers", required=True)
    p.add_argument("--rules", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--tamper", action="store_true")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("crash")
    p.add_argument("--requests", required=True)
    p.add_argument("--workers", required=True)
    p.add_argument("--rules", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--after", type=int, default=2)
    p.set_defaults(func=cmd_crash)

    p = sub.add_parser("recover")
    p.add_argument("--requests", required=True)
    p.add_argument("--workers", required=True)
    p.add_argument("--rules", required=True)
    p.add_argument("--ledger", required=True)
    p.set_defaults(func=cmd_recover)

    p = sub.add_parser("replay")
    p.add_argument("--requests", required=True)
    p.add_argument("--workers", required=True)
    p.add_argument("--rules", required=True)
    p.add_argument("--ledger", required=True)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("gui")
    p.add_argument("--requests", required=True)
    p.add_argument("--workers", required=True)
    p.add_argument("--rules", required=True)
    p.add_argument("--ledger", required=True)
    p.set_defaults(func=cmd_gui)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        if hasattr(exc, "to_dict"):
            print(dumps(exc.to_dict()), file=sys.stderr)
        else:
            print(dumps({"code": "E_GENERIC", "reason": str(exc)}),
                  file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

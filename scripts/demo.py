"""Repeatable end-to-end demonstration of the batch attribution component.

Runs fully offline against the deterministic samples in ``samples/`` and prints
business codes plus ledger comparison conclusions at every stage. All loopback
ports come from the worker declaration and are pre-checked for bindability.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from batch_diagnostic.cli import build_parser  # noqa: E402
from batch_diagnostic.engine import DiagnosticEngine  # noqa: E402
from batch_diagnostic.gui import build_normalization_rows, build_overview  # noqa: E402
from batch_diagnostic.invalidation import propagate  # noqa: E402
from batch_diagnostic.ledger import SnapshotLedger  # noqa: E402
from batch_diagnostic.planning import build_plan  # noqa: E402
from batch_diagnostic.replay import replay_ledger  # noqa: E402
from batch_diagnostic.spec import _validate_workers, load_json_file, load_spec  # noqa: E402
from batch_diagnostic.verifier import verify_ledger  # noqa: E402
from dataclasses import replace  # noqa: E402

SAMPLES = os.path.join(ROOT, "samples")


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    requests_path = os.path.join(SAMPLES, "requests.json")
    workers_path = os.path.join(SAMPLES, "workers.json")
    new_workers_path = os.path.join(SAMPLES, "workers_v2.json")
    budget_path = os.path.join(SAMPLES, "budget.json")

    workdir = tempfile.mkdtemp(prefix="batchdiag-demo-")
    ledger_path = os.path.join(workdir, "diagnostic.sqlite")
    print("workdir:", workdir)

    # 1. Strong validation + overview --------------------------------------
    section("1. Load and strongly validate local JSON declarations [E_SPEC guard]")
    spec = load_spec(requests_path, workers_path, budget_path)
    print(json.dumps(dict(build_overview(spec)), indent=2, ensure_ascii=False))

    # 2. Three-format normalization -----------------------------------------
    section("2. Normalize r1/r2/r3 historical receipts into one view [E_FIELD guard]")
    known = {request.request_id for request in spec.requests}
    for name in ("receipts_r1.json", "receipts_r2.json", "receipts_r3.json"):
        rows = build_normalization_rows(load_json_file(os.path.join(SAMPLES, name)), known)
        for row in rows:
            print(
                row["source_version"],
                row["request_id"],
                "filled=",
                [item["field"] for item in row["filled_fields"]],
            )

    # 3. Deterministic plan + remainder rule --------------------------------
    section("3. Deterministic batch plan and explicit remainder rule")
    plan = build_plan(spec)
    print("remainder rule:", plan.remainder_rule)
    for batch in plan.batches:
        print(batch.batch_id, batch.worker_id, "remainder=" + str(batch.remainder), batch.request_ids)
    plan_again = build_plan(spec)
    identical = [b.to_dict() for b in plan.batches] == [b.to_dict() for b in plan_again.batches]
    print("byte-identical rebuild:", identical)

    # 4. Full run over real loopback sockets --------------------------------
    section("4. Forward inference via real 127.0.0.1 socket round trips")
    ledger = SnapshotLedger.create(ledger_path, spec)
    engine = DiagnosticEngine(spec, ledger)
    engine.start_workers()
    try:
        summary = engine.run()
    finally:
        engine.stop_workers()
    print(
        json.dumps(
            {
                "code": "OK",
                "processed_batches": list(summary.processed_batches),
                "peak_global": summary.peak_global,
                "peak_worker": summary.peak_worker,
                "total_budget_ms": summary.total_budget_ms,
            },
            indent=2,
        )
    )
    ledger.close()

    # 5. Invalidation propagation + subset recompute -------------------------
    section("5. Parameter change -> invalidation chain -> subset recompute")
    old_spec = spec
    new_workers = _validate_workers(load_json_file(new_workers_path), old_spec.input_dim())
    new_spec = replace(old_spec, workers=new_workers)
    report = propagate(old_spec, new_spec, build_plan(new_spec))
    invalid_set = set(report.invalid_batch_ids)
    print("invalid workers:", report.invalid_workers)
    print("invalid batches:", report.invalid_batch_ids)
    print("invalid requests:", report.invalid_request_ids)
    print("propagation levels:", sorted({step.level for step in report.steps}))

    ledger = SnapshotLedger.open(ledger_path, old_spec)
    before = {bid: ledger.load_batch(bid) for bid in ledger.committed_batch_ids()}
    engine = DiagnosticEngine(new_spec, ledger)
    engine.start_workers()
    try:
        engine.run(only_batch_ids=invalid_set, invalidation=report)
    finally:
        engine.stop_workers()
    ledger.update_header(new_spec)
    preserved = {
        bid: before[bid] == ledger.load_batch(bid)
        for bid in before
        if bid not in invalid_set
    }
    print("preserved byte-identical batches:", preserved)
    ledger.close()

    # 6. Recovery replay + independent verification -------------------------
    section("6. Recovery replay and independent ledger verification")
    ledger = SnapshotLedger.open(ledger_path, new_spec)
    replay_report = replay_ledger(new_spec, ledger)
    print("replay:", json.dumps(replay_report))
    verify_report = verify_ledger(requests_path, new_workers_path, budget_path, ledger)
    print("verify:", json.dumps(verify_report))
    ledger.close()

    # 7. Fault injection business codes -------------------------------------
    section("7. Fault injection -> E_SPEC / E_CHAIN / E_SNAPSHOT")
    cli = build_parser()
    injection_cases = [
        ("request_corrupt", "E_SPEC"),
        ("dangling_ref", "E_CHAIN"),
        ("snapshot_digest", "E_SNAPSHOT"),
    ]
    conclusions = []
    for index, (kind, expected) in enumerate(injection_cases):
        case_ledger = os.path.join(workdir, f"inject-{index}.sqlite")
        shutil.copy2(ledger_path, case_ledger)
        argv = [
            "inject",
            "--kind",
            kind,
            "--requests",
            requests_path,
            "--workers",
            new_workers_path,
            "--budget",
            budget_path,
            "--ledger",
            case_ledger,
        ]
        try:
            cli.parse_args(argv)
            from batch_diagnostic.cli import main as cli_main

            code = cli_main(argv)
        except SystemExit as exc:  # pragma: no cover - defensive
            code = int(exc.code or 0)
        conclusions.append((kind, expected, code))
        print(f"{kind}: exit={code} (1 means diagnosed; business code {expected} written to events.jsonl)")

    section("Demo conclusion")
    print("replay status:", replay_report["status"])
    print("verify status:", verify_report["status"])
    print("preserved unimpacted batches byte-identical:", all(preserved.values()))
    print("injection results:", conclusions)
    print("ledger path:", ledger_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

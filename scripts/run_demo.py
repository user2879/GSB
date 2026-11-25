"""Repeatable end-to-end acceptance demonstration using only local samples."""

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from inference_attribution.errors import DiagnosticError
from inference_attribution.invalidation import (
    assert_replay_consistent,
    diagnose_invalidation,
)
from inference_attribution.jsonio import atomic_write_json, canonical_json, sha256_canonical
from inference_attribution.ledger import Ledger
from inference_attribution.planner import build_plan
from inference_attribution.receipts import normalize_receipts, read_receipt_file
from inference_attribution.snapshot import (
    create_snapshot,
    load_snapshot,
    restore_snapshot,
)
from inference_attribution.spec import (
    load_merge_rules,
    load_requests,
    load_workers,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
OUT = ROOT / "artifacts" / "demo"


def remove_if_present(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def cli_env() -> dict[str, str]:
    env = os.environ.copy()
    source = str(ROOT / "src")
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source if not previous else source + os.pathsep + previous
    return env


def run_cli(*args: str, timeout: float = 40.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "inference_attribution.cli", *args],
        cwd=ROOT,
        env=cli_env(),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def error_code(action) -> dict[str, Any]:
    try:
        action()
    except DiagnosticError as exc:
        return exc.to_dict()
    raise AssertionError("expected a DiagnosticError")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    full_db = OUT / "full.sqlite"
    crash_db = OUT / "crash.sqlite"
    restored_db = OUT / "restored.sqlite"
    for path in (full_db, crash_db, restored_db):
        remove_if_present(path)

    workers = load_workers(SAMPLES / "workers.json")
    requests = load_requests(SAMPLES / "requests.json", workers)
    rules = load_merge_rules(SAMPLES / "merge_rules.json")
    plan = build_plan(workers, requests)
    atomic_write_json(OUT / "plan.json", plan.to_dict())

    raw = {
        "r1": read_receipt_file(SAMPLES / "receipts_r1.jsonl", "r1"),
        "r2": read_receipt_file(SAMPLES / "receipts_r2.jsonl", "r2"),
        "r3": read_receipt_file(SAMPLES / "receipts_r3.jsonl", "r3"),
    }
    normalized = normalize_receipts(
        raw,
        request_set=requests,
        worker_set=workers,
        plan=plan,
        rules=rules,
    )
    atomic_write_json(OUT / "normalized_ledger.json", normalized.to_dict())

    updated_workers_path = SAMPLES / "workers_param_update.json"
    repartitioned_workers_path = SAMPLES / "workers_repartition.json"
    updated_workers = load_workers(updated_workers_path)
    repartitioned_workers = load_workers(repartitioned_workers_path)
    updated_plan = build_plan(updated_workers, requests)
    repartitioned_plan = build_plan(repartitioned_workers, requests)
    version_report = diagnose_invalidation(
        workers, plan, updated_workers, updated_plan
    )
    repartition_report = diagnose_invalidation(
        workers, plan, repartitioned_workers, repartitioned_plan
    )
    atomic_write_json(OUT / "invalidation_param_version.json", version_report.to_dict())
    atomic_write_json(
        OUT / "invalidation_repartition.json", repartition_report.to_dict()
    )

    common = (
        "--requests",
        str(SAMPLES / "requests.json"),
        "--workers",
        str(SAMPLES / "workers.json"),
        "--rules",
        str(SAMPLES / "merge_rules.json"),
    )
    full = run_cli("run", *common, "--ledger", str(full_db))
    if full.returncode != 0:
        raise RuntimeError(f"full run failed: {full.stderr}")
    full_summary = json.loads(full.stdout)

    crash = run_cli(
        "run", *common, "--ledger", str(crash_db), "--crash-after", "1"
    )
    if crash.returncode not in {137, -signal.SIGKILL}:
        raise RuntimeError(
            f"crash run returned {crash.returncode}, expected 137: {crash.stderr}"
        )
    with Ledger(crash_db) as ledger:
        partial = ledger.recovery_report().to_dict()
        crash_digest_at_kill = ledger.content_digest
    atomic_write_json(OUT / "crash_recovery_before_resume.json", partial)

    resumed = run_cli("run", *common, "--ledger", str(crash_db))
    if resumed.returncode != 0:
        raise RuntimeError(f"resume run failed: {resumed.stderr}")
    resume_summary = json.loads(resumed.stdout)
    with Ledger(full_db) as ledger:
        full_rows = ledger.rows()
    with Ledger(crash_db) as ledger:
        recovered_rows = ledger.rows()
        final_recovery = ledger.recovery_report().to_dict()
        final_recovered_digest = ledger.content_digest
    assert_replay_consistent(plan, full_rows, recovered_rows)

    runtime_receipt_digest = {
        row["batch_id"]: row["receipt_digest"] for row in full_rows
    }
    normalized_match = {
        batch["batch_id"]: batch["attribution_digest"]
        == runtime_receipt_digest[batch["batch_id"]]
        for batch in normalized.batches
    }
    if not all(normalized_match.values()):
        raise AssertionError("normalized historical ledger does not match runtime receipts")

    snapshot_path = OUT / "snapshot.json"
    with Ledger(full_db) as ledger:
        create_snapshot(workers, ledger, snapshot_path)
    restore_outcome = restore_snapshot(
        snapshot_path, restored_db, updated_workers
    )
    with Ledger(restored_db) as ledger:
        restored_rows = ledger.rows()
    if canonical_json(restored_rows) != canonical_json(full_rows):
        raise AssertionError("whole snapshot replacement did not preserve ledger rows")

    snapshot_document = json.loads(snapshot_path.read_text(encoding="utf-8"))
    tampered_snapshot = copy.deepcopy(snapshot_document)
    tampered_snapshot["ledger_digest"] = "0" * 64
    snapshot_body = {
        key: value
        for key, value in tampered_snapshot.items()
        if key != "content_digest"
    }
    tampered_snapshot["content_digest"] = sha256_canonical(snapshot_body)
    tampered_snapshot_path = OUT / "snapshot_tampered_ledger_digest.json"
    atomic_write_json(tampered_snapshot_path, tampered_snapshot)

    corrupt_workers_path = OUT / "corrupt_workers.json"
    corrupt_workers = json.loads(
        (SAMPLES / "workers.json").read_text(encoding="utf-8")
    )
    corrupt_workers["workers"][0]["host"] = "0.0.0.0"
    atomic_write_json(corrupt_workers_path, corrupt_workers)

    bad_expected = copy.deepcopy(full_rows)
    bad_expected[0]["receipt_digest"] = "0" * 64
    injections = {
        "corrupt_declaration": error_code(lambda: load_workers(corrupt_workers_path)),
        "snapshot_ledger_digest_mismatch": error_code(
            lambda: load_snapshot(tampered_snapshot_path)
        ),
        "replay_mismatch": error_code(
            lambda: assert_replay_consistent(plan, bad_expected, recovered_rows)
        ),
    }

    summary = {
        "sample_inputs": {
            "requests": str(SAMPLES / "requests.json"),
            "workers": str(SAMPLES / "workers.json"),
            "merge_rules": str(SAMPLES / "merge_rules.json"),
            "receipt_formats": ["r1", "r2", "r3"],
        },
        "plan": {
            "content_digest": plan.content_digest,
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "worker_id": batch.worker_id,
                    "batch_seq": batch.batch_seq,
                    "request_ids": list(batch.request_ids),
                    "param_version_seq": batch.param_version_seq,
                    "param_digest": batch.param_digest,
                }
                for batch in plan.batches
            ],
        },
        "normalization": {
            "content_digest": normalized.content_digest,
            "batches": len(normalized.batches),
            "runtime_attribution_match": normalized_match,
            "filled_fields": sorted(
                {
                    field
                    for batch in normalized.batches
                    for receipt in batch["receipts"]
                    for field in receipt["filled_fields"]
                }
            ),
        },
        "invalidation": {
            "param_version": version_report.to_dict(),
            "repartition": repartition_report.to_dict(),
        },
        "full_run": full_summary,
        "forced_termination": {
            "process_return_code": crash.returncode,
            "exit_report_available": False,
            "partial_ledger_digest_at_kill": crash_digest_at_kill,
            "recovery_before_resume": partial,
            "resume": resume_summary,
            "final_recovery": final_recovery,
            "final_ledger_digest": final_recovered_digest,
            "replay_consistent_with_full_run": True,
        },
        "snapshot": {
            "whole_replacement": restore_outcome.whole_replacement,
            "replacement_reasons": list(restore_outcome.replacement_reasons),
            "restored_rows_equal_full": True,
        },
        "injected_errors": injections,
    }
    atomic_write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

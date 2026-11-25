"""End-to-end local acceptance demonstration.

Verifies with deterministic samples:
  1. full rebuild from an empty state;
  2. incremental replay is field-for-field identical to the full rebuild;
  3. thread-pool concurrency (1 vs configured max) changes nothing;
  4. shuffled batch arrival order changes nothing;
  5. snapshot recovery detects the manifest and keeps the same global version;
  6. idempotent redelivery is refused and never re-billed.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffbudget.engine import full_replay, run_pipeline  # noqa: E402
from ffbudget.fingerprint import canonical_json  # noqa: E402

CONFIG = ROOT / "samples" / "config.json"
BATCHES = ROOT / "samples" / "batches"
REQUESTS = ROOT / "samples" / "requests.json"
WORK = ROOT / "state" / "demo"


def reset(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def comparable(canonical: dict) -> dict:
    # Recovery metadata and cache hit/miss counters describe execution, not
    # the canonical business result, so they are excluded from equivalence.
    clone = json.loads(json.dumps(canonical))
    clone["report"].pop("recovery", None)
    clone["report"].pop("cache", None)
    clone["report"]["budget_ledger"].pop("max_concurrency_used", None)
    clone["manifest"].pop("global_version", None)
    clone["manifest"].pop("input_fingerprint", None)
    return clone


def main() -> int:
    results = []

    reset(WORK / "full")
    full, _, _ = full_replay(CONFIG, BATCHES, REQUESTS, WORK / "full")
    results.append(("full_rebuild", True, "empty-state pipeline completed"))

    # Incremental run reusing an existing snapshot and cache book.
    inc, _, _ = run_pipeline(CONFIG, BATCHES, REQUESTS, WORK / "full",
                             mode="auto")
    same = canonical_json(comparable(inc)) == canonical_json(comparable(full))
    results.append(("incremental_equals_full", same,
                    "incremental replay field-for-field identical"))

    # Concurrency 1 vs max configured concurrency.
    reset(WORK / "serial")
    serial, _, _ = full_replay(CONFIG, BATCHES, REQUESTS, WORK / "serial",
                               concurrency_override=1)
    reset(WORK / "parallel")
    parallel, _, _ = full_replay(CONFIG, BATCHES, REQUESTS,
                                 WORK / "parallel", concurrency_override=8)
    same = canonical_json(comparable(serial)) == \
        canonical_json(comparable(parallel))
    results.append(("concurrency_equivalence", same,
                    "max_workers=1 and 8 produce identical artifacts"))

    # Shuffled arrival order: rename batch files so load order changes.
    shuffled_dir = WORK / "shuffled_batches"
    reset(shuffled_dir)
    batch_files = sorted(BATCHES.glob("*.ndjson"))
    permuted = [batch_files[4], batch_files[1], batch_files[3],
                batch_files[0], batch_files[2]]
    for index, source in enumerate(permuted):
        shutil.copyfile(source, shuffled_dir / f"{index:04d}_{source.name}")
    reset(WORK / "shuffled")
    shuffled, _, _ = full_replay(CONFIG, shuffled_dir, REQUESTS,
                                 WORK / "shuffled")
    same = canonical_json(comparable(shuffled)) == \
        canonical_json(comparable(full))
    results.append(("arrival_order_independence", same,
                    "shuffled batch file order produces identical artifacts"))

    # Snapshot recovery decision and preserved global version.
    recovery, _, _ = run_pipeline(CONFIG, BATCHES, REQUESTS, WORK / "full",
                                  mode="auto")
    mode = recovery["report"]["recovery"]["mode"]
    version_kept = recovery["manifest"]["global_version"] == \
        full["manifest"]["global_version"]
    results.append(("snapshot_recovery", mode == "incremental" and version_kept,
                    f"recovery mode={mode}"))

    # Idempotent redelivery: duplicates recorded with zero billed points.
    duplicates = full["report"]["duplicate_deliveries"]
    idempotent = bool(duplicates) and all(not item["counted"]
                                          for item in duplicates)
    results.append(("idempotent_redelivery", idempotent,
                    f"{len(duplicates)} duplicate receipt(s), zero points"))

    report_path = WORK / "acceptance_report.json"
    report_path.write_text(canonical_json({
        "checks": [{"name": n, "passed": ok, "detail": d}
                   for n, ok, d in results],
        "all_passed": all(ok for _, ok, _ in results),
        "diagnostic_summary": {
            "states": full["report"]["global_state_distribution"],
            "refusals": full["report"]["refusal_reason_counts"],
            "budget": {k: v for k, v in
                       full["report"]["budget_ledger"].items()
                       if k != "events"},
            "pending": [p["request_id"]
                        for p in full["report"]["requests_pending"]],
        }}), encoding="utf-8")

    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"report -> {report_path}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

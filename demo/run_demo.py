"""End-to-end demonstration covering requirements 1-8.

Run from the repository root with the project virtualenv:

    .venv/bin/python demo/run_demo.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckpt_diag.compute import compute_batch  # noqa: E402
from ckpt_diag.digest import canonical  # noqa: E402
from ckpt_diag.invalidation import (  # noqa: E402
    plan_invalidation, recompute_subset, verify_against_full_rebuild,
    verify_untouched_byte_stable)
from ckpt_diag.ledger import Ledger  # noqa: E402
from ckpt_diag.orchestrator import Orchestrator  # noqa: E402
from ckpt_diag.receipts import (  # noqa: E402
    decode_receipt, load_receipt_file, normalize_mixed_batch)
from ckpt_diag.samplegen import (  # noqa: E402
    build_spec_dict, write_bad_receipts, write_historical_receipts, write_spec)
from ckpt_diag.spec import load_spec  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    work = ROOT / "artifacts" / "demo"
    work.mkdir(parents=True, exist_ok=True)
    spec_path = write_spec(work / "spec.json")
    spec = load_spec(spec_path)
    section("R1 spec loaded and strongly validated")
    print(f"shards={len(spec.shards)} checkpoints={len(spec.checkpoints)} "
          f"events={len(spec.events)} budget={spec.global_budget}")
    for shard in spec.shards:
        print(f"  {shard.shard_id} {shard.host}:{shard.port} "
              f"lease_cap={shard.lease_cap} quota={shard.concurrency_quota} "
              f"layers={[layer.layer_id for layer in shard.layers]}")

    ledger = Ledger(work / "ledger.db")
    orch = Orchestrator(spec, spec_path, ledger, ttl_ms=5000)
    section("R3/R4/R6 shard subprocesses, leases, fixed-point compute, scheduler")
    orch.start_shards()
    try:
        event_log = orch.run_all()
    finally:
        orch.stop_shards()
    for summary in event_log:
        if summary["kind"] == "dispatch":
            result = summary["result"]
            print(f"  seq={summary['event_seq']} {result['batch_id']} "
                  f"pv={result['param_version']} loss={result['total_loss']} "
                  f"param_summary={result['param_summary'][:12]} "
                  f"loss_summary={result['loss_summary'][:12]} "
                  f"elapsed_ms={result['elapsed_ms']} samples={result['sample_ids']}")
        else:
            print(f"  seq={summary['event_seq']} {summary['kind']} "
                  f"{ {k: v for k, v in summary.items() if k not in ('event_seq','kind')} }")

    section("R6 scheduler wait report (monotonic ms) and peak in-flight invariants")
    for row in orch.scheduler.wait_report():
        print(f"  {row}")
    print(f"  final free capacity = {orch.arbiter.state.free_capacity()} / "
          f"{orch.arbiter.state.budget}")

    section("R5 unique ownership decisions")
    for decision in orch.arbiter.state.decisions:
        print(f"  seq={decision.event_seq:>2} {decision.kind:<15} "
              f"units={list(decision.units)} {decision.before_owner}->"
              f"{decision.after_owner} basis={decision.basis}")

    section("R2 c1/c2/c3 receipt normalization and mixed-format arbitration")
    receipt_path = write_historical_receipts(ROOT / "receipts", orch.results)
    receipts = load_receipt_file(receipt_path)
    grouped: dict[str, list] = {}
    for receipt in receipts:
        grouped.setdefault(receipt.batch_id, []).append(receipt)
    normalized_ledger: dict[str, dict] = {}
    for batch_id in sorted(grouped):
        accepted, rejected = normalize_mixed_batch(grouped[batch_id])
        normalized_ledger[batch_id] = accepted.to_dict()
        print(f"  batch={batch_id} accepted=c{accepted.source_format} "
              f"pv={accepted.param_version} backfilled={accepted.backfilled_fields} "
              f"basis={accepted.backfill_basis}")
        for reject in rejected:
            print(f"      rejected c{reject.source_format}: {reject.rejected_reason}")
    bad_path = write_bad_receipts(ROOT / "receipts" / "bad")
    print("  malformed receipt rejection:")
    for line_no, line in enumerate(bad_path.read_text().splitlines(), start=1):
        try:
            decode_receipt(line, f"{bad_path}:{line_no}")
            print(f"    line {line_no}: UNEXPECTED ACCEPT")
        except Exception as exc:  # noqa: BLE001 - demonstration wants the code
            print(f"    line {line_no}: {exc}")

    section("R3 determinism: order/concurrency independent bit-identical summaries")
    shard = spec.shard("shard-alpha")
    base = compute_batch(shard, "det-batch", 1, ("s-0", "s-1", "s-2"),
                         ("L0", "L1"))
    permuted = compute_batch(shard, "det-batch", 1, ("s-2", "s-0", "s-1"),
                             ("L1", "L0"))
    print(f"  equal loss_summary={base.loss_summary == permuted.loss_summary} "
          f"equal param_summary={base.param_summary == permuted.param_summary}")

    section("R7 invalidation propagation and subset recompute")
    new_dict = json.loads(spec_path.read_text())
    for layer in next(s for s in new_dict["shards"]
                      if s["shard_id"] == "shard-alpha")["layers"]:
        if layer["layer_id"] == "L1":
            layer["scale"] = layer["scale"] + 1
    new_spec_path = work / "spec_v2.json"
    new_spec_path.write_text(json.dumps(new_dict, indent=2))
    new_spec = load_spec(new_spec_path)
    batch_scopes = {
        shard_id: {batch_id: tuple(entry["scope_layers"])
                   for batch_id, entry in batches.items()}
        for shard_id, batches in orch.batch_inputs.items()}
    batch_pvs = {
        shard_id: {batch_id: entry["param_version"]
                   for batch_id, entry in batches.items()}
        for shard_id, batches in orch.batch_inputs.items()}
    report = plan_invalidation(spec, new_spec, batch_scopes, batch_pvs)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    before_results = {batch_id: compute_batch(
        spec.shard(shard_id), batch_id, entry["param_version"],
        tuple(entry["sample_ids"]), tuple(entry["scope_layers"]))
        for shard_id, batches in orch.batch_inputs.items()
        for batch_id, entry in batches.items()}
    subset = recompute_subset(new_spec, report, orch.batch_inputs)
    verify_against_full_rebuild(new_spec, subset, orch.batch_inputs)
    invalid_ids = {batch_id for ids in report.invalid_batches.values()
                   for batch_id in ids}
    after_map = dict(before_results)
    for shard_id, batches in subset.items():
        after_map.update(batches)
    verify_untouched_byte_stable(before_results, after_map, invalid_ids)
    print("  subset recompute matches blank full rebuild; untouched batches stable")

    section("R8 SQLite atomic ledger, snapshot replace, crash replay")
    progress = ledger.progress()
    print(f"  committed cursor seq={progress['last_seq']} "
          f"digest={progress['cursor_digest'][:16]} batches="
          f"{len(ledger.committed_batches())}")

    def recompute(row) -> dict:
        sh = spec.shard(row["shard_id"])
        result = compute_batch(sh, row["batch_id"], row["param_version"],
                               tuple(json.loads(row["sample_ids"])),
                               tuple(json.loads(row["scope_layers"])))
        return {"param_summary": result.param_summary,
                "loss_summary": result.loss_summary,
                "total_loss": result.total_loss,
                "param_version": result.param_version,
                "sample_ids": result.sample_ids}

    outcome = ledger.verify_replay(recompute)
    print(f"  replay verified batches={outcome.verified_batches} "
          f"rerun_required={outcome.rerun_required}")
    digest_value = ledger.snapshot_digest()
    reopened = Ledger(work / "ledger.db").open_snapshot(
        expected_version=1, expected_digest=digest_value)
    print(f"  matching snapshot kept; digest={reopened.snapshot_digest()[:16]}")
    replaced = Ledger(work / "ledger.db").open_snapshot(
        expected_version=999, expected_digest=digest_value)
    reason = replaced.conn.execute(
        "SELECT value FROM meta WHERE key='last_replace_reason'").fetchone()[0]
    print(f"  mismatched snapshot replaced whole; reason={reason}")
    replaced.close()

    # Simulate a forced termination: a pending row with no committed batch.
    crash_ledger = Ledger(work / "crash.db")
    crash_ledger.mark_pending("batch-crash-1", seq=99,
                              note="process killed before commit")
    crash_outcome = crash_ledger.recover_after_crash()
    print(f"  crash recovery rerun_required={crash_outcome.rerun_required} "
          f"committed={crash_outcome.verified_batches}")
    crash_ledger.close()
    ledger.close()

    report_path = work / "normalized_receipts.json"
    report_path.write_text(json.dumps(normalized_ledger, indent=2,
                                      ensure_ascii=False, sort_keys=True))
    print(f"\nartifacts written under {work} and {ROOT / 'receipts'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

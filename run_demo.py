"""Runnable acceptance demonstration on local deterministic samples.

Exercises the full chain end to end:
  1. strict spec loading and deterministic planning
  2. real 127.0.0.1 worker subprocesses and r1/r2/r3 mixed receipts
  3. lease-gated parallel scheduling with mixed parameter versions
  4. invalidation propagation after a parameter version bump
  5. SQLite commit, forced-termination recovery and E_REPLAY replay check
"""

import os
import tempfile

from leasebatch.engine import RunEngine
from leasebatch.invalidation import baseline_versions, propagate
from leasebatch.ledger import Ledger
from leasebatch.plan import build_plan
from leasebatch.spec import load_spec_dir


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    spec = load_spec_dir(os.path.join(root, "samples"))
    print("== 1. spec loaded ==")
    print("workers:", [w.worker_id for w in spec.workers])

    plan = build_plan(spec, batch_size=2, seed=42)
    print("\n== 2. deterministic plan ==")
    for batch in plan.batches:
        print(f"  {batch.batch_id}  {batch.worker_id}  "
              f"{list(batch.request_ids)}")
    print("plan signature:", plan.signature[:16])

    batch_ids = [b.batch_id for b in plan.batches]
    format_map = {
        batch_ids[0]: "r1",
        batch_ids[1]: "r2",
        batch_ids[2]: "r3",
    }
    version_map = {batch_ids[1]: 1, batch_ids[2]: 1}

    with tempfile.TemporaryDirectory() as directory:
        ledger_path = os.path.join(directory, "demo.db")
        print("\n== 3. lease-gated run over real loopback subprocesses ==")
        engine = RunEngine(spec, ledger_path)
        out = engine.execute(
            2, 42, version_map=version_map, format_map=format_map
        )
        for result in out["run"]["results"]:
            normalized = result["normalized"]
            print(
                f"  {result['batch'].batch_id} "
                f"src={normalized['source_format']} "
                f"filled={normalized['filled_fields']} "
                f"lease={result['lease_id']}"
            )

        print("\n== 4. invalidation after bumping w1 parameter version ==")
        old_versions = baseline_versions(spec)
        new_versions = dict(old_versions, w1=1)
        report = propagate(
            spec, plan, plan, old_versions, new_versions
        )
        print("  invalid batches:", report["invalid_batch_ids"])
        print("  invalid requests:", report["invalid_request_ids"])
        print("  untouched batches:", report["untouched_batch_ids"])

        print("\n== 5. ledger replay verification ==")
        ledger = Ledger(ledger_path)
        print("  replay identical:",
              ledger.replay_verify(out["expected"]))
        ledger.close()

        crash_path = os.path.join(directory, "crash.db")
        crash_engine = RunEngine(spec, crash_path)
        crashed = crash_engine.execute(
            2, 42, in_flight_batch_ids={batch_ids[0]}
        )
        print("\n== 6. forced termination recovery ==")
        print("  unknown batches:",
              [row["batch_id"] for row in crashed["recovery"]["unknown"]])
        print("  committed batches recovered:",
              len(crashed["recovery"]["committed"]))
        crash_ledger = Ledger(crash_path)
        print("  replay of committed subset identical:",
              crash_ledger.replay_verify(crashed["expected"]))
        crash_ledger.close()


if __name__ == "__main__":
    main()

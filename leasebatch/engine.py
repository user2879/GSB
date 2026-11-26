"""End-to-end orchestration glue for the validation run."""

import tempfile

from .digest import digest
from .invalidation import baseline_versions
from .lease import LeaseManager, MonotonicClock
from .ledger import Ledger
from .plan import build_plan
from .scheduler import Scheduler
from .worker import WorkerPool


def adoption_conclusion(normalized):
    """Deterministic accept/reject conclusion bound to the normalized digest."""
    fields = normalized["fields"]
    return {
        "decision": "accepted",
        "batch_id": fields["batch_id"],
        "source_format": normalized["source_format"],
        "filled_fields": normalized["filled_fields"],
        "normalized_digest": normalized["normalized_digest"],
        "conclusion_digest": digest(
            {
                "kind": "adoption",
                "batch_id": fields["batch_id"],
                "normalized_digest": normalized["normalized_digest"],
                "output_digest": fields["output_digest"],
                "param_digest": fields["param_digest"],
            }
        ),
    }


class RunEngine:
    def __init__(self, spec, ledger_path, *, clock=None):
        self.spec = spec
        self.ledger_path = ledger_path
        self.clock = clock or MonotonicClock()

    def execute(self, batch_size, seed, *, version_map=None,
                format_map=None, in_flight_batch_ids=None,
                use_real_pool=True, pool=None):
        """Plan, dispatch through real worker subprocesses and commit.

        ``in_flight_batch_ids`` reserves rows without committing them to model
        a forced termination for recovery/replay testing.
        """
        plan = build_plan(self.spec, batch_size, seed)
        ledger = Ledger(self.ledger_path)
        snapshot_id = ledger.replace_snapshot(plan, self.clock.now_ms())
        lease_manager = LeaseManager(self.spec, self.clock)
        in_flight = set(in_flight_batch_ids or ())

        for global_cursor, batch in enumerate(plan.batches, start=1):
            if batch.batch_id in in_flight:
                ledger.mark_in_flight(snapshot_id, batch, global_cursor)

        own_pool = False
        if pool is None and use_real_pool:
            tmp_dir = tempfile.mkdtemp(prefix="leasebatch-")
            pool = WorkerPool(self.spec, tmp_dir)
            pool.start()
            own_pool = True
        try:
            scheduler = Scheduler(
                self.spec, plan, pool,
                version_map=version_map or {},
                format_map=format_map or {},
                clock=self.clock,
            )
            run = scheduler.run(lease_manager)
        finally:
            if own_pool:
                pool.shutdown()

        expected = []
        cursor = 0
        for result in run["results"]:
            batch = result["batch"]
            if batch.batch_id in in_flight:
                # Forced termination: the reserved uncommitted row remains
                # "unknown" and is never promoted to a committed fact.
                continue
            cursor += 1
            receipt = result["normalized"]
            adoption = adoption_conclusion(receipt)
            lease_snapshot = lease_manager.get(
                result["lease_id"]
            ).snapshot()
            ledger.commit_batch(
                snapshot_id, batch, cursor,
                receipt=receipt,
                adoption=adoption,
                lease=lease_snapshot,
            )
            expected.append(
                {
                    "batch_id": batch.batch_id,
                    "receipt": receipt,
                    "adoption": adoption,
                    "lease": lease_snapshot,
                    "progress_cursor": cursor,
                }
            )
        recovery = ledger.recover()
        ledger.close()
        return {
            "plan": plan,
            "run": run,
            "recovery": recovery,
            "expected": expected,
            "versions": version_map or {},
        }


def initial_versions(spec):
    return baseline_versions(spec)

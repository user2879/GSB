import os
import shutil
import tempfile
import unittest
from pathlib import Path

from rlcb.engine import Engine
from rlcb.hashing import canonical_json
from rlcb.invalidate import apply_changes
from rlcb.ledger import Ledger
from rlcb.planner import choose_plan
from rlcb.spec import load_bundle

SERVICES = Path("samples/acceptance/services.json")
REQUESTS = Path("samples/acceptance/requests.json")


def plan_for(bundle, force="full", changed=None, batches=None):
    batches = batches or sorted({f"{r.service_id}:{r.batch_id}"
                                 for r in bundle.requests})
    windows = 0
    for spec in bundle.services:
        reqs = [r for r in bundle.requests if r.service_id == spec.service_id]
        if reqs:
            windows += max(r.arrival_seq for r in reqs) // spec.window_length + 1
    return choose_plan(
        service_count=len(bundle.services), batch_count=len(batches),
        window_count=windows, changed_services=changed or [],
        changed_batches=[b for b in batches
                         if changed and b.split(":", 1)[0] in set(changed)],
        all_batches=batches, force_mode=force)


class EngineIntegrationTest(unittest.TestCase):
    def test_real_workers_byte_stable_across_runs(self):
        bundle = load_bundle(SERVICES, REQUESTS)
        engine_a = Engine(bundle, use_workers=True)
        result_a = engine_a.run(plan_for(bundle))
        engine_b = Engine(bundle, use_workers=True)
        result_b = engine_b.run(plan_for(bundle))
        self.assertEqual(
            [canonical_json(d) for d in result_a.decisions],
            [canonical_json(d) for d in result_b.decisions])
        # Compute summaries are byte-stable even though elapsed_ms differs.
        self.assertEqual(
            {k: v["summary"] for k, v in result_a.summaries.items()},
            {k: v["summary"] for k, v in result_b.summaries.items()})

    def test_incremental_preserves_unaffected_decisions_byte_for_byte(self):
        tmp = tempfile.mkdtemp()
        try:
            ledger_path = os.path.join(tmp, "ledger.db")
            bundle = load_bundle(SERVICES, REQUESTS)
            ledger = Ledger(ledger_path)
            base_engine = Engine(bundle, use_workers=False)
            base = base_engine.run(plan_for(bundle))
            self._commit_all(ledger, bundle, base)
            prior = ledger.load_decisions()
            ledger.close()

            # Edit only svcB's breaker threshold; force incremental.
            edited_services, changes = apply_changes(
                bundle.services, {"svcB": {"failure_threshold_bp": 2500}})
            bundle.services = edited_services
            bundle.service_by_id = {s.service_id: s for s in edited_services}
            plan = plan_for(bundle, force="incremental",
                            changed=["svcB"])
            engine = Engine(bundle, use_workers=False)
            inc = engine.run(plan, change_sets=changes,
                             prior_decisions=prior,
                             prior_ownership=Ledger(ledger_path).load_lease_ownership())
            inc_by_id = {d["request_id"]: d for d in inc.decisions}
            # Unaffected services (svcA/svcC) stay byte-identical.
            for request in bundle.requests:
                if request.service_id != "svcB":
                    self.assertEqual(
                        canonical_json(prior[request.request_id]),
                        canonical_json(inc_by_id[request.request_id]))
            # At least one svcB decision changed due to the new threshold.
            changed_decisions = [
                rid for rid in inc.invalidation["affected_requests"]]
            self.assertTrue(changed_decisions)
            self.assertTrue(all(rid.startswith("svcB")
                                for rid in changed_decisions))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_full_recompute_matches_scratch_after_edits(self):
        bundle = load_bundle(SERVICES, REQUESTS)
        edited_services, _ = apply_changes(
            bundle.services, {"svcA": {"degrade_level": 2,
                                       "rps_limit": 3}})
        bundle.services = edited_services
        bundle.service_by_id = {s.service_id: s for s in edited_services}
        engine_one = Engine(bundle, use_workers=False)
        engine_two = Engine(bundle, use_workers=False)
        r1 = engine_one.run(plan_for(bundle))
        r2 = engine_two.run(plan_for(bundle))
        self.assertEqual(
            [canonical_json(d) for d in r1.decisions],
            [canonical_json(d) for d in r2.decisions])
        self.assertEqual(r1.content_digest, r2.content_digest)

    def _commit_all(self, ledger, bundle, result):
        by_batch = {}
        for decision in result.decisions:
            by_batch.setdefault(
                f"{decision['service_id']}:{decision['batch_id']}", []
            ).append(decision)
        from rlcb.ledger import LEDGER_FORMAT_VERSION
        for key in sorted(by_batch):
            service_id, batch_id = key.split(":", 1)
            spec = bundle.service_by_id[service_id]
            ledger.commit_batch(
                batch={"batch_id": batch_id, "service_id": service_id,
                       "config_version": spec.config_version},
                decisions=by_batch[key],
                lease_ownership=result.lease_ownership,
                lease_events=result.lease_events,
                invalidation=None,
                cursor={"last_batch": key,
                        "format_version": LEDGER_FORMAT_VERSION})


if __name__ == "__main__":
    unittest.main()

import unittest

from src import core, dag


def _plan(cfg, versions):
    return core.build_plan(cfg, versions)


def _cfg(worker_seeds=(101, 202), batch_size=2):
    return {
        "sample_seed": 7,
        "request_count": 8,
        "input_dim": 4,
        "hidden_dim": 6,
        "output_dim": 4,
        "batch_size": batch_size,
        "fixed_point": {"frac_bits": 8, "word_bits": 32,
                        "rounding": "floor", "saturation": "saturate"},
        "workers": [{"worker_id": "w0", "seed": worker_seeds[0]},
                    {"worker_id": "w1", "seed": worker_seeds[1]}],
        "workers_by_id": {"w0": {"worker_id": "w0", "seed": worker_seeds[0]},
                          "w1": {"worker_id": "w1", "seed": worker_seeds[1]}},
        "scheduler": {"global_limit": 2, "per_worker_limit": 1, "timeout_ms": 1000},
    }


def _old_state(plan):
    return {pb.batch_id: {"worker_id": pb.worker_id,
                          "param_version": pb.param_version,
                          "request_ids": list(pb.request_ids)} for pb in plan}


def _digests(cfg):
    return {wid: core.spec_digest(core.param_spec_dict(wid, w["seed"], cfg))
            for wid, w in cfg["workers_by_id"].items()}


class TestInvalidation(unittest.TestCase):
    def test_seed_change_invalidates_only_that_worker(self):
        cfg_old = _cfg()
        cfg_new = _cfg(worker_seeds=(101, 999))
        versions = {"w0": 1, "w1": 2}
        plan = _plan(cfg_new, versions)
        report = dag.compute_invalidation(_old_state(_plan(cfg_old, versions)),
                                          _digests(cfg_old), plan, _digests(cfg_new))
        w1_batches = sorted(pb.batch_id for pb in plan if pb.worker_id == "w1")
        self.assertEqual(list(report.invalidated_batches), w1_batches)
        self.assertEqual(len(report.unaffected_batches),
                         len(plan) - len(w1_batches))
        inv_reqs = set()
        for pb in plan:
            if pb.worker_id == "w1":
                inv_reqs.update(pb.request_ids)
        self.assertEqual(set(report.invalidated_requests), inv_reqs)
        levels = [s[1] for s in report.steps]
        self.assertEqual(levels, sorted(levels, key={"worker": 0, "batch": 1,
                                                     "request": 2}.get))

    def test_batch_size_change_invalidates_all(self):
        cfg_old = _cfg(batch_size=2)
        cfg_new = _cfg(batch_size=3)
        versions = {"w0": 1, "w1": 2}
        old_plan = _plan(cfg_old, versions)
        new_plan = _plan(cfg_new, versions)
        report = dag.compute_invalidation(_old_state(old_plan), _digests(cfg_old),
                                          new_plan, _digests(cfg_new))
        shared = set(pb.batch_id for pb in old_plan) & set(pb.batch_id for pb in new_plan)
        self.assertEqual(set(report.invalidated_batches), shared)
        self.assertIn("membership-changed", report.batch_cause["batch-0000"])

    def test_no_change_no_invalidation(self):
        cfg = _cfg()
        versions = {"w0": 1, "w1": 2}
        plan = _plan(cfg, versions)
        report = dag.compute_invalidation(_old_state(plan), _digests(cfg),
                                          plan, _digests(cfg))
        self.assertEqual(report.invalidated_batches, ())
        self.assertEqual(len(report.unaffected_batches), len(plan))

    def test_rows_for_batch_three_step_chain(self):
        cfg_old = _cfg()
        cfg_new = _cfg(worker_seeds=(101, 999))
        versions = {"w0": 1, "w1": 2}
        plan = _plan(cfg_new, versions)
        report = dag.compute_invalidation(_old_state(_plan(cfg_old, versions)),
                                          _digests(cfg_old), plan, _digests(cfg_new))
        bid = report.invalidated_batches[0]
        rows = report.rows_for_batch(bid, ["req-0001"])
        steps = [r[2] for r in rows]
        self.assertEqual(steps, [1, 2, 3])
        self.assertIn("worker-spec-changed:w1", rows[0][3])
        self.assertIn("req-0001", rows[2][3])


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from tier_mover.engine import TierEngine
from tier_mover.errors import CapacityError, IntegrityError
from tier_mover.ledger import Ledger
from tier_mover.manifest import load_manifest
from tier_mover.planner import derive_plan, split_batches
from tier_mover.workspace import write_sample_manifests


class PlannerEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = write_sample_manifests(self.root, seed=7)
        self.engine = TierEngine(self.root)
        self.engine.prepare_dirs(self.manifest)
        self.ledger = Ledger(self.root / "ledger.db")
        self.ledger.register_spec(self.manifest, 100)
        self.engine.seed_objects(self.manifest)
        self.ledger.initialize_objects(self.manifest, 101)

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def test_plan_byte_for_byte_deterministic(self):
        p1 = derive_plan(self.manifest, self.ledger.placement())
        p2 = derive_plan(self.manifest, self.ledger.placement())
        self.assertEqual(p1.to_json(), p2.to_json())
        self.assertEqual(p1.plan_digest, p2.plan_digest)

    def test_plan_steps_carry_heat_and_ranking(self):
        plan = derive_plan(self.manifest, self.ledger.placement())
        self.assertTrue(plan.steps)
        for step in plan.steps:
            self.assertGreaterEqual(step.heat, 0)
            self.assertGreaterEqual(step.ranking, 1)
            self.assertIn(step.direction, ("evict", "promote"))

    def test_plan_never_exceeds_capacity(self):
        plan = derive_plan(self.manifest, self.ledger.placement())
        for tier in self.manifest.tiers:
            self.assertLessEqual(plan.simulated_occupancy[tier.tier_id],
                                 tier.capacity_bytes)

    def test_real_move_is_atomic_and_bytes_match_ledger(self):
        plan = self.engine.plan(self.ledger, self.manifest)
        conclusion = self.engine.execute(self.ledger, self.manifest, plan=plan)
        self.assertGreater(conclusion["migrations_committed"], 0)
        self.engine.reconcile_bytes(self.manifest, self.ledger)
        # 对象文件按摘要命名，且不会同时存在于两层
        for obj in self.manifest.objects:
            hot = self.engine.object_path("hot", obj.content_digest)
            cold = self.engine.object_path("cold", obj.content_digest)
            self.assertNotEqual(hot.exists(), cold.exists())

    def test_batches_respect_batch_limit(self):
        plan = self.engine.plan(self.ledger, self.manifest)
        batches = split_batches(plan, self.manifest.rules.batch_limit)
        for batch in batches:
            self.assertLessEqual(len(batch), self.manifest.rules.batch_limit)
        self.engine.execute(self.ledger, self.manifest, plan=plan)
        committed = self.ledger.conn.execute(
            "SELECT COUNT(*) FROM batches WHERE status='committed'").fetchone()[0]
        self.assertEqual(committed, len(batches))

    def test_capacity_error_lists_tier_and_rejected_key(self):
        doc = json.loads((self.root / "tiers.json").read_text("utf-8"))
        min_size = min(o.size_bytes for o in self.manifest.objects)
        for tier in doc["tiers"]:
            if tier["role"] == "cold":
                tier["capacity_bytes"] = min_size - 1
        (self.root / "tiers.json").write_text(json.dumps(doc), "utf-8")
        bad = load_manifest(self.root / "tiers.json",
                            self.root / "objects.json",
                            self.root / "seed.json")
        with self.assertRaises(CapacityError) as ctx:
            plan = self.engine.plan(self.ledger, bad)
            self.engine.execute(self.ledger, bad, plan=plan)
        self.assertEqual(ctx.exception.code, "E_CAPACITY")
        self.assertTrue(ctx.exception.rejected_keys)
        self.assertEqual(ctx.exception.tier_id, "cold")

    def test_truncated_file_raises_integrity(self):
        from tier_mover.faults import inject_truncated_object
        from tier_mover.applog import StructuredLogger
        logger = StructuredLogger(self.root / "logs" / "x.jsonl")
        inject_truncated_object(self.root, logger)
        with self.assertRaises(IntegrityError) as ctx:
            self.engine.reconcile_bytes(self.manifest, self.ledger)
        self.assertEqual(ctx.exception.code, "E_INTEGRITY")

    def test_rewrite_policy_skips_oversized_move(self):
        doc = json.loads((self.root / "tiers.json").read_text("utf-8"))
        doc["rules"]["on_capacity"] = "rewrite"
        for tier in doc["tiers"]:
            if tier["role"] == "cold":
                tier["capacity_bytes"] = 1  # 任何对象都放不下
        (self.root / "tiers.json").write_text(json.dumps(doc), "utf-8")
        bad = load_manifest(self.root / "tiers.json",
                            self.root / "objects.json",
                            self.root / "seed.json")
        plan = derive_plan(bad, self.ledger.placement())
        self.assertEqual(plan.steps, ())


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from tier_mover.engine import TierEngine
from tier_mover.ledger import Ledger
from tier_mover.manifest import load_manifest
from tier_mover.workspace import write_sample_manifests


class VersionEvolutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = write_sample_manifests(self.root, seed=11)
        self.engine = TierEngine(self.root)
        self.engine.prepare_dirs(self.manifest)
        self.ledger = Ledger(self.root / "ledger.db")
        self.ledger.register_spec(self.manifest, 100)
        self.engine.seed_objects(self.manifest)
        self.ledger.initialize_objects(self.manifest, 101)

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def test_pending_plan_marked_stale_on_spec_change(self):
        plan = self.engine.plan(self.ledger, self.manifest)
        self.assertEqual(len(self.ledger.superseded_plan_fragments()), 0)

        doc = json.loads((self.root / "tiers.json").read_text("utf-8"))
        doc["spec_version"] = 2
        doc["rules"]["batch_limit"] = 3
        (self.root / "tiers.json").write_text(json.dumps(doc), "utf-8")
        v2 = load_manifest(self.root / "tiers.json",
                           self.root / "objects.json",
                           self.root / "seed.json")
        new_version = self.ledger.register_spec(v2, 200)

        self.assertEqual(new_version, 2)
        stale = self.ledger.superseded_plan_fragments()
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0]["plan_id"],
                         self.ledger.conn.execute(
                             "SELECT plan_id FROM plans").fetchone()[0])
        # 当前视图不包含 stale 计划
        self.assertIsNone(self.ledger.current_view()["pending_plan"])

    def test_conclusion_replaced_not_merged(self):
        plan = self.engine.plan(self.ledger, self.manifest)
        self.engine.execute(self.ledger, self.manifest, plan=plan)
        first = self.ledger.active_conclusion()
        self.assertIsNotNone(first)

        doc = json.loads((self.root / "tiers.json").read_text("utf-8"))
        doc["spec_version"] = 2
        doc["rules"]["hot_watermark_permille"] = 800
        (self.root / "tiers.json").write_text(json.dumps(doc), "utf-8")
        v2 = load_manifest(self.root / "tiers.json",
                           self.root / "objects.json",
                           self.root / "seed.json")
        self.ledger.register_spec(v2, 300)
        plan2 = self.engine.plan(self.ledger, v2)
        self.engine.execute(self.ledger, v2, plan=plan2)
        active = self.ledger.active_conclusion()
        self.assertEqual(active["spec_version"], 2)
        stale_rows = self.ledger.conn.execute(
            "SELECT COUNT(*) FROM conclusions WHERE status='stale'").fetchone()[0]
        self.assertEqual(stale_rows, 1)

    def test_version_monotonic_and_dedup_by_digest(self):
        v1 = self.ledger.active_spec_version()
        same = self.ledger.register_spec(self.manifest, 400)
        self.assertEqual(same, v1)
        doc = json.loads((self.root / "tiers.json").read_text("utf-8"))
        doc["tiers"][0]["unit_cost"] = 9
        (self.root / "tiers.json").write_text(json.dumps(doc), "utf-8")
        changed = load_manifest(self.root / "tiers.json",
                                self.root / "objects.json",
                                self.root / "seed.json")
        self.assertEqual(self.ledger.register_spec(changed, 500), v1 + 1)


if __name__ == "__main__":
    unittest.main()

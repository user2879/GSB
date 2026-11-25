import json
import shutil
import unittest
from pathlib import Path

from ffbudget.cache import CacheBook
from ffbudget.engine import full_replay, run_pipeline
from ffbudget.fingerprint import canonical_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "samples" / "config.json"
BATCHES = ROOT / "samples" / "batches"
REQUESTS = ROOT / "samples" / "requests.json"


class CacheTests(unittest.TestCase):
    def test_hit_miss_accounting_and_isolation(self):
        previous = CacheBook()
        entry = {"score_cache_key": "k1", "value": 1, "request_id": "q1",
                 "item_id": "a", "points": 3}
        previous.record_score(dict(entry))
        current = CacheBook(previous)
        reused = current.record_score(
            {**entry, "request_id": "q2", "item_id": "b"})
        self.assertEqual(current.hits["score"], 1)
        self.assertEqual(reused["request_id"], "q2")
        self.assertEqual(reused["item_id"], "b")
        self.assertEqual(reused["value"], 1)

    def test_invalidated_groups_only_cover_touched_slots(self):
        book = CacheBook()
        book.mark_invalidated_groups(["i#a", "r"])
        summary = book.summary()
        self.assertEqual(summary["invalidated_groups"], ["i#a", "r"])


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.work = ROOT / "state" / "test_snapshot"
        shutil.rmtree(self.work, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def test_full_then_incremental_recovery_and_equivalence(self):
        full, _, store = full_replay(CONFIG, BATCHES, REQUESTS,
                                     self.work / "run")
        manifest_path = self.work / "run" / "manifest.json"
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["stage_artifact_versions"]),
                         {"01_load", "02_authority", "03_view",
                          "04_score"})

        inc, _, _ = run_pipeline(CONFIG, BATCHES, REQUESTS,
                                 self.work / "run", mode="auto")
        self.assertEqual(inc["report"]["recovery"]["mode"], "incremental")
        # Same inputs -> the manifest version is preserved, not bumped.
        self.assertEqual(inc["manifest"]["global_version"],
                         full["manifest"]["global_version"])
        self.assertEqual(inc["manifest"]["input_fingerprint"],
                         full["manifest"]["input_fingerprint"])

        def strip(payload):
            clone = json.loads(json.dumps(payload))
            clone["report"].pop("recovery", None)
            clone["report"].pop("cache", None)
            clone["report"]["budget_ledger"].pop("max_concurrency_used", None)
            clone["manifest"].pop("global_version", None)
            clone["manifest"].pop("input_fingerprint", None)
            return clone

        self.assertEqual(canonical_json(strip(full)),
                         canonical_json(strip(inc)))

    def test_config_change_forces_full_rebuild(self):
        full_replay(CONFIG, BATCHES, REQUESTS, self.work / "run")
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        cfg["budget"]["score_point_budget"] += 1
        changed = self.work / "config_changed.json"
        changed.write_text(json.dumps(cfg), encoding="utf-8")
        result, _, _ = run_pipeline(changed, BATCHES, REQUESTS,
                                    self.work / "run", mode="auto")
        self.assertEqual(result["report"]["recovery"]["mode"],
                         "full_rebuild")
        self.assertEqual(result["report"]["recovery"]["reason"],
                         "E_CONFIG_MISMATCH")

    def test_manifest_corruption_is_detected(self):
        full_replay(CONFIG, BATCHES, REQUESTS, self.work / "run")
        manifest = self.work / "run" / "manifest.json"
        manifest.write_text("{not-json", encoding="utf-8")
        result, _, _ = run_pipeline(CONFIG, BATCHES, REQUESTS,
                                    self.work / "run", mode="auto")
        self.assertEqual(result["report"]["recovery"]["mode"],
                         "full_rebuild")


if __name__ == "__main__":
    unittest.main()

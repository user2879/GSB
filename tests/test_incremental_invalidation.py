import json
import shutil
import unittest
from pathlib import Path

from ffbudget.engine import full_replay, run_pipeline

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "samples" / "config.json"
REQUESTS = ROOT / "samples" / "requests.json"


class IncrementalInvalidationTests(unittest.TestCase):
    def setUp(self):
        self.work = ROOT / "state" / "test_incremental"
        shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir(parents=True)
        self.batch_dir = self.work / "batches"
        self.batch_dir.mkdir()
        source = ROOT / "samples" / "batches"
        for name in ["0001_first.ndjson", "0002_redelivery.ndjson",
                     "0003_second.ndjson"]:
            shutil.copyfile(source / name, self.batch_dir / name)

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def test_new_batch_only_invalidates_covered_groups(self):
        first, _, _ = full_replay(CONFIG, self.batch_dir, REQUESTS,
                                  self.work / "state")
        first_groups = set(first["report"]["cache"]["invalidated_groups"])

        shutil.copyfile(
            ROOT / "samples" / "batches" / "0005_tie.ndjson",
            self.batch_dir / "0004_tie.ndjson")
        second, _, _ = run_pipeline(CONFIG, self.batch_dir, REQUESTS,
                                    self.work / "state", mode="auto")

        self.assertEqual(second["report"]["recovery"]["mode"],
                         "incremental")
        cache = second["report"]["cache"]
        self.assertGreater(cache["hits"]["view"], 0)
        self.assertGreater(cache["hits"]["score"], 0)
        # Every request ranks candidate item_c, so the tie group invalidates
        # all five ranking results: precise, request-level downstream
        # invalidation rather than a coarse global flush.
        self.assertEqual(cache["misses"]["ranking"], 5)
        self.assertEqual(cache["hits"]["ranking"], 0)

        newly_covered = {"cross_affinity#item_c"}
        self.assertEqual(
            set(cache["invalidated_groups"]) - first_groups,
            newly_covered)
        unaffected = first_groups - newly_covered
        self.assertTrue(unaffected)

        # The newly added tie group must now be adjudicated.
        authority = second["authority"]["authorities"]
        self.assertIn("cross_affinity#item_c", authority)
        self.assertTrue(
            authority["cross_affinity#item_c"]["tie_resolved"])

    def test_full_rebuild_matches_final_incremental_state(self):
        full_replay(CONFIG, self.batch_dir, REQUESTS,
                    self.work / "state")
        shutil.copyfile(
            ROOT / "samples" / "batches" / "0004_late.ndjson",
            self.batch_dir / "0005_late.ndjson")
        incremental, _, _ = run_pipeline(
            CONFIG, self.batch_dir, REQUESTS, self.work / "state",
            mode="auto")

        full_state = self.work / "state_full"
        full, _, _ = full_replay(CONFIG, self.batch_dir, REQUESTS,
                                 full_state)

        def strip(payload):
            clone = json.loads(json.dumps(payload))
            clone["report"].pop("recovery", None)
            clone["report"].pop("cache", None)
            clone["report"]["budget_ledger"].pop(
                "max_concurrency_used", None)
            clone["manifest"].pop("global_version", None)
            clone["manifest"].pop("input_fingerprint", None)
            return clone

        from ffbudget.fingerprint import canonical_json
        self.assertEqual(canonical_json(strip(incremental)),
                         canonical_json(strip(full)))


    def test_unaffected_request_keeps_ranking_cache_hit(self):
        # Two requests with disjoint candidate sets; a new batch touches only
        # request B's candidate, so request A's ranking key must hit.
        narrow_requests = self.work / "narrow_requests.json"
        narrow_requests.write_text(json.dumps({"requests": [
            {"request_id": "ra", "rank_ms": 3000,
             "candidates": ["item_a"]},
            {"request_id": "rb", "rank_ms": 3000,
             "candidates": ["item_b"]},
        ]}), encoding="utf-8")
        narrow_batches = self.work / "narrow_batches"
        narrow_batches.mkdir(exist_ok=True)
        shutil.copyfile(
            ROOT / "samples" / "batches" / "0001_first.ndjson",
            narrow_batches / "0001_first.ndjson")
        state_dir = self.work / "narrow_state"
        full_replay(CONFIG, narrow_batches, narrow_requests, state_dir)

        extra = {
            "batch_seq": 2001, "rev_seq": 50,
            "slot_key": "item_profile", "effective_ms": 1000,
            "points": 4, "vector": [7, 7, 7, 7], "item_id": "item_b"}
        with open(narrow_batches / "0002_extra.ndjson", "w",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(extra) + "\n")
        result, _, _ = run_pipeline(CONFIG, narrow_batches,
                                    narrow_requests, state_dir,
                                    mode="auto")
        cache = result["report"]["cache"]
        self.assertGreaterEqual(cache["hits"]["ranking"], 1)
        self.assertEqual(cache["misses"]["ranking"], 1)


if __name__ == "__main__":
    unittest.main()

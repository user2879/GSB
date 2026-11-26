from __future__ import annotations

import tempfile
import unittest

from tests.helpers import conclusion_signature, prepare_project, read_json, run, write_json


def replace_q010_with_music(data: dict) -> None:
    target = next(item for item in data["queries"] if item["query_id"] == "q010")
    target["terms"] = ["请", "想", "播放", "周杰伦", "查", "一下"]
    target["text"] = " ".join(target["terms"])
    target["ground_truth_intent"] = "music"
    target["candidates"] = ["music", "translate", "news"]
    target["slots"] = [
        {"name": "action", "start": 2, "end": 3},
        {"name": "media", "start": 3, "end": 4},
    ]


def add_coffee_weight_revision(data: dict) -> None:
    data["revisions"].append(
        {
            "revision_id": "r08",
            "parent_id": "r06",
            "seq": 8,
            "label": "adjust-coffee-weight",
            "ops": [{"op": "update", "trigger_id": "t_nearby_coffee", "weight": 1100}],
        }
    )
    data["active_revision_ids"] = ["r08"]


class IncrementalTests(unittest.TestCase):
    def test_single_query_replacement_reuses_every_unreferenced_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_project(tmp, include_edits=False)
            baseline = run(paths)
            baseline_intents = {
                item["query_id"]: item["score"]["predicted_intent"] for item in baseline.report["queries"]
            }
            queries_doc = read_json(paths.queries_path)
            replace_q010_with_music(queries_doc)
            write_json(paths.queries_path, queries_doc)
            incremental = run(paths)

            self.assertEqual(incremental.report["summary"]["queries_processed"], 24)
            for item in incremental.report["queries"]:
                if item["query_id"] == "q010":
                    self.assertFalse(item["match_reused"])
                    self.assertFalse(item["score_reused"])
                    self.assertEqual(item["score"]["predicted_intent"], "music")
                else:
                    self.assertTrue(item["match_reused"], item["query_id"])
                    self.assertTrue(item["score_reused"], item["query_id"])
                    self.assertEqual(item["score"]["predicted_intent"], baseline_intents[item["query_id"]])
            self.assertEqual(incremental.report["invalidation_by_revision"]["query-content"]["match"], 1)
            self.assertEqual(incremental.report["invalidation_by_revision"]["query-content"]["score"], 1)

        with tempfile.TemporaryDirectory() as full_tmp:
            full_paths = prepare_project(full_tmp, include_edits=False, mutate_queries=replace_q010_with_music)
            full = run(full_paths)
            self.assertEqual(
                conclusion_signature(incremental.report),
                conclusion_signature(full.report),
            )

    def test_single_trigger_weight_change_only_recomputes_referencing_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_project(tmp, include_edits=False)
            baseline = run(paths)
            baseline_intents = {
                item["query_id"]: item["score"]["predicted_intent"] for item in baseline.report["queries"]
            }
            rules_doc = read_json(paths.rules_path)
            add_coffee_weight_revision(rules_doc)
            write_json(paths.rules_path, rules_doc)
            incremental = run(paths)

            self.assertEqual(incremental.report["summary"]["match_reuse_hits"], 24)
            self.assertEqual(incremental.report["summary"]["score_reuse_hits"], 21)
            changed_queries = {
                item["query_id"]
                for item in incremental.report["queries"]
                if not item["score_reused"]
            }
            self.assertEqual(changed_queries, {"q004", "q016", "q018"})
            self.assertEqual(incremental.report["invalidation_by_revision"]["r08"]["score"], 3)
            self.assertGreaterEqual(incremental.report["invalidation_by_revision"]["r08"]["merge"], 2)
            for item in incremental.report["queries"]:
                if item["query_id"] not in changed_queries:
                    self.assertEqual(item["score"]["predicted_intent"], baseline_intents[item["query_id"]])
            changed = {item["query_id"]: item for item in incremental.report["queries"]}
            self.assertEqual(changed["q004"]["score"]["predicted_intent"], "nearby")
            self.assertEqual(changed["q016"]["score"]["predicted_intent"], "nearby")
            self.assertEqual(changed["q018"]["score"]["predicted_intent"], "unknown")

        with tempfile.TemporaryDirectory() as full_tmp:
            full_paths = prepare_project(full_tmp, include_edits=False, mutate_rules=add_coffee_weight_revision)
            full = run(full_paths)
            self.assertEqual(
                conclusion_signature(incremental.report),
                conclusion_signature(full.report),
            )

    def test_fixed_point_change_reuses_all_matches_and_invalidates_all_scores(self) -> None:
        def change_fixed_point(data: dict) -> None:
            data["fixed_point"] = {"fraction_bits": 16, "quantum": 32}

        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_project(tmp, include_edits=False)
            baseline = run(paths)
            baseline_rule_key = baseline.report["rules"]["cache_key"]
            config = read_json(paths.config_path)
            change_fixed_point(config)
            write_json(paths.config_path, config)
            incremental = run(paths)
            self.assertEqual(incremental.report["rules"]["cache_key"], baseline_rule_key)
            self.assertEqual(incremental.report["summary"]["match_reuse_hits"], 24)
            self.assertEqual(incremental.report["summary"]["score_reuse_hits"], 0)
            self.assertEqual(incremental.report["invalidation_by_revision"]["score-config-key"]["score"], 24)

        with tempfile.TemporaryDirectory() as full_tmp:
            full_paths = prepare_project(full_tmp, include_edits=False, mutate_config=change_fixed_point)
            full = run(full_paths)
            self.assertEqual(
                conclusion_signature(incremental.report),
                conclusion_signature(full.report),
            )


if __name__ == "__main__":
    unittest.main()

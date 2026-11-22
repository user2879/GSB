from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from slowdiag.attribution import attribute_conflicts
from slowdiag.canonical import canonical_json, sha256_canonical
from slowdiag.config import default_config_dict, load_config_data
from slowdiag.errors import E_REPLAY, SlowDiagError
from slowdiag.generator import generate_all
from slowdiag.merge import merge_evidence
from slowdiag.migration import migrate_conclusions
from slowdiag.pipeline import run_pipeline
from slowdiag.snapshots import create_snapshot, restore_snapshot
from slowdiag.storage import Ledger, read_all_sources


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.config_path = self.workspace / "slow-sources.json"
        self.raw = default_config_dict()
        self.config_path.write_text(
            json.dumps(self.raw, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.config = load_config_data(self.raw, self.workspace)
        generate_all(self.config)
        self.ledger = Ledger(self.workspace)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_merge_and_attribution_are_order_independent(self) -> None:
        datasets = read_all_sources(self.config)
        merged_forward = merge_evidence(self.config, datasets)
        merged_reverse = merge_evidence(self.config, list(reversed(datasets)))
        self.assertEqual(
            canonical_json(merged_forward), canonical_json(merged_reverse)
        )
        migration_forward = migrate_conclusions(self.config, datasets)
        attribution_forward = attribute_conflicts(
            self.config, datasets, migration_forward
        )
        attribution_reverse = attribute_conflicts(
            self.config, list(reversed(datasets)), migration_forward
        )
        self.assertEqual(
            canonical_json(attribution_forward),
            canonical_json(attribution_reverse),
        )
        for result in attribution_forward["results"]:
            sort_keys = [
                (-conflict["conflict_score_milli_units"], conflict["metric"])
                for conflict in result["conflicts"]
            ]
            self.assertEqual(sort_keys, sorted(sort_keys))
            for conflict in result["conflicts"]:
                self.assertEqual(
                    [cause["cause"] for cause in conflict["causes"]],
                    ["semantics", "format_version", "sample_period"],
                )
                self.assertEqual(
                    sum(cause["share_bp"] for cause in conflict["causes"]), 10000
                )
                self.assertEqual(
                    sum(
                        cause["contribution_milli_units"]
                        for cause in conflict["causes"]
                    ),
                    conflict["delta"] * 1000,
                )

    def test_full_pipeline_has_all_migration_classes_and_arbitration(self) -> None:
        result = run_pipeline(self.config, self.ledger)
        view = result.view
        self.assertEqual(view["plan_version"], 1)
        self.assertEqual(len(view["queries"]), 8)
        self.assertEqual(
            [query["query_id"] for query in view["queries"] if query["diagnosis"]["slow"]],
            ["q002", "q003", "q005", "q006"],
        )
        for query in view["queries"]:
            for metric in ("elapsed_ms", "scanned_rows"):
                merged = query["metrics"][metric]
                self.assertEqual(merged["adopted"]["source_id"], "db-prod")
                self.assertTrue(merged["comparisons"])
                self.assertEqual(merged["basis"]["rule"][0], "source_type_bias:desc")
        counts = view["migration"]["counts"]
        self.assertGreater(counts["migratable"], 0)
        self.assertGreater(counts["not_migratable"], 0)
        self.assertGreater(counts["insufficient"], 0)
        self.assertGreater(view["attribution"]["counts"]["metric_conflicts"], 0)

    def test_invalidation_propagates_and_preserves_unaffected_bytes(self) -> None:
        first = run_pipeline(self.config, self.ledger)
        old_q7 = sha256_canonical(
            next(query for query in first.view["queries"] if query["query_id"] == "q007")
        )
        old_q8 = sha256_canonical(
            next(query for query in first.view["queries"] if query["query_id"] == "q008")
        )

        raw = copy.deepcopy(self.raw)
        for source in raw["sources"]:
            if source["id"] == "slow-canary":
                source["sample_period_ms"] = 1200
        self.config_path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        changed_config = load_config_data(raw, self.workspace)
        generate_all(changed_config)
        second = run_pipeline(changed_config, self.ledger)
        self.assertEqual(second.view["plan_version"], 2)
        invalidation = second.view["invalidation"]
        self.assertEqual(invalidation["changed_sources"], ["slow-canary"])
        self.assertEqual(
            invalidation["affected_queries"],
            ["q001", "q002", "q003", "q004", "q005", "q006"],
        )
        self.assertTrue(invalidation["unaffected_byte_identical"])
        self.assertEqual(
            [item["query_id"] for item in invalidation["unaffected_checks"]],
            ["q007", "q008"],
        )
        new_q7 = sha256_canonical(
            next(query for query in second.view["queries"] if query["query_id"] == "q007")
        )
        new_q8 = sha256_canonical(
            next(query for query in second.view["queries"] if query["query_id"] == "q008")
        )
        self.assertEqual(old_q7, new_q7)
        self.assertEqual(old_q8, new_q8)

        third = run_pipeline(changed_config, self.ledger)
        self.assertEqual(third.view["plan_version"], 2)
        self.assertEqual(third.view["invalidation"]["affected_queries"], [])
        self.assertTrue(third.view["invalidation"]["unaffected_byte_identical"])

    def test_snapshot_restore_and_tamper_replay_error(self) -> None:
        result = run_pipeline(self.config, self.ledger)
        snapshot_path = create_snapshot(
            self.workspace, result.view, self.config, "test-snapshot.json"
        )
        restored = restore_snapshot(self.ledger, snapshot_path, self.workspace)
        self.assertEqual(restored["snapshot_diff_count"], 0)
        self.assertEqual(restored["view_sha256"], result.view_sha256)

        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        q003 = next(
            query for query in payload["merged"] if query["query_id"] == "q003"
        )
        q003["metrics"]["scanned_rows"]["adopted"]["value"] += 1
        tampered_path = self.workspace / "snapshots" / "tampered.json"
        tampered_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(SlowDiagError) as ctx:
            restore_snapshot(self.ledger, tampered_path, self.workspace)
        self.assertEqual(ctx.exception.code, E_REPLAY)
        self.assertEqual(ctx.exception.query_id, "q003")
        self.assertEqual(ctx.exception.metric, "scanned_rows")
        self.assertEqual(ctx.exception.field, "adopted.value")
        self.assertEqual(ctx.exception.details["replacement_mode"], "whole_view")
        self.assertIsNotNone(self.ledger.get_current_view())

    def test_snapshot_version_mismatch_is_replaced_and_listed(self) -> None:
        result = run_pipeline(self.config, self.ledger)
        snapshot_path = create_snapshot(
            self.workspace, result.view, self.config, "version-mismatch.json"
        )
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        payload["format_version"] = 99
        snapshot_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        restored = restore_snapshot(self.ledger, snapshot_path, self.workspace)
        self.assertGreaterEqual(restored["snapshot_diff_count"], 2)
        paths = [item["path"] for item in restored["snapshot_diffs"]]
        self.assertEqual(paths, sorted(paths))
        self.assertIn("format_version", paths)
        self.assertIn("content_sha256", paths)
        current = self.ledger.get_current_view()
        self.assertIsNotNone(current)
        self.assertEqual(
            current["view"]["invalidation"]["replacement_mode"], "whole_view"
        )
        self.assertEqual(
            current["view"]["invalidation"]["snapshot_diff_count"],
            restored["snapshot_diff_count"],
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from lexicon_pipeline.cache import CacheStore
from lexicon_pipeline.commands import apply_command_stream
from lexicon_pipeline.config import load_config
from lexicon_pipeline.errors import E_FIELD_TRUNCATED, E_FORMAT_INVALID, E_SNAPSHOT_STALE
from lexicon_pipeline.fixedpoint import numpy_integer_sum, quantize_int
from lexicon_pipeline.formats import jv1_to_jv2, jv2_to_jv1, parse_jv1, parse_jv2
from lexicon_pipeline.hashing import canonical_json
from lexicon_pipeline.pipeline import comparison_view, run_state_once
from lexicon_pipeline.runner import execute_local_delivery
from lexicon_pipeline.scheduler import execute_pool, plan_batch
from lexicon_pipeline.segmentation import build_index, segment_document
from lexicon_pipeline.snapshot import SnapshotManager
from lexicon_pipeline.tagging import normalize_transition_weights, tag_document
from lexicon_pipeline.validation import validate_revision_chain


ROOT = Path(__file__).resolve().parents[1]


class SampleMixin:
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs/local.json")
        from lexicon_pipeline.validation import load_and_validate_inputs

        cls.inputs = load_and_validate_inputs(ROOT / "samples")
        cls.base_state = {**cls.inputs, "config": cls.config, "exchange_version": "jv2"}
        cls.index = build_index(cls.inputs["lexicon"])
        cls.transitions = normalize_transition_weights(cls.inputs["weights"], cls.index)


class FormatTests(unittest.TestCase):
    def test_sample_errors_have_locations(self):
        bad_jv1 = parse_jv1((ROOT / "samples/jv1/truncated.jv1").read_text(encoding="utf-8"))
        self.assertTrue(any(i.code == E_FIELD_TRUNCATED and i.line == 2 and i.field == "fragment_seq" for i in bad_jv1.issues))
        bad_jv2 = parse_jv2((ROOT / "samples/jv2/invalid.jv2.json").read_text(encoding="utf-8"))
        self.assertTrue(any(i.code == E_FORMAT_INVALID and i.path == "receipts[0].score" for i in bad_jv2.issues))

    def test_downgrade_sidecar_restores_float_and_naming(self):
        pos_codes = {"NN": 1, "VV": 3}
        original = [
            {
                "doc_seq": 9,
                "fragment_seq": 2,
                "score": 0.8123719,
                "pos": "VV",
                "rounding_mode": "ROUND_HALF_UP",
                "precision_label": "unit-test",
                "precision_labels": ["unit-test"],
            }
        ]
        downgraded = jv2_to_jv1(original, pos_codes)
        self.assertEqual(downgraded.issues, [])
        parsed = parse_jv1(downgraded.text)
        upgraded = jv1_to_jv2(parsed.records, {1: "NN", 3: "VV"}, downgraded.sidecar)
        restored = upgraded.records[0]
        self.assertEqual(restored["score"], original[0]["score"])
        self.assertEqual(restored["pos"], "VV")
        self.assertEqual(restored["rounding_mode"], "ROUND_HALF_UP")
        self.assertTrue(restored["restored_from_sidecar"])
        self.assertEqual(downgraded.sidecar["records"]["0009:0002"]["rounding_code"], 1)


class ValidationTests(unittest.TestCase):
    def test_revision_cycle_detected(self):
        issues = validate_revision_chain(
            {
                "revisions": [
                    {"revision": 0, "parent_revision": 2},
                    {"revision": 1, "parent_revision": 0},
                    {"revision": 2, "parent_revision": 1},
                ]
            }
        )
        self.assertTrue(any(i.code == "E_REVISION_CHAIN" for i in issues))

    def test_missing_parent_detected(self):
        issues = validate_revision_chain({"revisions": [{"revision": 1, "parent_revision": 9}]})
        self.assertTrue(any("missing parent" in i.message for i in issues))


class SegmentationTests(SampleMixin, unittest.TestCase):
    def test_overlap_suppression_and_versioned_rule(self):
        doc = next(d for d in self.inputs["documents"]["documents"] if d["doc_id"] == "D002")
        result = segment_document(doc, self.index, self.inputs["rules"], 3)
        suppressed_terms = {item["term"] for item in result["suppressed"]}
        self.assertIn("人工", suppressed_terms)
        self.assertIn("智能", suppressed_terms)
        surfaces = [fragment["surface"] for fragment in result["fragments"]]
        self.assertEqual(surfaces[:2], ["人工", "智能"])
        decision = next(item for item in result["rule_decisions"] if item["position"] == 2)
        self.assertTrue(decision["conflict"])
        self.assertEqual(decision["winning_rule_id"], "R-SPLIT-AI-AT-2")
        self.assertTrue(result["segmentation_fingerprint"])


class TaggingTests(SampleMixin, unittest.TestCase):
    def test_boundary_fragment_and_high_precision_reference(self):
        doc = next(d for d in self.inputs["documents"]["documents"] if d["doc_id"] == "D006")
        seg = segment_document(doc, self.index, self.inputs["rules"], 3)
        tagged = tag_document(seg, self.index, self.transitions, self.config)
        self.assertTrue(tagged["boundary_fragments"])
        boundary = tagged["boundary_fragments"][0]
        self.assertIn("candidate_tie", boundary["reasons"])
        self.assertEqual(boundary["high_precision_bits"], self.config["effective_fraction_bits"] + 1)

    def test_quantization_width_changes_integer_total(self):
        doc = next(d for d in self.inputs["documents"]["documents"] if d["doc_id"] == "D004")
        seg = segment_document(doc, self.index, self.inputs["rules"], 3)
        totals = set()
        for bits, tier in [(8, "standard"), (10, "standard"), (10, "fine")]:
            config = json.loads(json.dumps(self.config))
            config["fixed_point"]["fraction_bits"] = bits
            config["fixed_point"]["quantization_tier"] = tier
            effective = bits + config["fixed_point"]["tier_shifts"][tier]
            config["effective_fraction_bits"] = effective
            config["fixed_point_scale"] = 1 << effective
            config["high_precision_bits"] = effective + 1
            config["high_precision_scale"] = 1 << (effective + 1)
            tagged = tag_document(seg, self.index, self.transitions, config)
            totals.add((tagged["scale"], tagged["path_total_int"]))
        self.assertEqual(len(totals), 3)

    def test_numpy_fixed_point_accumulation_cross_check(self):
        doc = next(d for d in self.inputs["documents"]["documents"] if d["doc_id"] == "D004")
        seg = segment_document(doc, self.index, self.inputs["rules"], 3)
        tagged = tag_document(seg, self.index, self.transitions, self.config)
        scale = self.config["fixed_point_scale"]
        mode = self.config["fixed_point"]["rounding_mode"]
        values: list[int] = []
        previous_code: int | None = None
        for fragment, code in zip(tagged["fragments"], tagged["selected_codes"]):
            candidate = next(item for item in fragment["candidate_scores"] if item["code"] == code)
            if previous_code is not None:
                raw = self.transitions["by_code"].get((previous_code, code), self.transitions["default"])
                values.append(quantize_int(raw, scale, mode))
            values.append(candidate["emission_int"])
            previous_code = code
        self.assertEqual(numpy_integer_sum(values), tagged["path_total_int"])

    def test_all_zero_ties_choose_lowest_code(self):
        lexicon = {
            "pos_tags": [{"name": "NN", "code": 1}, {"name": "VV", "code": 3}],
            "entries": [
                {"id": "X1", "term": "甲", "frequency": 1, "pos": "NN", "pos_tags": ["NN", "VV"], "enabled": True, "version": 1, "tag_weights": {"NN": 0, "VV": 0}},
                {"id": "X2", "term": "乙", "frequency": 1, "pos": "NN", "pos_tags": ["NN", "VV"], "enabled": True, "version": 1, "tag_weights": {"NN": 0, "VV": 0}},
            ],
        }
        index = build_index(lexicon)
        transitions = normalize_transition_weights({"default_transition_weight": 0, "transition_weights": {}}, index)
        doc = {"doc_id": "X", "doc_seq": 1, "text": "甲乙"}
        seg = segment_document(doc, index, {"rules": []}, 1)
        tagged = tag_document(seg, index, transitions, self.config)
        self.assertEqual(tagged["selected_codes"], [1, 1])


class SchedulerTests(SampleMixin, unittest.TestCase):
    def test_budget_pending_and_quota(self):
        plan = plan_batch(self.inputs["documents"]["documents"], self.index, 700, 300, 1)
        self.assertEqual([t["doc_id"] for t in plan.tasks], ["D001", "D002", "D003"])
        self.assertEqual([p["doc_id"] for p in plan.pending], ["D004", "D005", "D006"])
        self.assertEqual(plan.failures[0]["code"], "E_DOC_QUOTA_EXCEEDED")

    def test_pool_respects_concurrency(self):
        barrier = threading.Barrier(3, timeout=2)

        def worker(task):
            barrier.wait()
            return {"doc_seq": task["doc_seq"], "doc_id": task["doc_id"]}

        tasks = [{"doc_id": f"D{i}", "doc_seq": i} for i in range(1, 4)]
        _, metrics = execute_pool(tasks, worker, 3)
        self.assertEqual(metrics["pool_size"], 3)
        self.assertEqual(metrics["max_observed_concurrency"], 3)


class PipelineEquivalenceTests(SampleMixin, unittest.TestCase):
    def test_command_stream_incremental_and_serial_parallel_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache = tmp_path / "cache.json"
            baseline = run_state_once(self.base_state, cache, None, "concurrent", "baseline")
            applied = apply_command_stream(self.base_state, self.inputs["commands"])
            final_state = applied["state"]
            incremental = run_state_once(final_state, cache, baseline["manifest"], "concurrent", "incremental", applied["command_failures"])
            full = run_state_once(final_state, tmp_path / "full.json", baseline["manifest"], "serial", "full", applied["command_failures"])
            serial = run_state_once(final_state, tmp_path / "serial.json", baseline["manifest"], "serial", "serial", applied["command_failures"])
            parallel = run_state_once(final_state, tmp_path / "parallel.json", baseline["manifest"], "concurrent", "parallel", applied["command_failures"])

            self.assertEqual(canonical_json(comparison_view(incremental)), canonical_json(comparison_view(full)))
            self.assertEqual(canonical_json(comparison_view(serial)), canonical_json(comparison_view(parallel)))
            self.assertTrue(incremental["roundtrip"]["matches"]["ok"])
            self.assertEqual(incremental["roundtrip"]["matches"]["mismatch_count"], 0)

    def test_frequency_edit_reuses_unaffected_fragments(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "reuse.json"
            baseline = run_state_once(self.base_state, cache, None, "serial", "baseline")
            edited = json.loads(json.dumps(self.base_state))
            entry = next(e for e in edited["lexicon"]["entries"] if e["id"] == "E010")
            entry["frequency"] += 51
            entry["version"] += 1
            incremental = run_state_once(edited, cache, baseline["manifest"], "serial", "edit")
            counts = {}
            for event in incremental["cache_events"]:
                counts.setdefault(event["stage"], {"hit": 0, "miss": 0})[ "hit" if event["hit"] else "miss"] += 1
            self.assertGreater(counts["annotation"]["hit"], 0)
            self.assertLessEqual(counts["annotation"]["miss"], 2)
            self.assertIn("E010", incremental["impact"]["changed_entry_ids"])

    def test_delete_and_transition_weight_change_are_precise_and_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = run_state_once(self.base_state, tmp_path / "base.json", None, "serial", "base")

            deleted = json.loads(json.dumps(self.base_state))
            deleted["lexicon"]["entries"] = [e for e in deleted["lexicon"]["entries"] if e["id"] != "E010"]
            inc_delete = run_state_once(deleted, tmp_path / "delete.json", baseline["manifest"], "serial", "delete-inc")
            full_delete = run_state_once(deleted, tmp_path / "delete-full.json", baseline["manifest"], "serial", "delete-full")
            self.assertIn("E010", inc_delete["impact"]["changed_entry_ids"])
            self.assertTrue(any("referenced-entry-changed" in item["basis"] for item in inc_delete["impact"]["invalidations"]))
            self.assertEqual(canonical_json(comparison_view(inc_delete)), canonical_json(comparison_view(full_delete)))

            weighted = json.loads(json.dumps(self.base_state))
            weighted["weights"]["transition_weights"]["NN"]["VV"] = 9.876
            inc_weight = run_state_once(weighted, tmp_path / "weight.json", baseline["manifest"], "serial", "weight-inc")
            full_weight = run_state_once(weighted, tmp_path / "weight-full.json", baseline["manifest"], "serial", "weight-full")
            self.assertIn("NN>VV", inc_weight["impact"]["changed_weight_keys"])
            self.assertTrue(any("referenced-transition-weight-changed" in item["basis"] for item in inc_weight["impact"]["invalidations"]))
            self.assertEqual(canonical_json(comparison_view(inc_weight)), canonical_json(comparison_view(full_weight)))

    def test_added_entry_occurrence_invalidates_overlapping_fragment(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = run_state_once(self.base_state, tmp_path / "base.json", None, "serial", "base")
            added = json.loads(json.dumps(self.base_state))
            added["lexicon"]["entries"].append(
                {
                    "id": "E030",
                    "term": "命",
                    "frequency": 7,
                    "pos": "NN",
                    "pos_tags": ["NN"],
                    "enabled": True,
                    "version": 1,
                    "tag_weights": {"NN": 0.0},
                }
            )
            incremental = run_state_once(added, tmp_path / "add.json", baseline["manifest"], "serial", "add-inc")
            full = run_state_once(added, tmp_path / "add-full.json", baseline["manifest"], "serial", "add-full")
            self.assertIn("E030", incremental["impact"]["changed_entry_ids"])
            self.assertTrue(any("added-entry-match:E030" in item["basis"] for item in incremental["impact"]["invalidations"]))
            self.assertEqual(canonical_json(comparison_view(incremental)), canonical_json(comparison_view(full)))

    def test_full_local_delivery_in_temp_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "samples", root / "samples")
            shutil.copytree(ROOT / "configs", root / "configs")
            report = execute_local_delivery(root)
            self.assertTrue(report["equivalence"]["incremental_vs_blank_full_equal"])
            self.assertTrue(report["equivalence"]["serial_vs_concurrent_equal"])
            self.assertTrue((root / "artifacts/diagnostic-report.json").exists())


class SnapshotTests(unittest.TestCase):
    def test_missing_snapshot_is_stale_and_commit_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SnapshotManager(Path(tmp) / "snapshot")
            missing = manager.startup_decision("hash", 3)
            self.assertEqual(missing["code"], E_SNAPSHOT_STALE)
            manifest = manager.commit({"load": {"ok": True}}, {"config_hash": "hash"}, [], {"cursor": 1}, 10, 3)
            self.assertEqual(manifest["snapshot_version"], 1)
            resume = manager.startup_decision("hash", 3)
            self.assertEqual(resume["action"], "resume")
            rebuild = manager.startup_decision("other", 3)
            self.assertEqual(rebuild["action"], "rebuild")

    def test_interrupted_staging_without_commit_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "snapshot"
            staging = root / ".staging-demo"
            staging.mkdir(parents=True)
            (staging / "partial").write_text("partial", encoding="utf-8")
            manager = SnapshotManager(root)
            decision = manager.startup_decision("hash", 3)
            self.assertEqual(decision["code"], E_SNAPSHOT_STALE)
            self.assertFalse(staging.exists())


if __name__ == "__main__":
    unittest.main()

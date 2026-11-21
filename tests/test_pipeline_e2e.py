import json
import os
import shutil
import tempfile
import unittest

from workbench.cache import task_sort_key
from workbench.engine import Runner, compute_utterance
from workbench.injections import run_injections
from workbench.io_utils import canonical_json, read_json
from workbench.loader import load_samples, load_vocabulary
from workbench.models import HierarchyConfig
from workbench.pipeline import (
    Pipeline,
    compare_runners,
    run_full_rebuild,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = os.path.join(ROOT, "data/vocab/vocabulary.json")
CONFIG = os.path.join(ROOT, "data/config/hierarchy_config.json")
SAMPLES = os.path.join(ROOT, "data/samples/batch_0.jsonl")
PLAN = os.path.join(ROOT, "data/plans/edit_plan.json")
INJECT = os.path.join(ROOT, "data/injections/injection_manifest.json")


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wb-e2e-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.plan = read_json(PLAN)

    def test_incremental_matches_full_rebuild_field_by_field(self):
        inc_dir = os.path.join(self.tmp, "inc")
        full_dir = os.path.join(self.tmp, "full")
        pipeline = Pipeline(VOCAB, CONFIG, SAMPLES, inc_dir)
        incremental = pipeline.run_plan(self.plan)["runner"]
        rebuilt = run_full_rebuild(VOCAB, CONFIG, SAMPLES, self.plan, full_dir)
        report = compare_runners(incremental, rebuilt)
        self.assertTrue(report["identical"], report["mismatches"])
        # every per-utterance record is byte-canonical identical
        for uid, record in incremental.results.items():
            self.assertEqual(canonical_json(record),
                             canonical_json(rebuilt.results[uid]))

    def test_budget_pending_and_todo_priority(self):
        inc_dir = os.path.join(self.tmp, "inc2")
        pipeline = Pipeline(VOCAB, CONFIG, SAMPLES, inc_dir)
        runner = pipeline.run_plan(self.plan)["runner"]
        pending_batches = [r for r in runner.batch_reports if r["budget_pending"]]
        self.assertTrue(pending_batches, "budget exhaustion must produce pending items")
        # at least one pending item later drains (todo priority evidence)
        ever_pending = {uid for r in pending_batches for uid in r["budget_pending"]}
        final_status = {uid: record["status"]
                        for uid, record in runner.results.items()}
        self.assertTrue(any(uid in final_status and final_status[uid] == "completed"
                            for uid in ever_pending))
        # ledger balances
        for entry in runner.ledger:
            self.assertEqual(
                entry["remaining_before"] - entry["deducted"],
                entry["remaining_after"])

    def test_precise_invalidation_only_touches_referenced_intent(self):
        inc_dir = os.path.join(self.tmp, "inc3")
        pipeline = Pipeline(VOCAB, CONFIG, SAMPLES, inc_dir)
        runner = pipeline.run_plan(self.plan)["runner"]
        # report 0 = bootstrap; report 1 = plan B1 (no config change): the
        # unaffected corpus must be reused, only new utterances recomputed.
        b1 = runner.batch_reports[1]
        reused_uids = {item.split(":", 1)[0] for item in b1["reused"]}
        recomputed_uids = {item.split(":", 1)[0] for item in b1["recomputed"]}
        self.assertTrue(reused_uids)
        self.assertIn("U005", reused_uids)
        self.assertNotIn("U005", recomputed_uids)
        self.assertEqual(recomputed_uids, {"U101", "U102"})

        # plan B2 (migration) is report index 2: the utterance referencing the
        # migrated intent template is recomputed rather than silently reused.
        b2 = runner.batch_reports[2]
        migrated_recomputed = {item.split(":", 1)[0]
                               for item in b2["recomputed"]}
        self.assertIn("U002", migrated_recomputed)

    def test_serial_and_parallel_identical(self):
        vocab = load_vocabulary(VOCAB)
        config = HierarchyConfig.from_dict(
            json.load(open(CONFIG, encoding="utf-8")), vocab)
        samples, _ = load_samples(SAMPLES, vocab, config)
        serial = []
        parallel = []
        for workers, sink in ((1, serial), (8, parallel)):
            cfg = json.load(open(CONFIG, encoding="utf-8"))
            cfg["max_workers"] = workers
            local_config = HierarchyConfig.from_dict(cfg, vocab)
            runner = Runner(local_config, os.path.join(self.tmp, f"w{workers}"),
                            mode="incremental")
            runner.run_batch(samples, {
                "affected_utterances": set(samples),
                "stages": ["domain", "intent", "slots", "recalibration"],
                "migrated_intents": set(), "routing_changed": True})
            sink.extend(canonical_json(runner.results[uid])
                        for uid in sorted(runner.results))
        self.assertEqual(serial, parallel)

    def test_stable_task_sort_key_is_deterministic(self):
        key_a = task_sort_key("U001", "domain")
        key_b = task_sort_key("U001", "domain")
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(task_sort_key("U001", "domain"),
                            task_sort_key("U002", "domain"))

    def test_snapshot_manifest_and_atomic_files(self):
        inc_dir = os.path.join(self.tmp, "inc4")
        pipeline = Pipeline(VOCAB, CONFIG, SAMPLES, inc_dir)
        runner = pipeline.run_plan(self.plan)["runner"]
        for name in ("manifest.json", "results.jsonl", "failures.jsonl",
                     "budget_ledger.jsonl", "batch_reports.json"):
            self.assertTrue(os.path.exists(os.path.join(inc_dir, name)))
        manifest = read_json(os.path.join(inc_dir, "manifest.json"))
        self.assertEqual(manifest["config_hash"], runner.config.full_fingerprint())
        self.assertIn("global_version", manifest)
        self.assertIn("intent_index", manifest)

    def test_config_changed_guard(self):
        inc_dir = os.path.join(self.tmp, "inc5")
        pipeline = Pipeline(VOCAB, CONFIG, SAMPLES, inc_dir)
        pipeline.run_plan(self.plan)
        runner = pipeline.runner
        with self.assertRaises(RuntimeError) as ctx:
            runner.verify_manifest_config("deadbeef")
        self.assertEqual(ctx.exception.args[0]["code"], "E_CONFIG_CHANGED")

    def test_injection_isolation_report(self):
        report_path = os.path.join(self.tmp, "injection_report.json")
        manifest = read_json(INJECT)
        report = run_injections(VOCAB, CONFIG, manifest, report_path)
        self.assertTrue(report["all_hit"])
        self.assertEqual(len(report["unique_ids"]), report["total"])
        codes = {row["code"] for row in report["results"]}
        self.assertIn("E_INJECTION_ILLEGAL_TOKEN", codes)
        self.assertIn("E_INJECTION_MIGRATION_CONFLICT", codes)
        self.assertIn("E_THRESHOLD_NOT_FINITE", codes)
        self.assertIn("E_CONFIG_INVALID", codes)
        on_disk = read_json(report_path)
        self.assertEqual(on_disk["results"], report["results"])

    def test_quantization_switch_changes_recorded_values(self):
        inc_dir = os.path.join(self.tmp, "inc6")
        pipeline = Pipeline(VOCAB, CONFIG, SAMPLES, inc_dir)
        runner = pipeline.run_plan(self.plan)["runner"]
        record = runner.results["U102"]
        levels = record["recalibration"]["all_quantization_levels"]
        self.assertNotEqual(
            levels["coarse"]["recal_quantized"],
            levels["fine"]["recal_quantized"])
        self.assertNotEqual(
            record["recalibration"]["confidence_before"],
            record["recalibration"]["confidence_after"])


if __name__ == "__main__":
    unittest.main()

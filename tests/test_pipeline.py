"""Determinism, recovery and validation tests for the news pipeline."""

import json
import os
import shutil
import tempfile
import unittest

from news_pipeline.cli import run
from news_pipeline.config import load_config, load_rewrite_table
from news_pipeline.documents import load_documents
from news_pipeline.edits import apply_all, load_edit_stream
from news_pipeline.fixedpoint import make_qformat
from news_pipeline.pipeline import run_pipeline
from news_pipeline.rewrite import rewrite_sentence
from news_pipeline.sample_generator import generate_all
from news_pipeline.segmentation import segment_document, tokenize


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples", "data")


class PipelineTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = tempfile.mkdtemp(prefix="news-pipeline-")
        cls.sample_dir = os.path.join(cls.temp_root, "samples")
        generate_all(cls.sample_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_root, ignore_errors=True)

    def load_final_inputs(self):
        config = load_config(os.path.join(self.sample_dir, "config.yaml"))
        invalid = []
        table = load_rewrite_table(os.path.join(self.sample_dir, "rewrite_table.yaml"), config, invalid)
        documents, doc_failures = load_documents(os.path.join(self.sample_dir, "documents.jsonl"))
        commands, edit_failures = load_edit_stream(os.path.join(self.sample_dir, "edit_stream.jsonl"))
        failures = invalid + doc_failures + edit_failures
        documents, config, table = apply_all(documents, config, table, commands, failures)
        return documents, config, table, failures

    def snapshot_path(self, name):
        return os.path.join(self.temp_root, name)


class TestValidation(PipelineTestBase):
    def test_invalid_documents_and_rules_enter_failure_list(self):
        config = load_config(os.path.join(self.sample_dir, "config.yaml"))
        invalid = []
        load_rewrite_table(os.path.join(self.sample_dir, "rewrite_table.yaml"), config, invalid)
        _, doc_failures = load_documents(os.path.join(self.sample_dir, "documents.jsonl"))
        codes = {item["code"] for item in invalid + doc_failures}
        self.assertIn("E_INPUT_INVALID", codes)
        self.assertIn("E_REWRITE_RULE_INVALID", codes)
        rule_ids = {item["object_id"] for item in invalid}
        self.assertIn("rule-bad-template", rule_ids)
        self.assertIn("rule-empty-target", rule_ids)

    def test_fixedpoint_quantization_is_deterministic(self):
        q_format = make_qformat(6, "standard")
        self.assertEqual(q_format.frac_bits, 8)
        self.assertEqual(q_format.quantize(0.123456), 32)
        self.assertEqual(q_format.quantize(-0.123456), -32)


class TestSegmentationAndScoring(PipelineTestBase):
    def test_offsets_and_deterministic_tokens(self):
        documents, _ = load_documents(os.path.join(self.sample_dir, "documents.jsonl"))
        first = documents[0]
        run_a = segment_document(first)
        run_b = segment_document(first)
        self.assertEqual(run_a, run_b)
        joined = "".join(sentence["text"] for sentence in run_a)
        self.assertEqual(joined, "".join(p["text"] for p in first["paragraphs"]))
        for sentence in run_a:
            self.assertEqual(first["paragraphs"][sentence["paragraph_index"]]["text"][
                sentence["start_offset"] - _para_cursor(first, sentence["paragraph_index"]):
                sentence["end_offset"] - _para_cursor(first, sentence["paragraph_index"])],
                sentence["text"])
        vocabulary = ("人工智能", "人工", "华新通讯社")
        tokens = tokenize("人工智能由人工驱动", vocabulary)
        self.assertEqual(tokens, ["人工智能", "由", "人工", "驱", "动"])


def _para_cursor(document, paragraph_index):
    return sum(len(p["text"]) for p in document["paragraphs"][:paragraph_index])


class TestEquivalenceAndRecovery(PipelineTestBase):
    def _full_run(self, snapshot_name):
        documents, config, table, failures = self.load_final_inputs()
        codes = {}
        report, results, _store, _cache = run_pipeline(
            documents, config, table, self.snapshot_path(snapshot_name), code_counter=codes
        )
        report["failures"] = failures
        return report, results

    def test_incremental_equals_clean_rebuild(self):
        clean_report, clean_results = self._full_run("snap-clean")

        config = load_config(os.path.join(self.sample_dir, "config.yaml"))
        invalid = []
        table = load_rewrite_table(os.path.join(self.sample_dir, "rewrite_table.yaml"), config, invalid)
        documents, doc_failures = load_documents(os.path.join(self.sample_dir, "documents.jsonl"))
        commands, edit_failures = load_edit_stream(os.path.join(self.sample_dir, "edit_stream.jsonl"))
        failures = invalid + doc_failures + edit_failures
        snapshot = self.snapshot_path("snap-incremental")

        # Baseline run before edits, then one run per command so each edit is
        # recovered from the previous snapshot rather than rebuilt wholesale.
        run_pipeline(documents, config, table, snapshot)
        for command in commands:
            documents, config, table = apply_all(documents, config, table, [command], failures)
            run_pipeline(documents, config, table, snapshot)
        inc_report, inc_results, _, _ = run_pipeline(documents, config, table, snapshot)
        inc_report["failures"] = failures
        self.assertEqual(_artifacts(clean_results), _artifacts(inc_results))
        self.assertEqual(_business_totals(clean_report), _business_totals(inc_report))
        self.assertGreater(
            sum(inc_report["totals"]["cache_reuse"].values()), 0,
            "incremental run must reuse cached sentence artifacts",
        )
        self.assertEqual(
            [d["summary_text"] for d in clean_report["documents"]],
            [d["summary_text"] for d in inc_report["documents"]],
        )

    def test_concurrent_equals_serial(self):
        documents, config, table, _ = self.load_final_inputs()
        serial_config = _clone_config(config, workers=1)
        parallel_config = _clone_config(config, workers=3)
        _, serial_results, _, _ = run_pipeline(
            documents, serial_config, table, self.snapshot_path("snap-serial"))
        _, parallel_results, _, parallel_cache = run_pipeline(
            documents, parallel_config, table, self.snapshot_path("snap-parallel"))
        self.assertEqual(_artifacts(serial_results), _artifacts(parallel_results))
        self.assertLessEqual(
            parallel_cache.stats()["scoring"], len(serial_results) and
            sum(len(r["scores"]) for r in serial_results)
        )

    def test_worker_hard_cap_is_never_exceeded(self):
        import news_pipeline.pipeline as pipeline_mod
        documents, config, table, _ = self.load_final_inputs()
        parallel_config = _clone_config(config, workers=2, sentence_budget=100)
        original = pipeline_mod.process_document
        guard_holder = {}

        import threading
        import time

        lock = threading.Lock()

        def instrumented(document, vocabulary, cfg, tbl, cache):
            with lock:
                active = getattr(instrumented, "active", 0) + 1
                instrumented.active = active
                instrumented.peak = max(getattr(instrumented, "peak", 0), active)
                guard_holder.setdefault("peaks", []).append(active)
            time.sleep(0.01)
            try:
                return original(document, vocabulary, cfg, tbl, cache)
            finally:
                with lock:
                    instrumented.active -= 1

        pipeline_mod.process_document = instrumented
        try:
            report, _, _, _ = run_pipeline(
                documents, parallel_config, table, self.snapshot_path("snap-concurrency-cap"))
        finally:
            pipeline_mod.process_document = original
        self.assertLessEqual(report["run"]["peak_concurrency"], 2)
        self.assertEqual(max(guard_holder["peaks"]), report["run"]["peak_concurrency"])
        self.assertGreaterEqual(report["run"]["peak_concurrency"], 2)

    def test_interrupted_run_resumes_and_matches_full_run(self):
        documents, config, table, _ = self.load_final_inputs()
        snapshot = self.snapshot_path("snap-interrupt")
        report_partial = run(
            self.sample_dir, snapshot,
            os.path.join(self.temp_root, "partial.json"),
            edit_stream=os.path.join(self.sample_dir, "edit_stream.jsonl"),
            stop_after_docs=2,
        )
        self.assertTrue(report_partial["interrupted"])
        self.assertLess(report_partial["run"]["document_count"], len(documents))
        report_resumed = run(
            self.sample_dir, snapshot,
            os.path.join(self.temp_root, "resumed.json"),
            edit_stream=os.path.join(self.sample_dir, "edit_stream.jsonl"),
        )
        clean_report, clean_results = self._full_run("snap-clean-resume")
        resumed_documents, resumed_config, resumed_table, _ = self.load_final_inputs()
        _, resumed_results, _, _ = run_pipeline(
            resumed_documents, resumed_config, resumed_table, snapshot)
        self.assertEqual(_artifacts(resumed_results), _artifacts(clean_results))
        self.assertEqual(_business_totals(report_resumed), _business_totals(clean_report))
        self.assertGreater(sum(report_resumed["totals"]["cache_reuse"].values()), 0)

    def test_stale_snapshot_triggers_rebuild_code(self):
        documents, config, table, _ = self.load_final_inputs()
        snapshot = self.snapshot_path("snap-stale")
        run_pipeline(documents, config, table, snapshot)
        changed = _clone_config(config, coverage_floor=0.9)
        codes = {}
        report, _results, _, _ = run_pipeline(documents, changed, table, snapshot, code_counter=codes)
        self.assertEqual(codes.get("E_SNAPSHOT_STALE"), 1)
        self.assertTrue(report["snapshot_stale_events"])


class TestRewriteAndBudget(PipelineTestBase):
    def test_overlap_suppression_and_rule_order(self):
        config = load_config(os.path.join(self.sample_dir, "config.yaml"))
        invalid = []
        table = load_rewrite_table(os.path.join(self.sample_dir, "rewrite_table.yaml"), config, invalid)
        record = rewrite_sentence("人工智能与人工协同", table, "formal")
        applied_ids = [item["rule_id"] for item in record["applied_rules"]]
        suppressed_ids = [item["rule_id"] for item in record["suppressed_rules"]]
        self.assertIn("rule-ai", applied_ids)
        self.assertIn("rule-ai-short", suppressed_ids)

    def test_budget_pending_code_and_report(self):
        documents, config, table, _ = self.load_final_inputs()
        tight = _clone_config(config, sentence_budget=1, workers=2)
        codes = {}
        report, results, _, _ = run_pipeline(
            documents, tight, table, self.snapshot_path("snap-budget"), code_counter=codes
        )
        self.assertEqual(len(results), len(documents))
        self.assertIn("E_SENTENCE_BUDGET_PENDING", codes)
        self.assertLessEqual(report["run"]["peak_concurrency"], 2)


class TestFineGrainedInvalidation(PipelineTestBase):
    def test_paragraph_edit_only_invalidates_its_sentences(self):
        config = load_config(os.path.join(self.sample_dir, "config.yaml"))
        invalid = []
        table = load_rewrite_table(os.path.join(self.sample_dir, "rewrite_table.yaml"), config, invalid)
        documents, _ = load_documents(os.path.join(self.sample_dir, "documents.jsonl"))
        snapshot = self.snapshot_path("snap-invalidation")
        _, first_results, _, cache = run_pipeline(documents, config, table, snapshot)
        first_score_keys = {
            record["cache_key"]
            for result in first_results for record in result["scores"]
        }

        target = next(doc for doc in documents if doc["doc_id"] == "doc-0002")
        untouched_pid = target["paragraphs"][1]["pid"]
        commands = [{
            "command_id": "edit-para", "op": "update_paragraph", "doc_id": "doc-0002",
            "pid": target["paragraphs"][0]["pid"],
            "text": "今日华新通讯社在深圳发布人工智能合作协议。东方研究院推进新一代平台建设。",
        }]
        failures = []
        changed_documents, _, _ = apply_all(documents, config, table, commands, failures)
        _, second_results, _, _ = run_pipeline(changed_documents, config, table, snapshot)

        second = next(result for result in second_results if result["doc_id"] == "doc-0002")
        untouched = [s for s in second["segmentation"]["sentences"] if s["paragraph_pid"] == untouched_pid]
        self.assertTrue(untouched)
        # Every sentence from the untouched paragraph must still resolve to a
        # score key that existed before the edit, proving no re-scoring.
        for record in second["scores"]:
            if record["paragraph_pid"] == untouched_pid:
                self.assertIn(record["cache_key"], first_score_keys)
        self.assertGreaterEqual(second["reused"]["scoring"], len(untouched))

    def test_rewrite_rule_edit_invalidates_only_hit_sentences(self):
        config = load_config(os.path.join(self.sample_dir, "config.yaml"))
        invalid = []
        table = load_rewrite_table(os.path.join(self.sample_dir, "rewrite_table.yaml"), config, invalid)
        documents, _ = load_documents(os.path.join(self.sample_dir, "documents.jsonl"))
        snapshot = self.snapshot_path("snap-rule-invalidation")
        _, first_results, _, _ = run_pipeline(documents, config, table, snapshot)
        first_rewrite_keys = {
            record["cache_key"]
            for result in first_results for record in result["rewrites"]
        }

        commands = [{
            "command_id": "edit-rule", "op": "update_rule", "rule_id": "rule-ai",
            "target": "人工智能核心技术",
        }]
        failures = []
        changed_documents, changed_config, changed_table = apply_all(
            documents, config, table, commands, failures)
        _, second_results, _, _ = run_pipeline(
            changed_documents, changed_config, changed_table, snapshot)
        reused_keys = {
            record["cache_key"]
            for result in second_results for record in result["rewrites"]
            if record["cache_key"] in first_rewrite_keys
        }
        self.assertTrue(reused_keys, "sentences without the edited rule must reuse rewrite artifacts")


class TestBoundarySensitivity(unittest.TestCase):
    def _boundary_document(self):
        from news_pipeline.hashing import fingerprint
        document = {
            "doc_id": "doc-boundary",
            "title": "边界测试",
            "paragraphs": [
                {"pid": "p0", "text": "阿尔法贝塔伽马德尔塔。"},
                {"pid": "p1", "text": "阿尔法贝塔伽马。"},
            ],
            "entities": [
                {"text": "阿尔法", "type": "x"}, {"text": "贝塔", "type": "x"},
                {"text": "伽马", "type": "x"},
            ],
            "events": [],
        }
        document["content_fingerprint"] = fingerprint(document)
        return document

    def test_quantized_tie_vs_float_order_is_flagged_as_boundary(self):
        from news_pipeline.segmentation import assign_sentence_ids
        from news_pipeline.scoring import score_document
        from news_pipeline.selection import select_sentences

        document = self._boundary_document()
        vocabulary = ("阿尔法", "贝塔", "伽马", "德尔塔")
        sentences = assign_sentence_ids(segment_document(document))
        config = _clone_config(
            load_config(os.path.join(SAMPLES, "config.yaml")),
            fixedpoint_width=6, quantization_tier="standard",
            coverage_floor=0.0, max_sentences=1, workers=1,
            term_weight=0.04, entity_weight=0.0, position_decay=0.3,
        )
        scores = score_document(sentences, document, vocabulary, config, config.q_format)
        # Each paragraph is its own position unit (factor 1). Float prefers
        # sentence 1, but both quantize to one bin and the offset tie-break
        # picks sentence 0, producing a symmetric boundary decision.
        self.assertLess(scores[0]["reference_score"], scores[1]["reference_score"])
        self.assertEqual(scores[0]["fixed_score"], scores[1]["fixed_score"])
        selection = select_sentences(scores, document, config)
        self.assertEqual(len(selection["selected_sentence_ids"]), 1)
        boundary = selection["boundary_sentences"]
        self.assertTrue(boundary, "fixed/float disagreement must be reported, not dropped")
        self.assertEqual(selection["selected_sentence_ids"], ["doc-boundary#s000"])
        self.assertEqual(
            {item["sentence_id"] for item in boundary},
            {"doc-boundary#s000", "doc-boundary#s001"},
        )
        for entry in boundary:
            self.assertIn("fixed_score", entry)
            self.assertIn("reference_score", entry)
            self.assertGreater(entry["score_difference"], 0.0)
        fixed_ids = set(selection["selected_sentence_ids"])
        self.assertEqual({item["sentence_id"] for item in boundary if item["in_fixed_selection"]},
                         fixed_ids)

    def test_tier_switch_changes_fixed_integers(self):
        from news_pipeline.segmentation import assign_sentence_ids
        from news_pipeline.scoring import score_document

        document = self._boundary_document()
        vocabulary = ("阿尔法", "贝塔", "伽马", "德尔塔")
        sentences = assign_sentence_ids(segment_document(document))
        coarse = _clone_config(load_config(os.path.join(SAMPLES, "config.yaml")),
                               fixedpoint_width=6, quantization_tier="coarse",
                               coverage_floor=0.0, max_sentences=1, workers=1)
        fine = _clone_config(coarse, fixedpoint_width=8, quantization_tier="fine")
        coarse_scores = score_document(sentences, document, vocabulary, coarse, coarse.q_format)
        fine_scores = score_document(sentences, document, vocabulary, fine, fine.q_format)
        self.assertNotEqual(
            [s["fixed_score"] for s in coarse_scores],
            [s["fixed_score"] for s in fine_scores],
        )


def _clone_config(config, **changes):
    from news_pipeline.edits import _rebuild_config
    return _rebuild_config(config, **changes)


def _artifacts(results):
    stripped = [
        {
            key: (
                {k: v for k, v in value.items() if k != "cache_reuse"}
                if key == "summary" and isinstance(value, dict)
                else value
            )
            for key, value in result.items()
            if key != "reused"
        }
        for result in results
    ]
    return json.dumps(stripped, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _business_totals(report):
    totals = dict(report["totals"])
    totals.pop("cache_reuse", None)
    return totals


if __name__ == "__main__":
    unittest.main()

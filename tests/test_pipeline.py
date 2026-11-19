from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pipeline.config import load_config
from pipeline.generator import generate_sample
from pipeline.runner import PipelineRunner


def _comparable(report: dict) -> dict:
    return {
        "determinism": report.get("determinism"),
        "queries": report.get("queries"),
        "budget": report.get("budget"),
        "evaluation": report.get("evaluation"),
        "failures": report.get("failures"),
        "injections": report.get("injections"),
        "snapshot_version": report.get("snapshot", {}).get("version"),
        "completed_batches": report.get("snapshot", {}).get("completed_batches"),
    }


class PipelineAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp(prefix="pipeline-tests-"))
        cls.sample = cls.root / "sample"
        generate_sample(cls.sample, seed=7719, documents=18, queries=8, zh_ratio=0.5, mixed_ratio=0.25)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def _run(self, name: str, **kwargs):
        options = {
            "sample_dir": self.sample,
            "state_dir": self.root / f"state-{name}",
            "report_path": self.root / f"{name}.json",
            "fresh": True,
            "max_workers": 4,
        }
        options.update(kwargs)
        return PipelineRunner.from_sample(**options).run()

    def test_language_detection_bridge_suppression_and_missing_translation(self) -> None:
        report = self._run("normal")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["queries"]), 8)
        for query in report["queries"]:
            self.assertIn(query["detected_language"], {"zh", "en", "mixed"})
            evidence = query["language_evidence"]
            self.assertIn("han_ratio", evidence)
            self.assertIn("latin_ratio", evidence)
            self.assertIn("translation_coverage_zh", evidence)
            self.assertIn("rule", evidence)
        expansions = [expansion for query in report["queries"] for expansion in query["translation_expansions"]]
        self.assertTrue(expansions)
        self.assertTrue(any(expansion["suppressed"] for expansion in expansions))
        self.assertTrue(any(query["missing_translation_terms"] for query in report["queries"]))
        self.assertTrue(any(result["bridge_score"] > 0 for query in report["queries"] for result in query["results"]))
        for query in report["queries"]:
            for result in query["results"]:
                self.assertEqual(result["fixed_width"], 64)
                self.assertEqual(result["fixed_scale"], 1000000)
                self.assertEqual(result["candidate_key"], f"{query['query_id']}|{result['doc_id']}|retrieval_scoring")
                keys = [term["stable_key"] for term in result["term_breakdown"]]
                self.assertEqual(keys, sorted(keys))

    def test_stable_tie_break_by_document_id(self) -> None:
        report = self._run("ties")
        for query in report["queries"]:
            results = query["results"]
            for left, right in zip(results, results[1:]):
                if left["score_fixed"] == right["score_fixed"] and left["mono_fixed"] == right["mono_fixed"]:
                    self.assertLess(left["doc_id"], right["doc_id"])

    def test_serial_and_parallel_are_field_identical(self) -> None:
        serial = self._run("serial", max_workers=1)
        parallel = self._run("parallel", max_workers=4)
        self.assertEqual(serial["status"], "complete")
        self.assertEqual(parallel["status"], "complete")
        self.assertEqual(_comparable(serial), _comparable(parallel))
        self.assertLessEqual(serial["run"]["observed_max_active_workers"], 1)
        self.assertLessEqual(parallel["run"]["observed_max_active_workers"], 4)

    def test_full_and_resumed_incremental_are_equal(self) -> None:
        full = self._run("full")
        interrupted = self._run("incremental", stop_after_batches=3)
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertEqual(interrupted["run"]["exit_code"], 0)
        resumed = PipelineRunner.from_sample(
            sample_dir=self.sample,
            state_dir=self.root / "state-incremental",
            report_path=self.root / "incremental-resumed.json",
            fresh=False,
            max_workers=4,
        ).run()
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(_comparable(full), _comparable(resumed))
        self.assertEqual(full["snapshot"]["version"], resumed["snapshot"]["version"])

    def test_budget_allocation_and_exhaustion_codes(self) -> None:
        report = self._run("budget", global_budget=8, per_query_quota=4)
        sequence = report["budget"]["sequence"]
        self.assertEqual([item["query_id"] for item in sequence], sorted(item["query_id"] for item in sequence))
        self.assertLessEqual(report["budget"]["consumed_total"], 8)
        self.assertTrue(any(item["code"] == "E_QUERY_BUDGET_EXHAUSTED" for item in sequence))
        self.assertTrue(any(item["code"] == "E_BUDGET_EXHAUSTED" for item in sequence))
        positions = [query["position"] for query in report["queries"]]
        self.assertEqual(positions, list(range(1, len(positions) + 1)))

    def test_all_injection_kinds_hit_and_remain_isolated(self) -> None:
        report = self._run("injection", injections_path=self.sample / "injections.json")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["injections"]), 4)
        self.assertTrue(all(item["hit"] for item in report["injections"]))
        codes = {item["failure"]["code"] for item in report["injections"]}
        self.assertEqual(
            codes,
            {
                "E_RECORD_INVALID",
                "E_TRANSLATION_MISSING",
                "E_RECALL_LIMIT_ZERO",
                "E_RELEVANCE_GRADE_OUT_OF_RANGE",
            },
        )
        self.assertTrue(any(query["status"] == "ok" for query in report["queries"]))

    def test_config_hash_mismatch_refuses_resume_without_mutation(self) -> None:
        first = self._run("mismatch")
        self.assertEqual(first["status"], "complete")
        manifest_path = self.root / "state-mismatch" / "manifest.json"
        before = manifest_path.read_bytes()
        config, _ = load_config(self.sample / "config.json")
        changed = copy.deepcopy(config)
        changed["bm25"]["k1"] = changed["bm25"]["k1"] + 0.1
        changed_path = self.root / "changed-config.json"
        changed_path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
        report = PipelineRunner.from_sample(
            sample_dir=self.sample,
            state_dir=self.root / "state-mismatch",
            report_path=self.root / "mismatch-report.json",
            config_path=changed_path,
            fresh=False,
            max_workers=4,
        ).run()
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["run"]["exit_code"], 2)
        self.assertEqual(report["fatal_errors"][0]["code"], "E_CONFIG_MISMATCH")
        self.assertEqual(manifest_path.read_bytes(), before)

    def test_invalid_records_are_isolated(self) -> None:
        invalid_sample = self.root / "invalid-sample"
        shutil.copytree(self.sample, invalid_sample)
        with (invalid_sample / "documents.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"doc_id": "d0001", "lang": "zh", "title": "重复", "body": "重复"}, ensure_ascii=False) + "\n")
            handle.write(json.dumps({"doc_id": "d9001", "lang": "fr", "title": "错误语言", "body": "正文"}, ensure_ascii=False) + "\n")
            handle.write(json.dumps({"doc_id": "d9002", "lang": "zh", "title": "", "body": "正文"}, ensure_ascii=False) + "\n")
            handle.write("{not valid json}\n")
        with (invalid_sample / "queries.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"query_id": "q0001", "lang": "zh", "query": "重复查询"}, ensure_ascii=False) + "\n")
        with (invalid_sample / "translations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"source_lang":"zh","source_term":"坏置信度","target_lang":"en","target_term":"bad_confidence","confidence":NaN}\n')
        report = PipelineRunner.from_sample(
            sample_dir=invalid_sample,
            state_dir=self.root / "state-invalid",
            report_path=self.root / "invalid.json",
            fresh=True,
            max_workers=2,
        ).run()
        self.assertEqual(report["status"], "complete")
        self.assertGreaterEqual(len(report["failures"]), 6)
        self.assertTrue(all(item["code"] == "E_RECORD_INVALID" for item in report["failures"]))
        self.assertEqual(len(report["queries"]), 8)
        self.assertTrue(any(item["exception_type"] == "JSONDecodeError" for item in report["failures"]))

    def test_generator_is_byte_for_byte_reproducible(self) -> None:
        rebuilt = self.root / "rebuilt-sample"
        generate_sample(rebuilt, seed=7719, documents=18, queries=8, zh_ratio=0.5, mixed_ratio=0.25)
        for filename in (
            "documents.jsonl",
            "queries.jsonl",
            "translations.jsonl",
            "qrels.jsonl",
            "config.json",
            "injections.json",
            "sample.json",
        ):
            self.assertEqual((self.sample / filename).read_bytes(), (rebuilt / filename).read_bytes(), filename)


if __name__ == "__main__":
    unittest.main()

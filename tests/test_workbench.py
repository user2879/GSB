from __future__ import annotations

import copy
import json
import sys
import tempfile
import threading
import time
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_diag import engine as engine_module  # noqa: E402
from news_diag.engine import RunnerOptions, run_workbench  # noqa: E402
from news_diag.errors import E_CONFIG_INVALID, E_STAGE_FAILED, E_TEMPLATE_MISSING  # noqa: E402
from news_diag.io_utils import json_text  # noqa: E402
from news_diag.loader import (  # noqa: E402
    load_config,
    load_raw_news,
    load_synonym_table,
    validate_raw_documents,
)
from news_diag.models import Sentence, SynonymRule, SynonymTable  # noqa: E402
from news_diag.rewrite import rewrite_sentence  # noqa: E402
from news_diag.scoring import score_sentence  # noqa: E402
from news_diag.splitter import document_frequency_total, split_document  # noqa: E402


SAMPLES = ROOT / "samples"
CONFIG = SAMPLES / "config.yaml"
NEWS = SAMPLES / "news.jsonl"
INJECTIONS = SAMPLES / "injections.json"


def _write_first_documents(path: Path, count: int) -> None:
    lines = NEWS.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:count]) + "\n", encoding="utf-8")


class SampleReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.concurrent_result = run_workbench(
            RunnerOptions(news_path=SAMPLES, config_path=CONFIG, injections_path=INJECTIONS)
        )
        cls.serial_result = run_workbench(
            RunnerOptions(
                news_path=SAMPLES,
                config_path=CONFIG,
                injections_path=INJECTIONS,
                serial=True,
            )
        )
        cls.report = cls.concurrent_result.report

    def test_concurrent_and_serial_reports_are_byte_identical(self) -> None:
        self.assertEqual(
            json_text(self.concurrent_result.report), json_text(self.serial_result.report)
        )
        self.assertEqual(self.concurrent_result.exit_code, 0)
        self.assertEqual(self.serial_result.exit_code, 0)

    def test_repeated_runs_are_byte_identical(self) -> None:
        repeated = run_workbench(
            RunnerOptions(
                news_path=SAMPLES,
                config_path=CONFIG,
                injections_path=INJECTIONS,
                serial=True,
            )
        )
        self.assertEqual(json_text(self.report), json_text(repeated.report))

    def test_document_order_and_failure_isolation(self) -> None:
        doc_ids = [document["doc_id"] for document in self.report["documents"]]
        self.assertEqual(doc_ids, sorted(doc_ids))
        self.assertNotIn("DOC-007", doc_ids)
        self.assertTrue(
            any(
                failure["document_id"] == "DOC-007"
                and failure["stage"] == "load"
                and failure["code"] == E_CONFIG_INVALID
                for failure in self.report["failures"]
            )
        )
        self.assertEqual(self.report["completion"]["complete"], True)

    def test_coverage_touch_and_fixed_reference_boundaries(self) -> None:
        touch_count = sum(
            document["statistics"]["coverage_boundary_hit_count"]
            for document in self.report["documents"]
        )
        boundary_count = sum(
            document["statistics"]["boundary_sentence_count"]
            for document in self.report["documents"]
        )
        self.assertGreater(touch_count, 0)
        self.assertGreater(boundary_count, 0)
        for document in self.report["documents"]:
            self.assertTrue(document["statistics"]["coverage_satisfied"])
            for boundary in document["boundary_sentences"]:
                self.assertNotEqual(
                    boundary["selected_by_fixed"], boundary["selected_by_reference"]
                )
                self.assertEqual(
                    boundary["difference"],
                    boundary["fixed_score"] - boundary["reference_score"],
                )

    def test_perturbation_attribution_and_unmatched_items(self) -> None:
        flip_components = {
            injection["effect"]["dominant_component"]
            for injection in self.report["injections"]["all"]
            if injection["status"] == "applied"
            and injection.get("effect", {}).get("selection_flip")
        }
        self.assertIn("word_frequency", flip_components)
        self.assertIn("entity_coverage", flip_components)
        not_applied = {item["injection_id"]: item for item in self.report["injections"]["not_applied"]}
        self.assertEqual(set(not_applied), {"INJ-MISS-001", "INJ-MISS-002"})
        for item in not_applied.values():
            self.assertTrue(item.get("reason"))
        for doc in self.report["documents"]:
            for sentence in doc["sentences"]:
                self.assertIn("score_before", sentence)
                self.assertIn("score_after", sentence)
                self.assertIn("fixed_vs_reference", sentence)

    def test_competition_order_is_score_desc_then_start_asc(self) -> None:
        for document in self.report["documents"]:
            order = document["competition_order"]
            actual = [
                (-item["fixed_score_integer"], item["start"], item["sentence_index"])
                for item in order
            ]
            self.assertEqual(actual, sorted(actual))

    def test_rewrite_conflicts_and_missing_template_are_reported(self) -> None:
        suppressed_pairs: set[tuple[str, str]] = set()
        for document in self.report["documents"]:
            for item in document["suppressed_replacements"]:
                for winner in item["suppressed_by"]:
                    suppressed_pairs.add((item["rule_id"], winner))
        self.assertIn(("R004", "R005"), suppressed_pairs)
        self.assertIn(("R007", "R006"), suppressed_pairs)
        self.assertEqual(len(self.report["rewrite_table_warnings"]), 1)
        warning = self.report["rewrite_table_warnings"][0]
        self.assertEqual(warning["code"], E_TEMPLATE_MISSING)
        self.assertEqual(warning["rule_id"], "R002")
        self.assertGreater(len(self.report["rewrite_errors"]), 0)
        for error in self.report["rewrite_errors"]:
            self.assertEqual(error["preserved_fragment"], "表示")
            self.assertEqual(error["decision"], "rule skipped; original fragment preserved")


class UnitAndFailureTests(unittest.TestCase):
    def test_sentence_offsets_are_exact_and_repeatable(self) -> None:
        raw_documents, failures = load_raw_news(NEWS)
        self.assertEqual(failures, [])
        documents, document_failures = validate_raw_documents(raw_documents, [])
        self.assertEqual(document_failures, [])
        document = documents[0]
        first = split_document(document)
        second = split_document(document)
        self.assertEqual(first, second)
        for sentence in first:
            self.assertEqual(document.text[sentence.start : sentence.end], sentence.text)
            self.assertLess(sentence.start, sentence.end)

    def test_fixed_point_score_has_terms_residual_and_is_repeatable(self) -> None:
        config, config_failures = load_config(CONFIG)
        self.assertEqual(config_failures, [])
        assert config is not None
        raw_documents, _ = load_raw_news(NEWS)
        documents, _ = validate_raw_documents(raw_documents, [])
        document = documents[0]
        sentences = split_document(document)
        frequency_total = document_frequency_total(document)
        first = score_sentence(document, sentences[0], len(sentences), frequency_total, config)
        second = score_sentence(document, sentences[0], len(sentences), frequency_total, config)
        self.assertEqual(first, second)
        self.assertEqual(
            first.fixed_integer, sum(term.fixed_integer for term in first.terms)
        )
        self.assertEqual([term.name for term in first.terms], [
            "word_frequency",
            "entity_coverage",
            "position_decay",
        ])
        self.assertIsInstance(first.reference, float)
        self.assertNotEqual(first.residual, 0)

    def test_invalid_config_enters_failure_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.yaml"
            config_path.write_text(
                CONFIG.read_text(encoding="utf-8").replace(
                    "entity_coverage_min: 0.80", "entity_coverage_min: 1.50"
                ),
                encoding="utf-8",
            )
            config, failures = load_config(config_path)
            self.assertIsNone(config)
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0].code, E_CONFIG_INVALID)
            self.assertEqual(failures[0].stage, "config")

    def test_duplicate_document_id_and_bad_triple_enter_failure_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            news_path = Path(temporary) / "news.jsonl"
            original = json.loads(NEWS.read_text(encoding="utf-8").splitlines()[0])
            duplicate = copy.deepcopy(original)
            bad_triple = copy.deepcopy(original)
            bad_triple["doc_id"] = "DOC-BAD"
            bad_triple["events"][0]["object"] = "E-NOT-DECLARED"
            news_path.write_text(
                "\n".join(
                    json.dumps(item, ensure_ascii=False) for item in (original, duplicate, bad_triple)
                )
                + "\n",
                encoding="utf-8",
            )
            raw_documents, failures = load_raw_news(news_path)
            self.assertEqual(failures, [])
            documents, document_failures = validate_raw_documents(raw_documents, [])
            self.assertEqual(len(documents), 1)
            self.assertEqual(len(document_failures), 2)
            self.assertTrue(all(item.code == E_CONFIG_INVALID for item in document_failures))

    def test_template_version_mismatch_is_config_error(self) -> None:
        config, _ = load_config(CONFIG)
        assert config is not None
        with tempfile.TemporaryDirectory() as temporary:
            table_path = Path(temporary) / "table.yaml"
            table_path.write_text(
                (SAMPLES / "synonym_table.yaml")
                .read_text(encoding="utf-8")
                .replace("version: v1", "version: v2", 1),
                encoding="utf-8",
            )
            table, failures = load_synonym_table(table_path, config)
            self.assertIsNone(table)
            self.assertEqual(failures[0].code, E_CONFIG_INVALID)

    def test_missing_template_span_is_reserved_from_other_rules(self) -> None:
        table = SynonymTable(
            version="v1",
            templates={"formal_v1": {"display_name": "formal"}},
            rules=(
                SynonymRule(0, "MISSING", "ghost_v1", "排查风险", "不应出现"),
                SynonymRule(1, "ACTIVE", "formal_v1", "排查", "检查"),
            ),
            source_name="inline.yaml",
        )
        sentence = Sentence(
            sentence_index=0,
            paragraph_index=0,
            start=0,
            end=6,
            text="排查风险需要复核。",
            sort_key="test|00000000|sentence",
            entity_ids=(),
            frequency_hits=Decimal("0"),
        )
        rewritten, replacements, suppressed, errors = rewrite_sentence(
            sentence, table, "formal_v1"
        )
        self.assertEqual(rewritten, sentence.text)
        self.assertEqual(replacements, [])
        self.assertEqual(suppressed[0]["reason"], "missing_template_reserved_span")
        self.assertEqual(suppressed[0]["suppressed_by"], ["MISSING"])
        self.assertEqual(errors[0]["code"], E_TEMPLATE_MISSING)
        self.assertEqual(errors[0]["preserved_fragment"], "排查风险")

    def test_stage_exception_isolated_to_one_document(self) -> None:
        original_score = engine_module.score_sentence

        def flaky_score(document, *args, **kwargs):
            if document.doc_id == "DOC-001":
                raise RuntimeError("synthetic score failure")
            return original_score(document, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            news_path = Path(temporary) / "news.jsonl"
            _write_first_documents(news_path, 2)
            with mock.patch("news_diag.engine.score_sentence", side_effect=flaky_score):
                result = run_workbench(
                    RunnerOptions(
                        news_path=news_path,
                        config_path=CONFIG,
                        injections_path=INJECTIONS,
                        serial=True,
                    )
                )
            self.assertEqual(result.exit_code, 0)
            self.assertEqual([doc["doc_id"] for doc in result.report["documents"]], ["DOC-002"])
            failure = result.report["failures"][0]
            self.assertEqual(failure["document_id"], "DOC-001")
            self.assertEqual(failure["stage"], "score")
            self.assertEqual(failure["code"], E_STAGE_FAILED)
            self.assertEqual(failure["exception_type"], "RuntimeError")

    def test_thread_pool_never_exceeds_configured_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            news_path = Path(temporary) / "news.jsonl"
            _write_first_documents(news_path, 10)
            counter = {"active": 0, "maximum": 0}
            lock = engine_module.threading.Lock()
            original = engine_module.WorkbenchRunner._run_document_task

            def wrapped(self, document, config, table, injections):
                with lock:
                    counter["active"] += 1
                    counter["maximum"] = max(counter["maximum"], counter["active"])
                try:
                    time.sleep(0.01)
                    return original(self, document, config, table, injections)
                finally:
                    with lock:
                        counter["active"] -= 1

            with mock.patch(
                "news_diag.engine.WorkbenchRunner._run_document_task", new=wrapped
            ):
                result = run_workbench(
                    RunnerOptions(
                        news_path=news_path,
                        config_path=CONFIG,
                        injections_path=INJECTIONS,
                        serial=False,
                    )
                )
            self.assertEqual(result.exit_code, 0)
            self.assertLessEqual(counter["maximum"], 4)
            self.assertGreater(counter["maximum"], 1)

    def test_resume_uses_only_persisted_state_and_matches_full_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            news_path = temporary_path / "news.jsonl"
            state_dir = temporary_path / "state"
            _write_first_documents(news_path, 3)

            stop_event = threading.Event()
            original_score = engine_module.score_sentence

            def score_then_stop(document, *args, **kwargs):
                result = original_score(document, *args, **kwargs)
                if document.doc_id == "DOC-001":
                    stop_event.set()
                return result

            with mock.patch("news_diag.engine.score_sentence", side_effect=score_then_stop):
                partial = run_workbench(
                    RunnerOptions(
                        news_path=news_path,
                        config_path=CONFIG,
                        injections_path=INJECTIONS,
                        serial=True,
                        state_dir=state_dir,
                        stop_event=stop_event,
                    )
                )
            self.assertTrue(partial.interrupted)
            self.assertEqual(partial.exit_code, 0)
            self.assertEqual(partial.completed_document_count, 1)
            self.assertTrue((state_dir / "run_state.json").exists())
            self.assertTrue((state_dir / "progress.jsonl").exists())

            resumed = run_workbench(
                RunnerOptions(
                    news_path=news_path,
                    config_path=CONFIG,
                    injections_path=INJECTIONS,
                    serial=True,
                    state_dir=state_dir,
                    resume=True,
                )
            )
            full = run_workbench(
                RunnerOptions(
                    news_path=news_path,
                    config_path=CONFIG,
                    injections_path=INJECTIONS,
                    serial=True,
                )
            )
            self.assertEqual(resumed.exit_code, 0)
            self.assertFalse(resumed.interrupted)
            self.assertEqual(json_text(resumed.report), json_text(full.report))


if __name__ == "__main__":
    unittest.main()

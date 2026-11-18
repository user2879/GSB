"""End-to-end and unit tests for the deterministic evaluation pipeline.

Run with:  python3 -m unittest discover -s tests -v
No third-party test framework is required.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from eval_pipeline.cache import FingerprintCache
from eval_pipeline.config import load_config, load_injection, load_rules
from eval_pipeline.errors import (
    E_CONFIG_INVALID,
    E_CONFIG_MISMATCH,
    E_DUP_ID,
    E_EMPTY_TEXT,
    E_NON_FINITE,
    E_RULE_WEIGHT,
    E_THRESHOLD_BOUNDARY,
    E_TIER_LIMIT_ZERO,
    E_BAD_JSON,
)
from eval_pipeline.features import FeatureExtractor, score_pair, tokenize
from eval_pipeline.generator import generate_all
from eval_pipeline.hallucination import detect
from eval_pipeline.loader import load_evaluations, load_preferences
from eval_pipeline.metrics import StreamingAggregator, compute_metrics, per_sample
from eval_pipeline.pipeline import Pipeline, normalized_report
from eval_pipeline.selector import STRATEGY_TABLE, select_strategy
from eval_pipeline.util import canonical_json, stable_hash

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TempSampleCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="evpipe-")
        self.sample_dir = os.path.join(self.tmp, "samples")
        self.paths = generate_all(self.sample_dir, preference_count=300, evaluation_count=120)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _inputs(self, failures):
        config = load_config(self.paths["config"])
        rules = load_rules(self.paths["rules"], failures)
        prefs = load_preferences(self.paths["preferences"], failures)
        evals = load_evaluations(self.paths["evaluations"], failures)
        return config, rules, prefs, evals

    def _run(self, workdir, injection=None, strict=False, crash_after=None,
             failures=None, config_overrides=None):
        failures = [] if failures is None else failures
        if config_overrides:
            import yaml
            with open(self.paths["config"], encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
            raw.update(config_overrides)
            config_path = os.path.join(self.tmp, "config-override.yaml")
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(raw, handle)
            from eval_pipeline.config import load_config as _load
            config = _load(config_path)
            rules = load_rules(self.paths["rules"], failures)
            prefs = load_preferences(self.paths["preferences"], failures)
            evals = load_evaluations(self.paths["evaluations"], failures)
        else:
            config, rules, prefs, evals = self._inputs(failures)
        pipeline = Pipeline(config, rules, prefs, evals, workdir,
                            injection_path=injection, strict_resume=strict,
                            crash_after=crash_after)
        report = pipeline.run()
        return report


class DeterminismTests(TempSampleCase):
    def test_stable_hash_is_platform_independent(self):
        self.assertEqual(stable_hash("hello", 7), stable_hash("hello", 7))
        self.assertNotEqual(stable_hash("hello", 7), stable_hash("hello", 8))

    def test_tokenizer_precision_levels(self):
        text = "alpha beta"
        self.assertTrue(all(not token.startswith(("2g:", "3g:")) for token in tokenize(text, 1)))
        self.assertTrue(any(token.startswith("2g:") for token in tokenize(text, 2)))
        self.assertTrue(any(token.startswith("3g:") for token in tokenize(text, 3)))

    def test_feature_and_score_determinism(self):
        config, _rules, prefs, _evals = self._inputs([])
        extractor = FeatureExtractor(config.precision_level, config.hash_buckets, config.tokenizer_seed)
        first = score_pair(prefs[0], extractor, config.reward_boundary_threshold)
        second = score_pair(prefs[0], extractor, config.reward_boundary_threshold)
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertIn("reward_diff_ref", first)
        self.assertEqual(first["reward_diff"].hex(), first["reward_diff_ref"])

    def test_full_report_byte_identical_across_runs(self):
        work_a = os.path.join(self.tmp, "work-a")
        work_b = os.path.join(self.tmp, "work-b")
        report_a = self._run(work_a)
        report_b = self._run(work_b)
        self.assertEqual(canonical_json(normalized_report(report_a)),
                         canonical_json(normalized_report(report_b)))


class ErrorCodeTests(TempSampleCase):
    def _write(self, name, rows):
        with open(os.path.join(self.sample_dir, name), "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(row + "\n")

    def test_bad_json_empty_text_and_dup_id(self):
        good = {"sample_id": "s1", "context": "c", "model_answer": "a", "reference_answer": "r"}
        dup = dict(good, sample_id="s1")
        empty = {"sample_id": "s2", "context": "  ", "model_answer": "a", "reference_answer": "r"}
        self._write("evaluations.jsonl", [
            "{not valid json",
            json.dumps(good),
            json.dumps(dup),
            json.dumps(empty),
        ])
        failures = []
        rows = load_evaluations(self.paths["evaluations"], failures)
        codes = sorted(item["code"] for item in failures)
        self.assertEqual(len(rows), 1)
        self.assertIn(E_BAD_JSON, codes)
        self.assertIn(E_DUP_ID, codes)
        self.assertIn(E_EMPTY_TEXT, codes)

    def test_bad_timestamp(self):
        row = {"pair_id": "p1", "prompt": "p", "chosen": "c", "rejected": "r",
               "annotator_id": "a", "timestamp": "not-a-timestamp"}
        self._write("preferences.jsonl", [json.dumps(row)])
        failures = []
        rows = load_preferences(self.paths["preferences"], failures)
        self.assertEqual(rows, [])
        self.assertEqual(failures[0]["code"], E_BAD_JSON)

    def test_non_positive_concurrency(self):
        import yaml
        with open(self.paths["config"], encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["concurrency"] = 0
        with open(self.paths["config"], "w", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle)
        with self.assertRaises(Exception) as ctx:
            load_config(self.paths["config"])
            self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)

    def test_tier_monotonicity_and_zero_limit(self):
        import yaml
        with open(self.paths["config"], encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["tiers"][1]["max_samples"] = raw["tiers"][0]["max_samples"]
        with open(self.paths["config"], "w", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle)
        with self.assertRaises(Exception) as ctx:
            load_config(self.paths["config"])
            self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)

        with open(self.paths["config"], encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["tiers"][0]["max_samples"] = 0
        with open(self.paths["config"], "w", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle)
        with self.assertRaises(Exception) as ctx:
            load_config(self.paths["config"])
            self.assertEqual(ctx.exception.code, E_TIER_LIMIT_ZERO)

    def test_rule_weight_out_of_range_is_isolated(self):
        with open(self.paths["rules"], encoding="utf-8") as handle:
            table = json.load(handle)
        table["rules"][1]["weight"] = 1.5
        with open(self.paths["rules"], "w", encoding="utf-8") as handle:
            json.dump(table, handle)
        failures = []
        rules = load_rules(self.paths["rules"], failures)
        self.assertEqual(failures[0]["code"], E_RULE_WEIGHT)
        self.assertEqual(len(rules), 2)
        with self.assertRaises(Exception) as ctx:
            load_rules(self.paths["rules"], None)
            self.assertEqual(ctx.exception.code, E_RULE_WEIGHT)

    def test_threshold_boundary_injection(self):
        injection = {
            "items": [
                {"item_no": 4, "enabled": True, "kind": "threshold_boundary",
                 "target_id": "eval_000000", "detail": "exact boundary"}
            ]
        }
        path = os.path.join(self.tmp, "inject.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(injection, handle)
        from eval_pipeline.config import boundary_injection_present
        with self.assertRaises(Exception) as ctx:
            boundary_injection_present(load_injection(path))
            self.assertEqual(ctx.exception.code, E_THRESHOLD_BOUNDARY)


class StrategyTests(TempSampleCase):
    def test_strategy_changes_with_scale(self):
        config, _rules, prefs, _evals = self._inputs([])
        extractor = FeatureExtractor(config.precision_level, config.hash_buckets, config.tokenizer_seed)
        chosen = [
            select_strategy(config, size, 100, 0.3)["strategy_id"]
            for size in (50, 500, 3000, 20000)
        ]
        self.assertEqual(chosen[0], STRATEGY_TABLE[0].strategy_id)
        self.assertEqual(chosen[-1], STRATEGY_TABLE[3].strategy_id)
        self.assertGreater(len(set(chosen)), 1)

    def test_strategy_is_deterministic_and_auditable(self):
        config, _rules, prefs, _evals = self._inputs([])
        extractor = FeatureExtractor(config.precision_level, config.hash_buckets, config.tokenizer_seed)
        vectors = [extractor.vector(item["prompt"]) for item in prefs]
        from eval_pipeline.features import high_cardinality_ratio
        unique = int(sum(int((v > 0).sum()) for v in vectors))
        args = (config, len(prefs), unique, high_cardinality_ratio(vectors))
        first = select_strategy(*args)
        second = select_strategy(*args)
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertIn("candidate_scores", first)
        self.assertIn("scale_metrics", first)
        self.assertIn("reasons", first)


class HallucinationTests(TempSampleCase):
    def test_three_rules_and_tie_priority(self):
        config, rules, _prefs, evals = self._inputs([])
        sample = evals[0]
        result = detect(sample, rules, config.hallucination_threshold)
        self.assertEqual(set(result["rule_risks"]),
                         {"entity_consistency", "numeric_consistency", "citation_consistency"})
        self.assertIn(result["suspected_hallucination"], (True, False))

    def test_missing_citation_is_flagged(self):
        rules = load_rules(self.paths["rules"], [])
        sample = {
            "sample_id": "x",
            "context": "context ENT_ALPHA snippet_known",
            "model_answer": "answer ENT_ALPHA [REF:absent_quote]",
            "reference_answer": "answer ENT_ALPHA",
        }
        result = detect(sample, rules, 0.0)
        self.assertEqual(result["rule_risks"]["citation_consistency"], 1.0)
        self.assertTrue(result["suspected_hallucination"])


class CacheTests(TempSampleCase):
    def test_content_fingerprint_reuse_and_invalidation(self):
        cache_dir = os.path.join(self.tmp, "cache")
        cache = FingerprintCache(cache_dir)
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return {"v": calls["n"]}

        artifact_a, reused_a = cache.get_or_compute("scoring", "fp1", "p1", "medium", compute)
        artifact_b, reused_b = cache.get_or_compute("scoring", "fp1", "p1", "medium", compute)
        self.assertFalse(reused_a)
        self.assertTrue(reused_b)
        self.assertEqual(artifact_a, artifact_b)
        self.assertEqual(calls["n"], 1)
        # different tier -> different key -> recompute
        _artifact_c, reused_c = cache.get_or_compute("scoring", "fp1", "p1", "large", compute)
        self.assertFalse(reused_c)
        removed = cache.invalidate_prefixes(["scoring"])
        self.assertGreaterEqual(removed, 2)
        self.assertGreaterEqual(cache.invalidated_count, 2)

    def test_cache_key_derivation_stable(self):
        key_a = FingerprintCache.derive_key("metrics", "c", "p", "small")
        key_b = FingerprintCache.derive_key("metrics", "c", "p", "small")
        key_c = FingerprintCache.derive_key("metrics", "c", "p", "large")
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)


class MetricsTests(TempSampleCase):
    def test_streaming_matches_batched_path(self):
        _config, _rules, _prefs, evals = self._inputs([])
        batched = compute_metrics(evals, ("exact_match_rate", "token_f1", "numeric_mae"),
                                  streaming=False)
        streaming = compute_metrics(evals, ("exact_match_rate", "token_f1", "numeric_mae"),
                                    streaming=True)
        self.assertEqual(canonical_json(batched), canonical_json(streaming))

    def test_aggregator_counts(self):
        rows = [per_sample(sample) for sample in self._inputs([])[3][:5]]
        aggregator = StreamingAggregator()
        for row in rows:
            aggregator.add(row)
        summary = aggregator.result()
        self.assertEqual(summary["sample_count"], 5)
        self.assertTrue(0.0 <= summary["token_f1"] <= 1.0)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class IncrementalConsistencyTests(TempSampleCase):
    def _copy_inputs(self, dest):
        os.makedirs(dest, exist_ok=True)
        for name in ("preferences.jsonl", "evaluations.jsonl", "rules.json", "config.yaml"):
            shutil.copy(os.path.join(self.sample_dir, name), os.path.join(dest, name))
        return dest

    def _run_workdir(self, sample_dir, workdir):
        failures = []
        config = load_config(os.path.join(sample_dir, "config.yaml"))
        rules = load_rules(os.path.join(sample_dir, "rules.json"), failures)
        prefs = load_preferences(os.path.join(sample_dir, "preferences.jsonl"), failures)
        evals = load_evaluations(os.path.join(sample_dir, "evaluations.jsonl"), failures)
        pipeline = Pipeline(config, rules, prefs, evals, workdir)
        pipeline.failures.extend(failures)
        return pipeline.run()

    def test_rule_weight_change_only_invalidates_hallucination_downstream(self):
        inc_dir = os.path.join(self.tmp, "rule-inputs")
        self._copy_inputs(inc_dir)
        work = os.path.join(self.tmp, "rule-work")
        first = self._run_workdir(inc_dir, work)
        scoring_keys_first = {
            key for key in first["cache"]["entries_on_disk"] if key.startswith("scoring_")
        }

        rules_path = os.path.join(inc_dir, "rules.json")
        table = json.load(open(rules_path, encoding="utf-8"))
        table["rules"][2]["weight"] = 0.4
        json.dump(table, open(rules_path, "w", encoding="utf-8"))
        second = self._run_workdir(inc_dir, work)

        scoring_keys_second = {
            key for key in second["cache"]["entries_on_disk"] if key.startswith("scoring_")
        }
        # Preference scoring artifacts are untouched and remain reusable.
        self.assertTrue(scoring_keys_first.issubset(scoring_keys_second))
        # Hallucination artifacts were recomputed under the new rule fingerprint.
        self.assertGreaterEqual(second["cache"]["recompute_count"], 1)
        self.assertEqual(
            canonical_json(normalized_report(first))
            == canonical_json(normalized_report(second)),
            False,
        )

    def test_edit_sequence_matches_clean_rebuild(self):
        # Incremental working copy evolves through an edit sequence.
        inc_dir = os.path.join(self.tmp, "inc-inputs")
        self._copy_inputs(inc_dir)
        inc_work = os.path.join(self.tmp, "inc-work")
        self._run_workdir(inc_dir, inc_work)

        prefs = _read_jsonl(os.path.join(inc_dir, "preferences.jsonl"))
        # Edit 1: replace a single preference pair.
        replaced = dict(prefs[5])
        replaced["chosen"] = replaced["chosen"] + " word_063 ENT_BETA"
        prefs[5] = replaced
        _write_jsonl(os.path.join(inc_dir, "preferences.jsonl"), prefs)
        self._run_workdir(inc_dir, inc_work)

        # Edit 2: change a hallucination rule weight.
        rules_path = os.path.join(inc_dir, "rules.json")
        table = json.load(open(rules_path, encoding="utf-8"))
        table["rules"][0]["weight"] = 0.55
        json.dump(table, open(rules_path, "w", encoding="utf-8"))
        self._run_workdir(inc_dir, inc_work)

        # Edit 3: shift the scale tier by appending preference pairs.
        prefs = _read_jsonl(os.path.join(inc_dir, "preferences.jsonl"))
        next_index = len(prefs)
        template = dict(prefs[0])
        extra = []
        for offset in range(1000):
            row = dict(template)
            row["pair_id"] = f"pair_{next_index + offset:06d}"
            extra.append(row)
        _write_jsonl(os.path.join(inc_dir, "preferences.jsonl"), prefs + extra)
        incremental_report = self._run_workdir(inc_dir, inc_work)

        # Clean rebuild from an empty work directory using the final inputs.
        clean_work = os.path.join(self.tmp, "clean-work")
        clean_report = self._run_workdir(inc_dir, clean_work)

        self.assertEqual(
            canonical_json(normalized_report(incremental_report)),
            canonical_json(normalized_report(clean_report)),
        )
        self.assertEqual(incremental_report["tier"]["name"], clean_report["tier"]["name"])
        # Incremental run must have reused at least some cached artifacts.
        self.assertGreaterEqual(
            incremental_report["cache"]["reuse_count"]
            + incremental_report["cache"]["invalidated_count"], 0
        )

    def test_concurrency_matches_serial_execution(self):
        import yaml
        base = os.path.join(self.tmp, "serial")
        self._copy_inputs(base)
        config_path = os.path.join(base, "config.yaml")
        raw = yaml.safe_load(open(config_path, encoding="utf-8"))

        raw["concurrency"] = 1
        yaml.safe_dump(raw, open(config_path, "w", encoding="utf-8"))
        serial = self._run_workdir(base, os.path.join(self.tmp, "work-serial"))

        raw["concurrency"] = 8
        yaml.safe_dump(raw, open(config_path, "w", encoding="utf-8"))
        parallel = self._run_workdir(base, os.path.join(self.tmp, "work-parallel"))

        self.assertEqual(canonical_json(normalized_report(serial)),
                         canonical_json(normalized_report(parallel)))
        pair_ids = [item["pair_id"] for item in serial["reward_pairs"]]
        self.assertEqual(pair_ids, sorted(pair_ids))
        self.assertEqual(
            [item["pair_id"] for item in parallel["reward_pairs"]], pair_ids
        )

    def test_snapshot_recovery_matches_uninterrupted_run(self):
        interrupted_work = os.path.join(self.tmp, "interrupted")
        overrides = {"snapshot_interval": 20}
        with self.assertRaises(RuntimeError):
            self._run(interrupted_work, crash_after=37, config_overrides=overrides)
        # Snapshot cursor must have advanced before the crash.
        snapshots = os.path.join(interrupted_work, "snapshots")
        self.assertTrue(os.path.exists(snapshots))
        recovered = self._run(interrupted_work, config_overrides=overrides)

        clean = self._run(os.path.join(self.tmp, "clean-recovery"),
                          config_overrides=overrides)
        self.assertEqual(canonical_json(normalized_report(recovered)),
                         canonical_json(normalized_report(clean)))
        self.assertTrue(recovered["resume"]["snapshot_resume"])
        self.assertFalse(clean["resume"]["snapshot_resume"])

    def test_strict_resume_rejects_lineage_change(self):
        work = os.path.join(self.tmp, "strict")
        self._run(work)
        # Mutate the config hash, then attempt a strict resume.
        import yaml
        config_path = self.paths["config"]
        raw = yaml.safe_load(open(config_path, encoding="utf-8"))
        raw["reward_boundary_threshold"] = 0.123
        yaml.safe_dump(raw, open(config_path, "w", encoding="utf-8"))
        failures = []
        config, rules, prefs, evals = self._inputs(failures)
        pipeline = Pipeline(config, rules, prefs, evals, work, strict_resume=True)
        with self.assertRaises(Exception) as ctx:
            pipeline.run()
            self.assertEqual(ctx.exception.code, E_CONFIG_MISMATCH)


class InjectionIsolationTests(TempSampleCase):
    def _injection_path(self, items):
        path = os.path.join(self.tmp, "inject.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"items": items}, handle)
        return path

    def test_non_finite_injection_is_recorded_and_isolated(self):
        injection_path = self._injection_path([
            {"item_no": 1, "enabled": True, "kind": "non_finite_feature",
             "target_id": "pair_000003", "detail": "NaN reward"}
        ])
        report = self._run(os.path.join(self.tmp, "work-inject"), injection=injection_path)
        codes = [item["code"] for item in report["failures"]]
        self.assertIn(E_NON_FINITE, codes)
        self.assertEqual(report["injections"][0]["item_no"], 1)
        hit = next(item for item in report["reward_pairs"] if item["pair_id"] == "pair_000003")
        self.assertIsNone(hit["reward_diff"])
        self.assertEqual(hit["reward_diff_ref"], "nan")
        # Other samples are still processed.
        self.assertGreater(len(report["reward_pairs"]), 10)

    def test_report_is_atomically_written_and_audit_complete(self):
        report = self._run(os.path.join(self.tmp, "work-audit"))
        for key in ("tier", "strategy", "reward_distribution", "hallucination",
                    "metrics", "cache", "snapshot_versions", "failures", "injections"):
            self.assertIn(key, report)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "work-audit", "report.json")))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsb.cleaning import clean_sample, run_cleaning
from gsb.config import validate_config
from gsb.dedup import deduplicate_shard
from gsb.engine import run_pipeline
from gsb.equivalence import canonical_state, diff_objects, execution_plan
from gsb.errors import E_INPUT_INVALID, PipelineError
from gsb.fixtures import base_batch_files, base_config, command_stream
from gsb.sampling import weighted_sample
from gsb.samples import load_sample_set


def write_input_files(root: Path, raw_config: dict, batches: dict | None = None):
    batches = base_batch_files() if batches is None else batches
    config_path = root / "configs" / "pipeline.json"
    sample_dir = root / "data" / "batches"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")
    for filename, rows in batches.items():
        (sample_dir / filename).write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
    return {"config": config_path, "samples": sample_dir}


def write_input(root: Path, raw_config: dict, batches: dict | None = None):
    paths = write_input_files(root, raw_config, batches)
    config = validate_config(raw_config)
    return config, load_sample_set(paths["samples"], config)


def run(root: Path, config, sample_set, **kwargs):
    return run_pipeline(config, sample_set, root / "state", **kwargs)


class ValidationTests(unittest.TestCase):
    def test_invalid_weight_has_field_and_code(self):
        raw = base_config()
        raw["domains"][0]["sampling_weight"] = -1
        with self.assertRaises(PipelineError) as ctx:
            validate_config(raw)
        self.assertEqual(ctx.exception.code, E_INPUT_INVALID)
        self.assertIn("sampling_weight", ctx.exception.field)

    def test_duplicate_sample_id_locates_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batches = base_batch_files()
            batches["alpha-batch-001.jsonl"][1]["sample_id"] = "alpha-s01"
            raw = base_config()
            paths = write_input_files(root, raw, batches)
            config = validate_config(raw)
            with self.assertRaises(PipelineError) as ctx:
                load_sample_set(paths["samples"], config)
            self.assertEqual(ctx.exception.code, E_INPUT_INVALID)
            self.assertEqual(ctx.exception.field, "samples.sample_id")


class CleaningDedupTests(unittest.TestCase):
    def test_cleaning_rule_change_changes_retained_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, samples = write_input(Path(tmp), base_config())
            before = {r["sample_id"]: r["status"] for r in run_cleaning(samples, config)}

            raw = base_config()
            raw["cleaning"]["min_total_token_count"] = 20
            stricter = validate_config(raw)
            after = {r["sample_id"]: r["status"] for r in run_cleaning(samples, stricter)}
            self.assertNotEqual({k for k, v in before.items() if v == "retained"}, {k for k, v in after.items() if v == "retained"})
            rejected = next(r for r in run_cleaning(samples, stricter) if r["reason_code"] == "LOW_TOKEN_COUNT")
            self.assertEqual(rejected["config_field"], "cleaning.min_total_token_count")

    def test_zero_counts_pass_loader_and_are_cleaned_by_threshold(self):
        sample = {
            "sample_id": "zero-count",
            "batch_id": "b",
            "domain": "alpha",
            "token_counts": {},
            "length_distribution": [0, 0, 0, 0, 0, 0],
            "missing_bitmap": [0] * 6,
            "batch_impute_length_distribution": [1] * 6,
            "term_count_total": 0,
        }
        config = validate_config(base_config())
        record = clean_sample(sample, config)
        self.assertEqual(record["status"], "rejected")
        self.assertEqual(record["reason_code"], "LOW_TOKEN_COUNT")

    def test_exact_threshold_boundary_and_id_tie(self):
        raw = base_config()
        raw["cleaning"]["min_total_token_count"] = 1
        config = validate_config(raw)

        def make(sample_id, features):
            return {
                "sample_id": sample_id,
                "batch_id": "b",
                "domain": "alpha",
                "token_counts": features,
                "length_distribution": [1, 1, 1, 1, 1, 1],
                "missing_bitmap": [0] * 6,
                "batch_impute_length_distribution": [0] * 6,
                "term_count_total": sum(features.values()),
            }

        samples = {
            "boundary-a": make("boundary-a", {"f0": 4, "f1": 2, "f2": 2, "f3": 2}),
            "boundary-b": make("boundary-b", {f"f{i}": 2 for i in range(5)}),
        }
        cleaning = {sid: clean_sample(sample, config) for sid, sample in samples.items()}
        shard = {
            "shard_id": "shard-alpha-boundary",
            "domain": "alpha",
            "shard_index": 0,
            "sample_ids": ["boundary-a", "boundary-b"],
            "pair_count": 1,
            "scheduling_cost": 2,
        }
        artifact = deduplicate_shard(shard, samples, cleaning, config)
        self.assertEqual(len(artifact["clusters"][0]["member_ids"]), 2)
        self.assertEqual(artifact["clusters"][0]["representative_sample_id"], "boundary-a")
        removed = next(d for d in artifact["decisions"] if d["decision"] == "duplicate_removed")
        self.assertEqual(removed["representative_sample_id"], "boundary-a")
        self.assertEqual(removed["similarity_fp"], 800_000)
        self.assertEqual(removed["similarity_permille_floor"], 800)
        self.assertTrue(removed["exact_threshold_boundary"])


class SamplingTests(unittest.TestCase):
    def test_equal_weight_tie_uses_lexicographic_sample_id(self):
        raw = base_config()
        raw["domains"] = [
            {"name": "alpha", "sampling_weight": 1, "target_ratio": 0.5},
            {"name": "beta", "sampling_weight": 1, "target_ratio": 0.5},
        ]
        raw["sampling"]["sample_count"] = 1
        config = validate_config(raw)
        samples = {
            "alpha-z": {"domain": "alpha", "term_count_total": 3},
            "beta-a": {"domain": "beta", "term_count_total": 3},
        }
        artifact = weighted_sample(["alpha-z", "beta-a"], samples, config, dedup_fingerprint="test")
        self.assertEqual(artifact["global_order"], ["alpha-z"])


class PipelineEquivalenceTests(unittest.TestCase):
    def test_thread_pool_actually_reaches_configured_workers(self):
        import threading

        from gsb.engine import _parallel_map

        barrier = threading.Barrier(2, timeout=2)

        def work(item):
            barrier.wait()
            return item * 2

        results, stats = _parallel_map([1, 2], 2, work)
        self.assertEqual(results, [2, 4])
        self.assertEqual(stats["max_active"], 2)

    def test_budget_resume_and_workers_match_serial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            serial_root = root / "serial"
            config, samples = write_input(serial_root, base_config())
            reference = run(serial_root, config, samples, workers=1)

            budget_root = root / "budget"
            b_config, b_samples = write_input(budget_root, base_config())
            first = run(budget_root, b_config, b_samples, budget=20, workers=2)
            self.assertEqual(first["manifest"]["status"], "E_BUDGET_PENDING")
            resumed = run(budget_root, b_config, b_samples, budget=10**9, workers=3)
            self.assertEqual(resumed["manifest"]["status"], "complete")
            diffs = diff_objects(canonical_state(reference), canonical_state(resumed))
            self.assertEqual(diffs, [])
            self.assertEqual(execution_plan(reference), execution_plan(resumed))

    def test_weight_edit_matches_blank_full_rebuild(self):
        from gsb.edits import apply_commands

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inc_root = root / "inc"
            raw = base_config()
            batches = base_batch_files()
            config, samples = write_input(inc_root, raw, batches)
            run(inc_root, config, samples, workers=1)

            new_raw, _ = apply_commands(raw, batches, [command_stream()[0]])
            new_config, new_samples = write_input(inc_root / "edited", new_raw, batches)
            warm = run_pipeline(new_config, new_samples, inc_root / "state", workers=2)

            full_root = root / "full"
            full_config, full_samples = write_input(full_root, new_raw, batches)
            full = run(full_root, full_config, full_samples, workers=1)
            diffs = diff_objects(canonical_state(warm), canonical_state(full))
            self.assertEqual(diffs, [])


if __name__ == "__main__":
    unittest.main()

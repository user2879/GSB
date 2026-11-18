import json
import os
import shutil
import tempfile
import unittest

from ab_workbench.config import config_from_dict, load_config
from ab_workbench.estimation import estimate_pair
from ab_workbench.fixedpoint import FixedPoint
from ab_workbench.generator import generate
from ab_workbench.loader import load_injections
from ab_workbench.pipeline import Workbench
from ab_workbench.stats import GroupStat


def make_group(group, num, den, metric="ctr"):
    fp = FixedPoint(6)
    stat = GroupStat(group=group, exposures=den, requests=1, drop_reasons={})
    stat.metrics[metric] = {
        "metric": metric,
        "numerator": num,
        "denominator": den,
        "value_scaled": fp.ratio(num, den),
        "value_text": fp.to_text(fp.ratio(num, den)),
        "precision": fp.precision_label,
    }
    return stat


def config(**over):
    raw = {
        "experiment_id": "t",
        "groups": ["c", "t"],
        "control_group": "c",
        "group_weights": {"c": 1, "t": 1},
        "hash_seed": 1,
        "num_buckets": 10,
        "primary_metric": "ctr",
        "guardrail_metrics": ["cvr"],
        "metric_direction": "higher_is_better",
        "min_sample_size": 10,
        "min_lift": "0.01",
        "attribution_tolerance": "0.01",
        "decimal_width": 6,
        "quantization_step": None,
        "concurrency": 4,
        "batch_size": 8,
        "snapshot_every_batches": 1,
    }
    raw.update(over)
    return config_from_dict(raw)


class EstimationBoundaryTests(unittest.TestCase):
    def test_exactly_at_threshold_is_sufficient(self):
        cfg = config()
        fp = cfg.fixedpoint
        control = make_group("c", 2, 10)
        treatment = make_group("t", 4, 10)
        estimate = estimate_pair(control, treatment, "ctr", cfg, fp)
        self.assertTrue(estimate.sufficient)
        self.assertFalse(estimate.suppressed)
        self.assertIn("exactly_at_threshold", estimate.boundary)

    def test_below_threshold_suppresses_lift(self):
        cfg = config(min_sample_size=11)
        fp = cfg.fixedpoint
        control = make_group("c", 2, 10)
        treatment = make_group("t", 4, 10)
        estimate = estimate_pair(control, treatment, "ctr", cfg, fp)
        self.assertTrue(estimate.suppressed)
        self.assertIsNone(estimate.absolute_lift_scaled)
        self.assertEqual(estimate.decision, "insufficient_sample")


class PipelineBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.sample = os.path.join(self.work, "sample")
        generate(self.sample, num_users=40, requests_per_user=3, candidates_per_request=4)
        self.log = os.path.join(self.sample, "exposures.jsonl")
        self.attrs = os.path.join(self.sample, "user_attributes.json")
        self.cfg_path = os.path.join(self.sample, "experiment_config.json")

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def _bench(self, concurrent, tag, cfg=None):
        cfg = cfg or load_config(self.cfg_path)
        return Workbench(
            config=cfg,
            snapshot_dir=os.path.join(self.work, tag, "sn"),
            cache_dir=os.path.join(self.work, tag, "cache"),
            tmp_dir=os.path.join(self.work, tag, "tmp"),
            concurrent=concurrent,
        )

    def test_concurrent_matches_serial(self):
        injections = load_injections(os.path.join(self.sample, "injections.json"))
        concurrent = self._bench(True, "cc").run(self.log, self.attrs, injections).to_dict()
        serial = self._bench(False, "ss").run(self.log, self.attrs, injections).to_dict()
        for field in ("group_stats", "estimation", "attribution", "failures"):
            self.assertEqual(concurrent[field], serial[field])

    def test_deterministic_across_runs(self):
        first = self._bench(True, "a").run(self.log, self.attrs).to_dict()
        second = self._bench(True, "b").run(self.log, self.attrs).to_dict()
        self.assertEqual(first["group_stats"], second["group_stats"])
        self.assertEqual(first["estimation"], second["estimation"])

    def test_injection_codes_hit(self):
        injections = load_injections(os.path.join(self.sample, "injections.json"))
        report = self._bench(True, "inj").run(self.log, self.attrs, injections).to_dict()
        codes = {h["injection_id"]: h["code"] for h in report["injection_hits"]}
        self.assertEqual(codes.get("INJ-001"), "E_LOG_INVALID")
        self.assertEqual(codes.get("INJ-002"), "E_LOG_INVALID")
        self.assertEqual(codes.get("INJ-003"), "E_LOG_INVALID")
        summary = report["failure_summary"]
        self.assertEqual(summary.get("E_LOG_INVALID"), 3)

    def test_snapshot_resume_and_manifest(self):
        bench = self._bench(True, "resume")
        result = bench.run(self.log, self.attrs).to_dict()
        manifest = result["manifest"]
        self.assertEqual(manifest["completed_batches"], result["total_batches"])
        self.assertTrue(os.path.exists(os.path.join(self.work, "resume", "sn", "manifest.json")))
        # Re-run resumes and reuses every artifact.
        again = bench.run(self.log, self.attrs).to_dict()
        self.assertTrue(again["cache"]["reused_keys"])
        self.assertEqual(again["manifest"]["global_version"], manifest["global_version"] + 1)

    def test_attribution_drift_under_tight_tolerance(self):
        generated = load_config(self.cfg_path)
        tight = config(
            groups=[generated.control_group, *generated.treatment_groups],
            control_group=generated.control_group,
            group_weights=generated.group_weights,
            hash_seed=generated.hash_seed,
            num_buckets=generated.num_buckets,
            attribution_tolerance="0.000000001",
        )
        bench = Workbench(
            config=tight,
            snapshot_dir=os.path.join(self.work, "drift", "sn"),
            cache_dir=os.path.join(self.work, "drift", "cache"),
            tmp_dir=os.path.join(self.work, "drift", "tmp"),
            concurrent=True,
        )
        report = bench.run(self.log, self.attrs).to_dict()
        self.assertEqual(report["failure_summary"].get("E_ATTRIBUTION_DRIFT"), 2)
        for layer in report["attribution"]:
            for result in layer["results"]:
                self.assertTrue(result["correction"])
                self.assertEqual(result["discrepancy_scaled"], 0)

    def test_stale_snapshot_injection_emits_config_mismatch(self):
        from ab_workbench.edits import update_config

        bench = self._bench(True, "stale")
        bench.run(self.log, self.attrs)
        # Change bucket count so the persisted snapshot lineage becomes stale,
        # then explicitly request the stale-snapshot-reuse fault injection.
        update_config(self.cfg_path, {"num_buckets": 24})
        report = self._bench(True, "stale").run(
            self.log,
            self.attrs,
            injections=[
                {
                    "id": "INJ-004",
                    "fault": "stale_snapshot_reuse",
                    "stage": "snapshot",
                    "group": "global",
                }
            ],
        ).to_dict()
        hit_codes = {h["injection_id"]: h["code"] for h in report["injection_hits"]}
        self.assertEqual(hit_codes.get("INJ-004"), "E_CONFIG_MISMATCH")
        self.assertEqual(report["failure_summary"].get("E_CONFIG_MISMATCH"), 1)


if __name__ == "__main__":
    unittest.main()

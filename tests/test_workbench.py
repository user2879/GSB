"""Acceptance tests for the tabular clustering workbench.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tabcluster.pipeline import run  # noqa: E402
from tabcluster.clustering import kmeans_train, dbscan_train  # noqa: E402
import numpy as np  # noqa: E402


def base_description() -> dict:
    return {
        "seed": 20240501,
        "base_rows": 120,
        "train_fraction": 0.8,
        "scales": {"small": 1, "medium": 4},
        "columns": [
            {"name": "age", "type": "numeric", "distribution": "normal",
             "mean": 42.0, "std": 12.0, "missing_rate": 0.05},
            {"name": "income", "type": "numeric", "distribution": "normal",
             "mean": 8000.0, "std": 2500.0, "missing_rate": 0.03},
            {"name": "segment", "type": "categorical",
             "categories": ["basic", "plus", "premium"],
             "weights": [0.5, 0.3, 0.2], "missing_rate": 0.04},
            {"name": "device_id", "type": "high_cardinality",
             "cardinality": 90, "missing_rate": 0.02},
            {"name": "age_copy", "type": "redundant", "source": "age"},
        ],
    }


def base_config(**overrides) -> dict:
    cfg = {
        "experiment_name": "test-exp",
        "seed": 20240501,
        "sample_scale": "small",
        "preprocess": {"method": "quantile"},
        "methods": {
            "kmeans": {"n_clusters": [2, 3]},
            "dbscan": {"eps": [0.2], "min_samples": [4]},
            "gmm": {"n_components": [2], "covariance_type": ["full", "diag"]},
        },
        "metrics": ["silhouette", "calinski_harabasz", "log_likelihood"],
        "budget": {"global_iterations": 100000, "candidate_quota": 1000},
        "concurrency": 4,
        "snapshot": {"interval": 2},
        "kmeans": {"max_iter": 60},
        "gmm": {"max_iter": 80, "tol": 1e-6, "reg_covar": 1e-6},
    }
    cfg.update(overrides)
    return cfg


class Workspace:
    def __init__(self, root: Path, desc=None, cfg=None, injections=None):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.desc_path = root / "desc.yaml"
        self.cfg_path = root / "cfg.yaml"
        self.inj_path = root / "inj.yaml"
        self.write_desc(desc if desc is not None else base_description())
        self.write_cfg(cfg if cfg is not None else base_config())
        self.write_inj(injections if injections is not None else {"injections": []})

    def write_desc(self, desc):
        self.desc_path.write_text(yaml.safe_dump(desc, allow_unicode=True))

    def write_cfg(self, cfg):
        self.cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))

    def write_inj(self, inj):
        self.inj_path.write_text(yaml.safe_dump(inj, allow_unicode=True))

    def run(self, workdir_name="work"):
        return run(str(self.cfg_path), str(self.desc_path), str(self.inj_path),
                   str(self.root / workdir_name))


def comparable(summary: dict, strip_execution: bool = False) -> dict:
    s = json.loads(json.dumps(summary))
    s["manifest_version"] = 0
    if strip_execution:
        # concurrency is an execution-level knob: it may differ between runs
        # whose *results* must still be field-identical.
        for k in ("config_hash", "diagnostics_fingerprint"):
            s.pop(k, None)
    return s


class TestWorkbench(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tabcluster-test-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # 1. determinism: two fresh runs are field-identical
    def test_determinism_two_runs(self):
        ws = Workspace(self.tmp / "a")
        s1 = ws.run("w1")
        s2 = ws.run("w2")
        self.assertEqual(comparable(s1), comparable(s2))

    # 2. concurrency invariance incl. budget-exhaustion timing
    def test_concurrency_invariance(self):
        summaries = []
        for conc in (1, 2, 8):
            ws = Workspace(self.tmp / f"c{conc}",
                           cfg=base_config(concurrency=conc))
            summaries.append(comparable(ws.run(), strip_execution=True))
        self.assertEqual(summaries[0], summaries[1])
        self.assertEqual(summaries[1], summaries[2])

    def test_concurrency_invariance_under_budget_pressure(self):
        summaries = []
        for conc in (1, 6):
            cfg = base_config(concurrency=conc)
            cfg["budget"] = {"global_iterations": 200, "candidate_quota": 1000}
            ws = Workspace(self.tmp / f"b{conc}", cfg=cfg)
            summaries.append(comparable(ws.run(), strip_execution=True))
        self.assertEqual(summaries[0], summaries[1])
        self.assertTrue(summaries[0]["budget"]["todo"])

    # 3. budget: pending candidates stay in todo, ledger is auditable
    def test_global_budget_pending(self):
        cfg = base_config()
        cfg["budget"] = {"global_iterations": 200, "candidate_quota": 1000}
        ws = Workspace(self.tmp / "bp", cfg=cfg)
        s = ws.run()
        codes = [f["code"] for f in s["failures"]]
        self.assertIn("E_ITER_BUDGET_PENDING", codes)
        self.assertTrue(s["budget"]["todo"])
        ledger = s["budget"]["ledger"]
        self.assertTrue(all(e["remaining"] >= 0 for e in ledger))
        deducted = sum(e["deducted"] for e in ledger)
        self.assertEqual(deducted, s["budget"]["consumed"])
        self.assertLessEqual(s["budget"]["consumed"], 200)
        # ledger sequence is stable-key ordered
        keys = [e["candidate"] for e in ledger]
        self.assertEqual(keys, sorted(keys))

    # 4. per-candidate quota
    def test_candidate_quota_exhausted(self):
        cfg = base_config()
        cfg["budget"] = {"global_iterations": 100000, "candidate_quota": 1}
        ws = Workspace(self.tmp / "q", cfg=cfg)
        s = ws.run()
        codes = [f["code"] for f in s["failures"]]
        self.assertIn("E_CAND_BUDGET_EXHAUSTED", codes)

    # 5. config validation: errors collected, valid items still parsed
    def test_config_validation_codes(self):
        desc = base_description()
        desc["columns"].append({"name": "bad", "type": "mystery"})
        desc["columns"].append({"name": "badref", "type": "redundant",
                                "source": "nope"})
        cfg = base_config()
        cfg["concurrency"] = -3
        cfg["budget"] = {"global_iterations": 0, "candidate_quota": 1000}
        cfg["methods"]["kmeans"]["n_clusters"] = [0, 2]
        cfg["methods"]["unknown_method"] = {"x": [1]}
        ws = Workspace(self.tmp / "bad", desc=desc, cfg=cfg)
        s = ws.run()
        codes = [f["code"] for f in s["failures"]]
        self.assertIn("E_CONFIG_INVALID", codes)
        self.assertIn("E_CLUSTER_ZERO", codes)
        self.assertIn("E_BUDGET_INVALID", codes)
        # valid candidates still executed
        ok = [c for c in s["candidates"] if c["status"] == "ok"]
        self.assertTrue(ok)

    # 6. exact cache invalidation
    def test_cache_reuse_and_exact_invalidation(self):
        ws = Workspace(self.tmp / "cache")
        s1 = ws.run()
        # second identical run: every ok candidate reused from cache
        s2 = ws.run()
        hits = {c["short_key"]: c["cache_hit"] for c in s2["candidates"]}
        self.assertTrue(all(hits.values()))
        strip = lambda cs: [{k: v for k, v in c.items() if k != "cache_hit"}
                            for c in cs]
        self.assertEqual(strip(comparable(s1)["candidates"]),
                         strip(comparable(s2)["candidates"]))

        # modify one gmm hyperparameter -> only gmm candidates recompute
        cfg = base_config()
        cfg["methods"]["gmm"]["n_components"] = [3]
        ws.write_cfg(cfg)
        s3 = ws.run()
        codes3 = [f["code"] for f in s3["failures"]]
        self.assertIn("E_SNAPSHOT_MISMATCH", codes3)
        for c in s3["candidates"]:
            if c["method"] == "gmm":
                self.assertFalse(c["cache_hit"], c["short_key"])
            elif c["status"] == "ok":
                self.assertTrue(c["cache_hit"], c["short_key"])

        # change preprocess method -> all candidates recompute
        cfg["preprocess"] = {"method": "standard"}
        ws.write_cfg(cfg)
        s4 = ws.run()
        for c in s4["candidates"]:
            if c["status"] == "ok":
                self.assertFalse(c["cache_hit"], c["short_key"])

        # replace a single sample -> preprocess fingerprint changes -> all recompute
        ws.write_cfg(base_config())
        ws.write_inj({"injections": [
            {"id": "r1", "type": "replace_sample", "row": 5,
             "column": "income", "value": 99999.0}]})
        s5 = ws.run()
        for c in s5["candidates"]:
            if c["status"] == "ok":
                self.assertFalse(c["cache_hit"], c["short_key"])

    # 7. snapshot mismatch rejects old products
    def test_snapshot_mismatch(self):
        ws = Workspace(self.tmp / "snap")
        ws.run()
        cfg = base_config()
        cfg["metrics"] = ["silhouette"]
        ws.write_cfg(cfg)
        s = ws.run()
        codes = [f["code"] for f in s["failures"]]
        self.assertIn("E_SNAPSHOT_MISMATCH", codes)

    # 8. injected fault isolation
    def test_injected_fault_isolation(self):
        inj = {"injections": [
            {"id": "f1", "type": "candidate_fault", "method": "kmeans",
             "stage": "iterate", "code": "E_INJECTED_FAULT",
             "params": {"n_clusters": 2}}]}
        ws = Workspace(self.tmp / "inj", injections=inj)
        s = ws.run()
        faulted = [c for c in s["candidates"] if c["status"] == "fault"]
        self.assertEqual(len(faulted), 1)
        self.assertEqual(faulted[0]["method"], "kmeans")
        self.assertEqual(faulted[0]["error_stage"], "iterate")
        codes = [f["code"] for f in s["failures"]]
        self.assertIn("E_INJECTED_FAULT", codes)
        ok = [c for c in s["candidates"] if c["status"] == "ok"]
        self.assertTrue(ok)

    # 9. incremental (cached) execution equals full fresh execution
    def test_incremental_equals_full(self):
        ws = Workspace(self.tmp / "inc")
        ws.run()  # base: gmm n_components [2]
        cfg = base_config()
        cfg["methods"]["gmm"]["n_components"] = [2, 3]
        ws.write_cfg(cfg)
        s_inc = ws.run()  # incremental: k=2 cached, k=3 computed
        ws2 = Workspace(self.tmp / "full", cfg=cfg)
        s_full = ws2.run()
        inc = {c["short_key"]: c for c in comparable(s_inc)["candidates"]}
        full = {c["short_key"]: c for c in comparable(s_full)["candidates"]}
        self.assertEqual(set(inc), set(full))
        for key in inc:
            a, b = dict(inc[key]), dict(full[key])
            a.pop("cache_hit")
            b.pop("cache_hit")
            self.assertEqual(a, b, key)
        # labels must match too (from cluster stage artifacts)
        for key in inc:
            la = json.loads((self.tmp / "inc" / "work" / "snapshots" /
                             "cluster" / f"{key}.json").read_text())
            lb = json.loads((self.tmp / "full" / "work" / "snapshots" /
                             "cluster" / f"{key}.json").read_text())
            if la["output"] is not None:
                self.assertEqual(la["output"]["labels"],
                                 lb["output"]["labels"], key)

    # 10. algorithm-level edge cases
    def test_kmeans_empty_cluster_recorded(self):
        X = np.array([[0.0], [0.0], [0.0], [5.0], [5.0], [9.0]])
        out = kmeans_train(X, 4, max_iter=20, quota=100)
        events = [e["event"] for e in out["events"]]
        self.assertIn("empty_cluster", events)
        self.assertEqual(len(out["labels"]), 6)

    def test_dbscan_all_noise_recorded(self):
        X = np.array([[0.0], [10.0], [20.0], [30.0]])
        out = dbscan_train(X, eps=0.5, min_samples=2, quota=100)
        events = [e["event"] for e in out["events"]]
        self.assertIn("all_noise", events)
        self.assertTrue(all(l == -1 for l in out["labels"]))

    def test_quantile_tie_rule_metadata(self):
        ws = Workspace(self.tmp / "qt")
        s = ws.run()
        self.assertEqual(s["preprocess"]["tie_rule"],
                         "average_rank_linear_clip")
        # identical re-run -> identical preprocess fingerprint
        s2 = ws.run("work2")
        self.assertEqual(s["preprocess_fingerprint"],
                         s2["preprocess_fingerprint"])


if __name__ == "__main__":
    unittest.main()

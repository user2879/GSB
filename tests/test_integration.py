import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from tspworkbench.config import load_config
from tspworkbench.demo import _canonical_run
from tspworkbench.inject import load_injections
from tspworkbench.runner import run_pipeline
from tspworkbench.sample import default_specs, write_jsonl

CONFIG = """\
backtest:
  window_length: 24
  window_step: 12
  horizon: 6
models:
  - {name: naive, weight: 0.2, lag_order: 0}
  - {name: moving_average, weight: 0.3, lag_order: 5}
  - {name: autoregressive, weight: 0.5, lag_order: 3}
features:
  lag_orders: [1, 2, 3]
missing:
  strategy: forward_fill
quantization:
  frac_bits: 4
  gear: 1
exchange:
  version: v1
execution:
  max_workers: 3
  observation_budget: 100000
  series_quota: 100000
  snapshot_every_windows: 4
  temp_dir: .tmp
"""


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sample = self.tmp / "data"
        self.sample.mkdir()
        write_jsonl(self.sample / "samples.jsonl", default_specs(series_count=3, length=80))
        (self.sample / "config.yaml").write_text(CONFIG, encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fresh_then_incremental_reuses_all(self):
        cfg = load_config(self.sample / "config.yaml")
        fresh = run_pipeline(cfg, self.sample / "samples.jsonl", self.tmp / "w", fresh=True)
        inc = run_pipeline(cfg, self.sample / "samples.jsonl", self.tmp / "w", fresh=False)
        self.assertTrue(all(e.status == "reused" for e in inc.ledger.entries))
        self.assertEqual(_canonical_run(fresh, cfg)["windows"],
                         _canonical_run(inc, cfg)["windows"])

    def test_series_edit_only_invalidates_affected(self):
        cfg = load_config(self.sample / "config.yaml")
        run_pipeline(cfg, self.sample / "samples.jsonl", self.tmp / "w", fresh=True)
        # 修改 S001 的一个观测点
        rows = (self.sample / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        payload = [json.loads(r) for r in rows]
        for row in payload:
            if row["series_id"] == "S001" and row["timestamp_ms"] == 1000 and not row["missing"]:
                row["value"] = row["value"] + 5.0
        (self.sample / "samples.jsonl").write_text(
            "\n".join(json.dumps(r) for r in payload) + "\n", encoding="utf-8")
        inc = run_pipeline(cfg, self.sample / "samples.jsonl", self.tmp / "w", fresh=False)
        affected = {e.series_id for e in inc.ledger.entries if e.status == "recomputed"}
        self.assertEqual(affected, {"S001"})
        self.assertTrue(all(e.status == "reused" for e in inc.ledger.entries
                            if e.series_id != "S001"))

    def test_budget_exhaustion_code(self):
        raw = yaml.safe_load(CONFIG)
        raw["execution"]["series_quota"] = 2 * 24
        cfg_path = self.sample / "budget.yaml"
        cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        cfg = load_config(cfg_path)
        run = run_pipeline(cfg, self.sample / "samples.jsonl", self.tmp / "b", fresh=True)
        self.assertIn("E_SERIES_BUDGET_EXHAUSTED", {f["code"] for f in run.failures})

    def test_injection_codes_reported(self):
        injections = {"injections": [
            {"id": "J1", "kind": "irregular_timestamp", "series_id": "S000"},
            {"id": "J2", "kind": "non_finite_value", "series_id": "S001", "window_index": 3},
        ]}
        inj_path = self.sample / "inj.json"
        inj_path.write_text(json.dumps(injections), encoding="utf-8")
        cfg = load_config(self.sample / "config.yaml")
        run = run_pipeline(cfg, self.sample / "samples.jsonl", self.tmp / "i",
                           injections=load_injections(inj_path), fresh=True)
        hits = {f["injection_id"]: f["code"] for f in run.failures}
        self.assertEqual(hits.get("J1"), "E_TIMESTAMP_IRREGULAR")
        self.assertEqual(hits.get("J2"), "E_NON_FINITE")

    def test_parallel_equals_serial(self):
        base = yaml.safe_load(CONFIG)
        par, ser = yaml.safe_load(CONFIG), yaml.safe_load(CONFIG)
        par["execution"]["max_workers"] = 4
        ser["execution"]["max_workers"] = 1
        cp, cs = self.sample / "p.yaml", self.sample / "s.yaml"
        cp.write_text(yaml.safe_dump(par), encoding="utf-8")
        cs.write_text(yaml.safe_dump(ser), encoding="utf-8")
        r_par = run_pipeline(load_config(cp), self.sample / "samples.jsonl",
                             self.tmp / "par", fresh=True)
        r_ser = run_pipeline(load_config(cs), self.sample / "samples.jsonl",
                             self.tmp / "ser", fresh=True)
        self.assertEqual(_canonical_run(r_par, load_config(cp))["windows"],
                         _canonical_run(r_ser, load_config(cs))["windows"])
        self.assertEqual(
            [(f["series_id"], f["code"]) for f in r_par.failures],
            [(f["series_id"], f["code"]) for f in r_ser.failures])


if __name__ == "__main__":
    unittest.main()

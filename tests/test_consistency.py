import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mm_pipeline.hashing import canonical_bytes
from mm_pipeline.pipeline import execute_pipeline
from mm_pipeline.config import parse_config

from .fixtures import build_corpus, make_png

# Fields that must match field-by-field between incremental and full rebuild.
COMPARE_FIELDS = (
    "region_results",
    "boundary_samples",
    "ledger",
    "failures",
)


def _comparable(report: dict) -> dict:
    out = {}
    for field in COMPARE_FIELDS:
        out[field] = report[field]
    # Strip reuse flags, which legitimately differ between cached and fresh runs.
    for row in out["region_results"]:
        row.pop("reused", None)
    # E_SNAPSHOT_STALE is a recovery signal present only on resumed runs.
    out["failures"] = [
        f for f in out["failures"] if f["code"] != "E_SNAPSHOT_STALE"
    ]
    return out


def _assert_reports_equal(testcase: unittest.TestCase, a: dict, b: dict):
    ca, cb = _comparable(a), _comparable(b)
    testcase.assertEqual(
        json.loads(canonical_bytes(ca)),
        json.loads(canonical_bytes(cb)),
        "incremental and full-rebuild reports diverged",
    )


def _clone(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


class IncrementalVsFullTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="mm-inc-"))
        self.samples, self.config_path = build_corpus(self.base, count=3)
        self.work_inc = self.base / "work_inc"
        self.counter = 0

    def _fresh_work(self) -> Path:
        self.counter += 1
        return self.base / f"work_full_{self.counter}"

    def _assert_reports_equal(self, a, b):
        _assert_reports_equal(self, a, b)

    def _run_incremental(self):
        return execute_pipeline(
            self.samples, self.config_path, self.work_inc
        )

    def _run_full_rebuild(self):
        return execute_pipeline(
            self.samples, self.config_path, self._fresh_work(), force_reset=True
        )

    def _edit_assertion_file(self, sample_id: str, mutate) -> None:
        path = self.samples / sample_id / "assertions.json"
        doc = json.loads(path.read_text())
        mutate(doc)
        path.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")

    def _edit_relation_file(self, sample_id: str, mutate) -> None:
        path = self.samples / sample_id / "relations.json"
        doc = json.loads(path.read_text())
        mutate(doc)
        path.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")

    def test_full_edit_sequence_matches_clean_rebuild(self):
        # Baseline.
        baseline = self._run_incremental()
        self._assert_reports_equal(baseline, self._run_full_rebuild())

        # 1. Add a brand new sample.
        new_dir = self.samples / "sample_003"
        from .fixtures import write_sample
        write_sample(self.samples, "sample_003")
        inc = self._run_incremental()
        full = self._run_full_rebuild()
        self._assert_reports_equal( inc, full)
        # Existing sample artifacts were reused, only the new one recomputed.
        reused_samples = {item.split(":")[1] for item in inc["audit"]["reused"]}
        recomputed_samples = {item.split(":")[1] for item in inc["audit"]["recomputed"]}
        self.assertIn("sample_000", reused_samples)
        self.assertIn("sample_003", recomputed_samples)

        # 2. Replace an existing sample image.
        target = self.samples / "sample_001" / "image.png"
        target.write_bytes(make_png(seed=424242))
        inc = self._run_incremental()
        self._assert_reports_equal( inc, self._run_full_rebuild())
        affected = {item.split(":")[1] for item in inc["audit"]["recomputed"]}
        self.assertIn("sample_001", affected)
        self.assertNotIn("sample_002", affected)

        # 3. Add an assertion (and a relation referencing it).
        self._edit_assertion_file(
            "sample_000",
            lambda doc: doc["assertions"].append(
                {"assertion_id": "a9", "entity": "ent1",
                 "attribute": "hue_bias", "value": 0.123}
            ),
        )
        self._edit_relation_file(
            "sample_000",
            lambda doc: doc["relations"].append(
                {"relation_id": "r9", "relation_type": "link",
                 "assertion_id": "a9", "region": 3, "weight": 0.9}
            ),
        )
        inc = self._run_incremental()
        self._assert_reports_equal( inc, self._run_full_rebuild())

        # 4. Delete an assertion (and its relations), then modify a relation.
        self._edit_assertion_file(
            "sample_002",
            lambda doc: doc["assertions"].__delitem__(
                next(i for i, a in enumerate(doc["assertions"])
                     if a["assertion_id"] == "a1")
            ),
        )
        self._edit_relation_file(
            "sample_002",
            lambda doc: doc.update(
                relations=[
                    r for r in doc["relations"] if r["assertion_id"] != "a1"
                ]
            ),
        )
        self._edit_relation_file(
            "sample_000",
            lambda doc: doc["relations"][0].__setitem__("weight", 1.37),
        )
        inc = self._run_incremental()
        self._assert_reports_equal( inc, self._run_full_rebuild())
        # Only the two edited samples are recomputed at feature/fusion stages.
        recomputed = {item.split(":")[1] for item in inc["audit"]["recomputed"]}
        self.assertIn("sample_000", recomputed)
        self.assertIn("sample_002", recomputed)
        self.assertNotIn("sample_001", recomputed)

    def test_config_switches_match_rebuild_and_flag_stale(self):
        self._run_incremental()  # establish a snapshot

        # Switching quant level changes the config hash -> stale reset path.
        cfg = parse_config(self.config_path)
        quant_cfg = self.base / "cfg_quant.yaml"
        quant_cfg.write_text(
            __import__("yaml").safe_dump(
                {**cfg.to_dict(), "quant_level": 16, "frac_bits": 24},
                sort_keys=True,
            )
        )
        self.config_path = quant_cfg
        report = execute_pipeline(
            self.samples, quant_cfg, self.work_inc
        )
        self.assertEqual(report["stale_snapshot"]["code"], "E_SNAPSHOT_STALE")
        self._assert_reports_equal(
            report,
            execute_pipeline(self.samples, quant_cfg, self._fresh_work(),
                             force_reset=True),
        )

        # A threshold-only change is a decision parameter, not a feature
        # parameter: it must NOT invalidate the feature fingerprint, yet still
        # converge field-by-field to a clean rebuild.
        thr_cfg = self.base / "cfg_thr.yaml"
        thr_cfg.write_text(
            __import__("yaml").safe_dump(
                {**cfg.to_dict(), "quant_level": 16, "frac_bits": 24,
                 "confidence_threshold": 0.63},
                sort_keys=True,
            )
        )
        self.config_path = thr_cfg
        report = execute_pipeline(self.samples, thr_cfg, self.work_inc)
        self.assertNotIn("stale_snapshot", report)
        self._assert_reports_equal(
            report,
            execute_pipeline(self.samples, thr_cfg, self._fresh_work(),
                             force_reset=True),
        )


class ConcurrencyTests(unittest.TestCase):
    def test_serial_and_concurrent_identical(self):
        base = Path(tempfile.mkdtemp(prefix="mm-conc-"))
        samples, config_path = build_corpus(base, count=4)
        import yaml
        cfg = parse_config(config_path).to_dict()

        serial_path = base / "cfg_serial.yaml"
        serial_path.write_text(yaml.safe_dump({**cfg, "max_workers": 1}))
        parallel_path = base / "cfg_parallel.yaml"
        parallel_path.write_text(yaml.safe_dump({**cfg, "max_workers": 8}))

        serial = execute_pipeline(samples, serial_path, base / "w_serial",
                                  force_reset=True)
        parallel = execute_pipeline(samples, parallel_path, base / "w_parallel",
                                    force_reset=True)
        self.assertEqual(
            json.loads(canonical_bytes(_comparable(serial))),
            json.loads(canonical_bytes(_comparable(parallel))),
        )
        # Deterministic ordering: stable (sample, region) sequence in ledger.
        order = [(e["sample_id"], e["region"]) for e in serial["ledger"]]
        self.assertEqual(order, sorted(order))


if __name__ == "__main__":
    unittest.main()

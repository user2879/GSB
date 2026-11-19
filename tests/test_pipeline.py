"""端到端与单元测试：确定性、误差账本、复算一致、错误码、增量闭包。"""

from __future__ import annotations

import filecmp
import json
import tempfile
import unittest
from pathlib import Path

from leveldiag.continuous import solve_continuous
from leveldiag.demo import run_demo
from leveldiag.errors import ErrorCode, LevelDiagError
from leveldiag.generator import generate_from_dict, run_pipeline
from leveldiag.incremental import (changed_leaf_paths, declared_closure,
                                   incremental_regenerate)
from leveldiag.quantize import quantize
from leveldiag.recompute import recompute, verify_freshness
from leveldiag.samples import (boundary_spec, mutate, normal_spec,
                               overflow_spec, weight_critical_spec)
from leveldiag.spec import parse_spec, spec_from_dict
from leveldiag.util import canonical_dumps, classify, content_hash


class SpecValidationTests(unittest.TestCase):
    def test_missing_field_reports_path(self):
        bad = normal_spec()
        del bad["resources"]["density_cap"]
        with self.assertRaises(LevelDiagError) as ctx:
            spec_from_dict(bad)
        self.assertEqual(ctx.exception.code, ErrorCode.E_SPEC_INVALID)
        self.assertEqual(ctx.exception.details["field_path"],
                         "resources.density_cap")

    def test_weight_tolerance_exceeded(self):
        bad = mutate(normal_spec(), "terrain.weights.grass", 0.6 + 5e-9)
        with self.assertRaises(LevelDiagError) as ctx:
            spec_from_dict(bad)
        self.assertEqual(ctx.exception.code, ErrorCode.E_SPEC_INVALID)

    def test_weight_tolerance_critical_accepted(self):
        spec = spec_from_dict(weight_critical_spec())
        self.assertTrue(spec.spec_hash)

    def test_points_overlap_and_out_of_bounds(self):
        with self.assertRaises(LevelDiagError) as ctx:
            spec_from_dict(mutate(normal_spec(), "points.target", [1, 1]))
        self.assertEqual(ctx.exception.code, ErrorCode.E_SPEC_INVALID)
        with self.assertRaises(LevelDiagError) as ctx:
            spec_from_dict(mutate(normal_spec(), "points.spawn", [99, 0]))
        self.assertEqual(ctx.exception.code, ErrorCode.E_SPEC_INVALID)

    def test_region_out_of_bounds(self):
        bad = mutate(normal_spec(), "regions[1].cells",
                     [[x, y] for x in range(5, 11) for y in range(0, 8)])
        with self.assertRaises(LevelDiagError) as ctx:
            spec_from_dict(bad)
        self.assertEqual(ctx.exception.code, ErrorCode.E_SPEC_INVALID)

    def test_spec_hash_stable_and_yaml_roundtrip(self):
        import yaml
        spec = spec_from_dict(normal_spec())
        text = yaml.safe_dump(spec.raw, sort_keys=True)
        again = parse_spec(text)
        self.assertEqual(spec.spec_hash, again.spec_hash)


class DeterminismTests(unittest.TestCase):
    def test_continuous_hash_twice_identical(self):
        spec = spec_from_dict(normal_spec())
        a = solve_continuous(spec)
        b = solve_continuous(spec)
        self.assertEqual(a["continuous_hash"], b["continuous_hash"])
        self.assertEqual(canonical_dumps(a["region_shards"]),
                         canonical_dumps(b["region_shards"]))

    def test_byte_identical_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            d1 = Path(tmp) / "a"
            d2 = Path(tmp) / "b"
            generate_from_dict(normal_spec(), d1)
            generate_from_dict(normal_spec(), d2)
            files = {str(p.relative_to(d1)) for p in d1.rglob("*")
                     if p.is_file()}
            for rel in files:
                self.assertTrue(filecmp.cmp(d1 / rel, d2 / rel, shallow=False),
                                f"非逐字节一致: {rel}")

    def test_dict_order_independent(self):
        import yaml
        text = yaml.safe_dump(normal_spec(), sort_keys=True)
        reversed_text = yaml.safe_dump(
            dict(reversed(list(normal_spec().items()))), sort_keys=True)
        h1 = parse_spec(text).spec_hash
        h2 = parse_spec(reversed_text).spec_hash
        self.assertEqual(h1, h2)


class QuantizationTests(unittest.TestCase):
    def test_mapping_deterministic_and_sharded(self):
        spec = spec_from_dict(normal_spec())
        cont = solve_continuous(spec)
        m1 = quantize(spec, cont)
        m2 = quantize(spec, cont)
        self.assertEqual(m1["mapping_hash"], m2["mapping_hash"])
        self.assertEqual(set(m1["region_shards"]),
                         {r["id"] for r in spec.raw["regions"]})
        for shard in m1["region_shards"].values():
            for entry in shard["mappings"]:
                for field in ("elevation", "density"):
                    delta = entry[field]["delta"]
                    q = entry[field]["quantized"]
                    x = entry[field]["continuous"]
                    self.assertAlmostEqual(q - x, delta)
                    self.assertTrue(0.0 <= q <= 1.0)

    def test_step_change_only_affects_dependent_quantities(self):
        spec1 = spec_from_dict(normal_spec())
        spec2 = spec_from_dict(
            mutate(normal_spec(), "quantization.density_step", 0.25))
        c1 = solve_continuous(spec1)
        c2 = solve_continuous(spec2)
        self.assertEqual(c1["continuous_hash"], c2["continuous_hash"])
        m1 = quantize(spec1, c1)
        m2 = quantize(spec2, c2)
        self.assertNotEqual(m1["mapping_hash"], m2["mapping_hash"])
        # 高程量化规则未变，逐图块高程量化值一致
        for rid in m1["region_shards"]:
            for e1, e2 in zip(m1["region_shards"][rid]["mappings"],
                              m2["region_shards"][rid]["mappings"]):
                self.assertEqual(e1["elevation"]["quantized"],
                                 e2["elevation"]["quantized"])
                self.assertEqual(e1["elevation"]["delta"],
                                 e2["elevation"]["delta"])

    def test_quant_bounds_error(self):
        bad = spec_from_dict(
            mutate(normal_spec(), "quantization.elevation_step", 1.5))
        with self.assertRaises(LevelDiagError) as ctx:
            run_pipeline(bad)
        self.assertEqual(ctx.exception.code, ErrorCode.E_QUANT_BOUNDS)


class ConstraintAndLedgerTests(unittest.TestCase):
    def _entries(self, factory):
        spec = spec_from_dict(factory())
        result = run_pipeline(spec)
        return spec, result

    def test_normal_all_pass(self):
        _, result = self._entries(normal_spec)
        self.assertEqual(result["ledger"]["summary"]["fail"], 0)
        self.assertEqual(result["ledger"]["summary"]["uncertain"], 0)

    def test_overflow_fails_density(self):
        _, result = self._entries(overflow_spec)
        entry = next(e for e in result["ledger"]["entries"]
                     if e["kind"] == "resource_density_cap")
        self.assertEqual(entry["status"], "FAIL")

    def test_boundary_is_uncertain_not_silently_passed(self):
        _, result = self._entries(boundary_spec)
        entry = next(e for e in result["ledger"]["entries"]
                     if e["kind"] == "cell_density_max")
        self.assertEqual(entry["status"], "UNCERTAIN")
        self.assertTrue(entry["on_exact_boundary"])
        report = result["boundary_report"]
        self.assertTrue(any(
            item["constraint_id"] == entry["constraint_id"]
            for item in report["boundary_items"]))

    def test_classify_zero_band_math(self):
        self.assertEqual(classify("le", 1.0, 1.0, 0.0, 0.0), "PASS")
        self.assertEqual(classify("lt", 1.0, 1.0, 0.0, 0.0), "FAIL")
        self.assertEqual(classify("le", 1.0, 1.0, 0.0, 0.1), "UNCERTAIN")

    def test_ledger_records_error_chain(self):
        _, result = self._entries(normal_spec)
        entry = next(e for e in result["ledger"]["entries"]
                     if e["kind"] == "cell_density_max")
        self.assertIn("quant_error_bound", entry)
        self.assertIn("normalization_error", entry)
        self.assertIn("propagated_error_bound", entry)
        self.assertIn("truth_sources", entry)
        self.assertIn("delta", entry["evidence"])


class RecomputeTests(unittest.TestCase):
    def test_recompute_matches_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "lv"
            generate_from_dict(normal_spec(), out)
            result = recompute(out)
            self.assertEqual(result["mismatch_count"], 0)

    def test_stale_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "lv"
            generate_from_dict(normal_spec(), out)
            ledger = json.loads((out / "ledger.json").read_text())
            ledger["entries"][0]["status"] = "FAIL"
            (out / "ledger.json").write_text(json.dumps(ledger, sort_keys=True))
            with self.assertRaises(LevelDiagError) as ctx:
                verify_freshness(out)
            self.assertEqual(ctx.exception.code, ErrorCode.E_LEDGER_STALE)

    def test_mismatch_detection_reports_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "lv"
            manifest = generate_from_dict(normal_spec(), out)
            ledger_path = out / "ledger.json"
            ledger = json.loads(ledger_path.read_text())
            target = next(e for e in ledger["entries"]
                          if e["kind"] == "resource_density_cap")
            target["status"] = "FAIL"
            ledger["ledger_hash"] = content_hash({
                "spec_hash": ledger["spec_hash"],
                "continuous_hash": ledger["continuous_hash"],
                "mapping_hash": ledger["mapping_hash"],
                "grid_hash": ledger["grid_hash"],
                "entries": ledger["entries"]})
            ledger_path.write_bytes(canonical_dumps(ledger) + b"\n")
            manifest = json.loads((out / "manifest.json").read_text())
            manifest["ledger_hash"] = ledger["ledger_hash"]
            manifest["files"]["ledger.json"] = \
                __import__("leveldiag.util", fromlist=["file_hash"]).file_hash(
                    ledger_path)
            (out / "manifest.json").write_bytes(
                canonical_dumps(manifest) + b"\n")
            with self.assertRaises(LevelDiagError) as ctx:
                recompute(out)
            self.assertEqual(ctx.exception.code, ErrorCode.E_LEDGER_MISMATCH)
            self.assertEqual(
                ctx.exception.details["first_mismatch"]["constraint_id"],
                "resources.density_cap")


class IncrementalTests(unittest.TestCase):
    def test_closure_table(self):
        changes = changed_leaf_paths(
            normal_spec(), mutate(normal_spec(),
                                  "resources.density_cap", 0.12))
        closure = declared_closure(changes)
        self.assertEqual(closure["scope"], "partial")
        self.assertIn("ledger.json", closure["files"])
        self.assertNotIn("grid.json", closure["files"])

    def test_threshold_change_keeps_grid_and_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "inc"
            base = normal_spec()
            generate_from_dict(base, out)
            before = {str(p.relative_to(out)): __import__(
                "leveldiag.util", fromlist=["file_hash"]).file_hash(p)
                for p in out.rglob("*") if p.is_file()}
            report = incremental_regenerate(
                base, mutate(base, "resources.density_cap", 0.12), out)
            after = {str(p.relative_to(out)): __import__(
                "leveldiag.util", fromlist=["file_hash"]).file_hash(p)
                for p in out.rglob("*") if p.is_file()}
            for rel in before:
                if rel not in report["rewritten_files"]:
                    self.assertEqual(before[rel], after[rel], f"{rel} 被意外修改")
            self.assertEqual(report["rewritten_ledger_entries"],
                             ["resources.density_cap"])
            recompute(out)

    def test_step_change_keeps_continuous_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "inc"
            base = normal_spec()
            generate_from_dict(base, out)
            report = incremental_regenerate(
                base, mutate(base, "quantization.density_step", 0.25), out)
            rewritten = report["rewritten_files"]
            self.assertTrue(any(r.startswith("mapping/") for r in rewritten))
            self.assertIn("grid.json", rewritten)
            self.assertFalse(any(r.startswith("continuous/") for r in rewritten))
            recompute(out)

    def test_closure_mismatch_error(self):
        from leveldiag import incremental as inc_mod
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "inc"
            base = normal_spec()
            generate_from_dict(base, out)
            original = inc_mod.declared_closure
            inc_mod.declared_closure = lambda paths: {
                "scope": "partial",
                "files": ["boundary_report.json", "ledger.json",
                          "manifest.json", "spec.canonical.json",
                          "grid.json"],
                "ledger_entries": ["*"]}
            try:
                with self.assertRaises(LevelDiagError) as ctx:
                    incremental_regenerate(
                        base, mutate(base, "resources.density_cap", 0.12), out)
                self.assertEqual(ctx.exception.code,
                                 ErrorCode.E_CLOSURE_MISMATCH)
            finally:
                inc_mod.declared_closure = original


class DemoTests(unittest.TestCase):
    def test_full_demo_covers_all_error_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_demo(Path(tmp))
            self.assertEqual(
                set(report["error_codes"]),
                {ErrorCode.E_SPEC_INVALID, ErrorCode.E_QUANT_BOUNDS,
                 ErrorCode.E_LEDGER_STALE, ErrorCode.E_LEDGER_MISMATCH,
                 ErrorCode.E_BOUNDARY_AMBIGUOUS,
                 ErrorCode.E_CLOSURE_MISMATCH})
            for info in report["byte_identical"].values():
                self.assertTrue(info["identical"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

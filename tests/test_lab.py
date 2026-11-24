from __future__ import annotations

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from opexlab.commands import apply_command
from opexlab.config import default_state, load_config
from opexlab.conversion import conversion_artifact
from opexlab.engine import ExecutionEngine
from opexlab.errors import (
    E_BUDGET_TOO_SMALL,
    E_DUPLICATE_LAYER,
    E_FIELD_TRUNCATED,
    E_LAYER_QUOTA,
    E_NONFINITE,
    E_SCHEME_UNKNOWN,
    E_WIDTH_OUT_OF_RANGE,
    LabError,
)
from opexlab.fingerprints import parameter_set_fingerprint
from opexlab.fixed import FixedContext
from opexlab.injections import run_injections
from opexlab.loaders import (
    load_activation_objects,
    load_all_sources,
    load_weights_v1,
    load_weights_v2,
    read_source_bundle,
)
from opexlab.models import WeightRow
from opexlab.operators import layernorm_row, softmax_row
from opexlab.samplegen import build_sample_files
from opexlab.scheduler import build_schedule, execute_schedule
from opexlab.verification import run_verification


class LabTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory(prefix="opexlab-test-")
        cls.root = Path(cls._temp.name)
        build_sample_files(cls.root)
        cls.config = load_config(cls.root / "samples" / "config.json", cls.root)
        cls.bundle = read_source_bundle(cls.config)
        cls.loaded = load_all_sources(cls.config, cls.bundle)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def runtime(self, name: str) -> Path:
        path = self.root / "runtimes" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_loading_validation(self) -> None:
        self.assertEqual(sorted(self.loaded.layers), ["layer_alpha", "layer_beta", "layer_gamma"])
        self.assertEqual(len(self.loaded.weights_v1), 11)
        self.assertEqual(len(self.loaded.weights_v2), 11)
        self.assertEqual(self.loaded.failures, [])

        duplicate = copy.deepcopy([json.loads(line) for line in self.bundle.activation_lines])
        duplicate.append(copy.deepcopy(duplicate[0]))
        _layers, failures = load_activation_objects(duplicate, self.config)
        self.assertIn(E_DUPLICATE_LAYER, {item["code"] for item in failures})

        truncated = list(self.bundle.v1_lines)
        truncated[0] = truncated[0][:-1]
        _weights, failures = load_weights_v1(self.config, truncated)
        self.assertIn(E_FIELD_TRUNCATED, {item["code"] for item in failures})

        unknown = copy.deepcopy(self.bundle.v2_values)
        unknown[0]["scheme"] = "layernorm_magic"
        _weights, failures = load_weights_v2(self.config, unknown)
        self.assertIn(E_SCHEME_UNKNOWN, {item["code"] for item in failures})

        bad_width = copy.deepcopy(self.bundle.v2_values)
        bad_width[0]["precision"] = "int7"
        _weights, failures = load_weights_v2(self.config, bad_width)
        self.assertIn("E_PRECISION_UNKNOWN", {item["code"] for item in failures})

        nonfinite = copy.deepcopy([json.loads(line) for line in self.bundle.activation_lines])
        nonfinite[0]["activation"][1][2] = float("nan")
        _layers, failures = load_activation_objects(nonfinite, self.config)
        self.assertIn(E_NONFINITE, {item["code"] for item in failures})

        bad_v1_width = list(self.bundle.v1_lines)
        bad_v1_width[0] = bad_v1_width[0][:-3] + "  7"
        _weights, failures = load_weights_v1(self.config, bad_v1_width)
        self.assertIn(E_WIDTH_OUT_OF_RANGE, {item["code"] for item in failures})

    def test_conversion_roundtrip_and_sidecar(self) -> None:
        for key in sorted(self.loaded.weights_v1):
            original = self.loaded.weights_v1[key]
            up = conversion_artifact(self.config, original, "v2")["converted_row"]
            down = conversion_artifact(self.config, up, "v1")["converted_row"]
            self.assertEqual(down.raw["scale_permille"], original.raw["scale_permille"])
            self.assertEqual(down.raw["bias_code"], original.raw["bias_code"])
            self.assertEqual(down.raw["quant_bits"], original.raw["quant_bits"])
        lossy = WeightRow(
            layer="layer_alpha",
            row=0,
            scale=1.234567,
            bias=-0.00789,
            scheme="layernorm_scale_only",
            precision="int16",
            quant_bits=16,
            source_version="v2",
        )
        artifact = conversion_artifact(self.config, lossy, "v1")
        sidecar = artifact["sidecar"]
        self.assertNotEqual(sidecar["scale_residual"], 0)
        self.assertEqual(sidecar["dropped_scheme"], "layernorm_scale_only")
        self.assertTrue(sidecar["sidecar_hash"])
        without_sidecar = parameter_set_fingerprint(lossy, None)
        with_sidecar = parameter_set_fingerprint(lossy, sidecar["sidecar_hash"])
        self.assertNotEqual(without_sidecar, with_sidecar)
        too_long = WeightRow(
            layer="layer_name_too_long",
            row=0,
            scale=1.0,
            bias=0.0,
            scheme="layernorm_affine",
            precision="int16",
            quant_bits=16,
            source_version="v2",
        )
        with self.assertRaises(LabError) as context:
            conversion_artifact(self.config, too_long, "v1")
        self.assertEqual(context.exception.code, E_FIELD_TRUNCATED)

    def test_softmax_stable_fixed_and_quantized(self) -> None:
        context = FixedContext(fractional_bits=12, quant_level=1)
        values = [1000.0, 1000.0, 999.0, -500.0, 0.25]
        first = softmax_row(
            values,
            context,
            polynomial_order=5,
            approximation_name="taylor",
        )
        second = softmax_row(
            values,
            context,
            polynomial_order=5,
            approximation_name="taylor",
        )
        self.assertEqual(first, second)
        self.assertEqual(sum(first["output_fixed"]), context.scale)
        self.assertEqual(first["row_max_tie_count"], 2)
        self.assertEqual(first["row_max_index"], 0)
        self.assertEqual(first["metadata"]["accumulation_order"], "column_ascending")
        q2 = softmax_row(
            values,
            FixedContext(fractional_bits=12, quant_level=2),
            polynomial_order=6,
            approximation_name="taylor",
        )
        self.assertNotEqual(first["output_fixed"], q2["output_fixed"])

    def test_layernorm_floor_and_v1_v2_equivalence(self) -> None:
        context = FixedContext(fractional_bits=12, quant_level=1)
        v1_row = WeightRow(
            layer="x",
            row=0,
            scale=0.95,
            bias=0.02,
            scheme="layernorm_affine",
            precision="int16",
            quant_bits=16,
            source_version="v1",
            raw={"bias_code": 20},
        )
        v2_row = WeightRow(
            layer="x",
            row=0,
            scale=0.95,
            bias=0.02,
            scheme="layernorm_affine",
            precision="int16",
            quant_bits=16,
            source_version="v2",
        )
        values = [0.1, -0.2, 0.5, 0.0]
        v1 = layernorm_row(
            values,
            v1_row,
            context,
            variance_floor=1e-6,
            path_version="v1",
            bias_enabled=True,
        )
        v2 = layernorm_row(
            values,
            v2_row,
            context,
            variance_floor=1e-6,
            path_version="v2",
            bias_enabled=True,
        )
        self.assertEqual(v1["output_fixed"], v2["output_fixed"])
        self.assertEqual(v1["stats"]["sum_fixed"], v2["stats"]["sum_fixed"])

        constant = layernorm_row(
            [0.5, 0.5, 0.5, 0.5],
            v2_row,
            context,
            variance_floor=1e-6,
            path_version="v2",
            bias_enabled=True,
        )
        self.assertTrue(constant["stats"]["variance_floor_applied"])
        self.assertGreater(constant["stats"]["std_fixed"], 0)
        self.assertTrue(all(isinstance(value, int) for value in constant["output_fixed"]))

    def test_incremental_reuse_does_not_recompute_unaffected_rows(self) -> None:
        engine = ExecutionEngine(self.config, runtime_dir=self.runtime("incremental-operators"))
        base = default_state(self.config)
        engine.save_state(base)
        engine.run(base, mode="incremental")
        edited = apply_command(
            self.config,
            base,
            {
                "type": "replace_activation_row",
                "layer": "layer_beta",
                "row": 1,
                "values": [1.1, -0.4, 0.2, 0.9, -1.3],
            },
        )
        engine.save_state(edited)

        softmax_calls = []
        stats_calls = []
        output_calls = []
        import opexlab.engine as engine_module

        original_softmax = engine_module.softmax_row
        original_stats = engine_module.layernorm_statistics
        original_output = engine_module.layernorm_apply_from_stats

        def softmax_wrapper(values, *args, **kwargs):
            softmax_calls.append(tuple(values))
            return original_softmax(values, *args, **kwargs)

        def stats_wrapper(values, *args, **kwargs):
            stats_calls.append(tuple(values))
            return original_stats(values, *args, **kwargs)

        def output_wrapper(statistics, weight, *args, **kwargs):
            output_calls.append((weight.layer, weight.row))
            return original_output(statistics, weight, *args, **kwargs)

        with mock.patch.object(engine_module, "softmax_row", softmax_wrapper), mock.patch.object(
            engine_module, "layernorm_statistics", stats_wrapper
        ), mock.patch.object(engine_module, "layernorm_apply_from_stats", output_wrapper):
            report = engine.run(edited, mode="incremental")
        self.assertEqual(len(softmax_calls), 1)
        self.assertEqual(len(stats_calls), 1)
        self.assertEqual(output_calls, [("layer_beta", 1)])
        self.assertEqual(
            report["cache_plan"]["softmax"]["recomputed_rows"],
            [{"layer": "layer_beta", "row": 1}],
        )
        self.assertEqual(
            report["cache_plan"]["layernorm_stats"]["recomputed_rows"],
            [{"layer": "layer_beta", "row": 1}],
        )
        self.assertEqual(len(report["cache_plan"]["version_conversion"]["reused_rows"]), 11)

    def test_layer_parameter_edit_keeps_statistics(self) -> None:
        engine = ExecutionEngine(self.config, runtime_dir=self.runtime("incremental-param"))
        base = default_state(self.config)
        engine.save_state(base)
        engine.run(base, mode="incremental")
        edited = apply_command(
            self.config,
            base,
            {
                "type": "replace_layer_params",
                "layer": "layer_beta",
                "scale": 1.35,
                "bias": 0.02,
                "scheme": "layernorm_affine",
                "precision": "int16",
            },
        )
        engine.save_state(edited)
        stats_calls = []
        output_calls = []
        import opexlab.engine as engine_module

        original_stats = engine_module.layernorm_statistics
        original_output = engine_module.layernorm_apply_from_stats

        def stats_wrapper(values, *args, **kwargs):
            stats_calls.append(values)
            return original_stats(values, *args, **kwargs)

        def output_wrapper(statistics, weight, *args, **kwargs):
            output_calls.append((weight.layer, weight.row))
            return original_output(statistics, weight, *args, **kwargs)

        with mock.patch.object(engine_module, "layernorm_statistics", stats_wrapper), mock.patch.object(
            engine_module, "layernorm_apply_from_stats", output_wrapper
        ):
            report = engine.run(edited, mode="incremental")
        self.assertEqual(stats_calls, [])
        self.assertEqual(
            output_calls,
            [("layer_beta", 0), ("layer_beta", 1), ("layer_beta", 2)],
        )
        self.assertEqual(report["cache_plan"]["softmax"]["recomputed_rows"], [])
        self.assertEqual(report["cache_plan"]["layernorm_stats"]["recomputed_rows"], [])

    def test_scheduler_budget_quota_and_parallel_bound(self) -> None:
        layers = [
            ("a", 3, [0, 1, 2, 3]),
            ("b", 2, [0, 1, 2, 3]),
        ]
        first = build_schedule(
            layers,
            batch_element_budget=10,
            per_layer_element_quota=6,
        )
        second = build_schedule(
            layers,
            batch_element_budget=10,
            per_layer_element_quota=6,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            [item["elements"] for item in first["budget_ledger"]],
            [3, 3, 2, 2],
        )
        self.assertIn(E_LAYER_QUOTA, {item["code"] for item in first["failures"]})

        small = build_schedule(
            [("a", 4, [0])],
            batch_element_budget=3,
            per_layer_element_quota=100,
        )
        self.assertIn(E_BUDGET_TOO_SMALL, {item["code"] for item in small["failures"]})

        tasks = [
            {"layer": "a", "row": row, "elements": 1}
            for row in range(8)
        ]

        def worker(task):
            time.sleep(0.02)
            return task["row"]

        results, metrics = execute_schedule(tasks, worker, max_workers=3)
        self.assertEqual(results, {("a", row): row for row in range(8)})
        self.assertLessEqual(metrics["peak_concurrent_tasks"], 3)

    def test_snapshot_manifest_and_injection_report(self) -> None:
        engine = ExecutionEngine(self.config, runtime_dir=self.runtime("snapshots"))
        state = default_state(self.config)
        engine.save_state(state)
        report = engine.run(state, mode="incremental")
        manifest = engine.store.load_manifest()
        self.assertIsNotNone(manifest)
        self.assertEqual(len(manifest["stage_versions"]), 5)
        self.assertEqual(
            set(manifest["stage_versions"]),
            {"load_validation", "version_conversion", "softmax", "layernorm", "summary"},
        )
        snapshot_path = engine.store.resolve_snapshot_path(manifest)
        for stage in manifest["stage_versions"]:
            self.assertTrue((snapshot_path / f"{stage}.json").exists())
        engine.run(state, mode="incremental")
        third = engine.run(state, mode="incremental")
        self.assertEqual(third["snapshot"]["commit_decision"], "unchanged_by_version_compare")

        injection = run_injections(
            self.config,
            report_path=self.runtime("snapshots") / "injection_report.json",
        )
        self.assertTrue(injection["all_matched"])
        self.assertEqual(len(injection["items"]), 5)

    def test_full_acceptance_suite(self) -> None:
        report_path = self.runtime("verification-report") / "verification.json"
        report = run_verification(self.config, report_path=report_path)
        self.assertTrue(report["all_passed"], report["summary"])
        self.assertEqual(report["summary"]["passed"], report["summary"]["total"])
        self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()

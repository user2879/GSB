import unittest

from fxpdiag.edits import apply_command, run_command_sequence
from fxpdiag.errors import ErrorCode, FailureList
from fxpdiag.fixedpoint import QuantSpec
from fxpdiag.models import (
    BIT_WIDTH_TO_SCHEME,
    EditCommand,
    OpConfig,
    TensorPackage,
    V1WeightRow,
    V2WeightRow,
)
from fxpdiag.operators import compute_row, float64_softmax
from fxpdiag.pipeline import PipelineRunner, PipelineState
from fxpdiag.scheduler import plan_batch, stable_jobs


def make_package():
    return TensorPackage(
        layers={
            "embedding": [[1.0, -1.0, 0.5], [0.2, 0.2, 0.2]],
            "attention": [[0.5, 0.0, -0.5], [1.0, 1.0, 1.0]],
        },
        depends_on={"embedding": [], "attention": ["embedding"]},
        seed=1,
    )


def make_config(**overrides):
    base = dict(
        frac_bits=6,
        tier="standard",
        rounding_mode="HALF_EVEN",
        max_workers=2,
        element_budget=12,
        layer_row_quota=2,
        tolerances={"abs": 0.05, "rel": 0.05},
        config_version=1,
    )
    base.update(overrides)
    return OpConfig(**base)


def make_weights(package):
    v1, v2 = {}, {}
    for layer, matrix in package.layers.items():
        for row_index in range(len(matrix)):
            v1[(layer, row_index)] = V1WeightRow(layer, row_index, 1000, 10, 16)
            v2[(layer, row_index)] = V2WeightRow(
                layer, row_index, 1.0,
                BIT_WIDTH_TO_SCHEME[16].value, "HALF_EVEN",
            )
    return v1, v2


def make_state(version="v1", **config_overrides):
    package = make_package()
    config = make_config(**config_overrides)
    v1, v2 = make_weights(package)
    state = PipelineState(package, config, version)
    state.weights.v1_rows = v1
    state.weights.v2_rows = v2
    return state


class OperatorTests(unittest.TestCase):
    def test_softmax_matches_float64_trend(self):
        row = [1.0, -1.0, 0.5]
        spec = QuantSpec(32, 10, "standard", "HALF_EVEN")
        result = compute_row(
            "l", 0, row, 1.0, 0.0, BIT_WIDTH_TO_SCHEME[32], spec, "v1"
        )
        truth = float64_softmax(row)
        for approx, exact in zip(result.softmax_float, truth):
            self.assertLess(abs(approx - exact), 0.02)
        self.assertEqual(result.softmax_argmax, 0)
        self.assertAlmostEqual(sum(result.softmax_float), 1.0, places=4)

    def test_bit_width_changes_integer_intermediates(self):
        row = [120.0, -45.5, 30.25]
        sm_out, ln_out, sat_by_width = {}, {}, {}
        for bit_width in (8, 16, 32):
            spec = QuantSpec(bit_width, 6, "standard", "HALF_EVEN")
            result = compute_row(
                "l", 0, row, 1.0, 0.0,
                BIT_WIDTH_TO_SCHEME[bit_width], spec, "v1",
            )
            sm_out[bit_width] = result.softmax_int
            ln_out[bit_width] = result.ln_int
            sat_by_width[bit_width] = sum(
                1
                for event in result.ln_events + result.softmax_events
                if event["direction"].startswith("SAT")
            )
        # 位宽真实改变整数中间值：Softmax 或 LayerNorm 至少一处逐档不同
        self.assertTrue(
            any(sm_out[a] != sm_out[b] for a, b in ((8, 16), (16, 32)))
            or any(ln_out[a] != ln_out[b] for a, b in ((8, 16), (16, 32)))
        )
        self.assertNotEqual(ln_out[8], ln_out[16])
        self.assertNotEqual(ln_out[16], ln_out[32])
        self.assertGreater(sat_by_width[8], sat_by_width[32],
                           "窄位宽应产生更多饱和事件")

    def test_zero_variance_degenerate_row(self):
        row = [1.0, 1.0, 1.0]
        spec = QuantSpec(16, 6, "standard", "HALF_EVEN")
        result = compute_row(
            "l", 1, row, 1.0, 0.2,
            BIT_WIDTH_TO_SCHEME[16], spec, "v1",
        )
        self.assertTrue(result.degenerate)
        self.assertEqual(result.ln_stats["rsqrt_float"], 1.0)

    def test_no_bias_scheme(self):
        from fxpdiag.models import Scheme
        row = [1.0, -1.0, 0.5]
        spec = QuantSpec(16, 6, "standard", "HALF_EVEN")
        result = compute_row(
            "l", 0, row, 1.0, 0.9,
            Scheme.Q8_8_NORM_NO_BIAS, spec, "v2",
        )
        self.assertFalse(result.bias_enabled)


class IncrementalTests(unittest.TestCase):
    def test_second_run_reuses_all_rows(self):
        state = make_state()
        runner = PipelineRunner(state)
        first = runner.execute(parallel=False)
        self.assertEqual(first["ledger_summary"]["recomputed"], 4)
        second = PipelineRunner(state).execute(parallel=False)
        self.assertEqual(second["ledger_summary"]["reused"], 4)
        self.assertEqual(second["ledger_summary"]["recomputed"], 0)
        self.assertEqual(first["rows"], second["rows"])

    def test_single_row_edit_only_invalidates_dependent(self):
        state = make_state()
        PipelineRunner(state).execute(parallel=False)
        command = EditCommand(
            1, "replace_row_activation",
            {"layer": "embedding", "row_index": 0,
             "activation": [2.0, -2.0, 0.25]},
        )
        affected = apply_command(state, command)
        # embedding 行0 + 下游 attention 全部行（跨层累计引用整个上游层）
        self.assertEqual(
            affected,
            {("embedding", 0), ("attention", 0), ("attention", 1)},
        )
        artifacts = PipelineRunner(state).execute(parallel=False)
        actions = {
            (item["layer"], item["row_index"]): item["action"]
            for item in artifacts["ledger"]
        }
        self.assertEqual(actions[("embedding", 0)], "RECOMPUTE")
        self.assertEqual(actions[("attention", 0)], "RECOMPUTE")
        self.assertEqual(actions[("attention", 1)], "RECOMPUTE")
        self.assertEqual(actions[("embedding", 1)], "REUSE")
        # 未受影响行必须给出复用依据
        reused = [item for item in artifacts["ledger"] if item["action"] == "REUSE"]
        self.assertTrue(all(item["reason"] == "fingerprint-hit" for item in reused))

    def test_full_equals_incremental_sequence(self):
        commands = [
            EditCommand(1, "switch_version", {"version": "v2"}),
            EditCommand(2, "set_frac_bits", {"frac_bits": 8}),
            EditCommand(3, "set_tier", {"tier": "fine"}),
            EditCommand(4, "switch_version", {"version": "v1"}),
        ]
        inc_state = make_state()
        incremental = run_command_sequence(
            inc_state, commands, PipelineRunner, parallel=False
        )
        for step, snapshot in enumerate(incremental):
            blank = make_state()
            rebuilt = run_command_sequence(
                blank, commands[:step], PipelineRunner, parallel=False
            )[-1]
            self.assertEqual(
                [(r["layer"], r["row_index"],
                  r["softmax"]["int"], r["layernorm"]["int"])
                 for r in snapshot["rows"]],
                [(r["layer"], r["row_index"],
                  r["softmax"]["int"], r["layernorm"]["int"])
                 for r in rebuilt["rows"]],
            )

    def test_parallel_equals_serial(self):
        commands = [EditCommand(1, "set_tier", {"tier": "fine"})]
        serial = run_command_sequence(
            make_state(), commands, PipelineRunner, parallel=False
        )[-1]
        parallel = run_command_sequence(
            make_state(), commands, PipelineRunner, parallel=True
        )[-1]
        self.assertEqual(
            [r["softmax"]["int"] for r in serial["rows"]],
            [r["softmax"]["int"] for r in parallel["rows"]],
        )
        self.assertEqual(serial["failures"], parallel["failures"])


class SchedulerTests(unittest.TestCase):
    def test_budget_pending_marker(self):
        jobs = stable_jobs(
            ["embedding", "attention"],
            {"embedding": 3, "attention": 2},
            {"embedding": 3, "attention": 3},
            {},
        )
        failures = FailureList()
        plan = plan_batch(jobs, element_budget=6, layer_row_quota=2,
                          failures=failures)
        self.assertEqual(len(plan.ready), 2)
        self.assertEqual(len(plan.pending), 3)
        self.assertTrue(all(
            code == ErrorCode.E_BUDGET_PENDING.value for code in failures.codes()
        ))

    def test_runner_completes_despite_tight_budget(self):
        state = make_state(element_budget=3, layer_row_quota=1)
        artifacts = PipelineRunner(state).execute(parallel=False)
        self.assertEqual(len(artifacts["rows"]), 4)
        self.assertTrue(any(
            item["code"] == ErrorCode.E_BUDGET_PENDING.value
            for item in artifacts["failures"]
        ))


class ValidationTests(unittest.TestCase):
    def test_cycle_detection(self):
        from fxpdiag.validation import validate_tensor_package
        package = TensorPackage(
            layers={"a": [[1.0]], "b": [[1.0]]},
            depends_on={"a": ["b"], "b": ["a"]},
        )
        failures = FailureList()
        self.assertFalse(validate_tensor_package(package, "mem", failures))
        self.assertIn(ErrorCode.E_LAYER_CYCLE.value, failures.codes())

    def test_unknown_reference(self):
        from fxpdiag.validation import validate_tensor_package
        package = TensorPackage(
            layers={"a": [[1.0]]}, depends_on={"a": ["ghost"]}
        )
        failures = FailureList()
        self.assertFalse(validate_tensor_package(package, "mem", failures))
        self.assertIn(ErrorCode.E_UNKNOWN_REF.value, failures.codes())


if __name__ == "__main__":
    unittest.main()

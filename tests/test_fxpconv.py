"""定点卷积与饱和截断算子组件单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fxpconv import (
    FailureLedger,
    InjectionEngine,
    PositionCache,
    RunConfig,
    compute_position,
    dequantize,
    load_calibration,
    load_descriptor,
    load_injections,
    load_tensor_pack,
    reference_position,
    run_pipeline,
    truncate,
)
from fxpconv.attribution import attribute_position
from fxpconv.cache import scale_fingerprint
from fxpconv.errors import (
    E_BATCH_BUDGET_EXHAUSTED,
    E_BUDGET_INVALID,
    E_DESC_INVALID,
    E_INJECT_DATA_OUT_OF_SHAPE,
    E_INJECT_ILLEGAL_STRATEGY,
    E_INJECT_NONFINITE_SCALE,
    E_INJECT_QBITS_OUT_OF_RANGE,
    E_SHAPE_MISMATCH,
    E_SNAPSHOT_STALE,
)
from fxpconv.kernel import positions_referencing_activation_cell
from fxpconv.models import OpDesc, int_range
from fxpconv.pipeline import _process_position
from fxpconv.scheduler import plan_batches
from fxpconv.tensorpack import Tensor, TensorPack, expected_main_shapes

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def base_descriptor(**overrides) -> dict:
    payload = json.loads((SAMPLES / "operator_desc.json").read_text(encoding="utf-8"))
    payload.update(overrides)
    return payload


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_desc(raw: dict) -> OpDesc:
    path = Path(tempfile.mkdtemp()) / "desc.json"
    write_json(path, raw)
    desc, ledger = load_descriptor(path)
    assert desc is not None, ledger.to_list()
    return desc


def make_pack(desc: OpDesc, activation_values, weight_values) -> TensorPack:
    act_shape = (desc.batch_n, desc.in_channels, desc.in_h, desc.in_w)
    weight_shape = (desc.out_channels, desc.in_channels, desc.k_h, desc.k_w)
    activation = Tensor("activation", act_shape, "int8", True, tuple(activation_values), ())
    weight = Tensor("weight", weight_shape, "int8", True, tuple(weight_values), ())
    return TensorPack(
        tensors={"activation": activation, "weight": weight},
        package_fingerprint="test-pack",
    )


class TinyFixture:
    """1x1x2x2 单通道 2x2 核、无填充无步长的确定性小夹具。"""

    def __init__(self, acc_bits=8, qbits=8, saturation="saturate", mac_budget=9,
                 concurrency=2, frac_bits=4, snapshot_every=1):
        self.raw = base_descriptor(
            batch_n=1, in_channels=1, in_h=2, in_w=2, out_channels=1,
            k_h=2, k_w=2, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
            acc_bits=acc_bits, qbits=qbits, saturation=saturation,
            rounding="round_nearest_even", frac_bits=frac_bits,
            out_low=-8.0, out_high=8.0, concurrency=concurrency,
            mac_budget=mac_budget, snapshot_every=snapshot_every,
            act_scale=0.5, weight_scales=[0.25],
            tolerance={"accum": {"abs": 0, "rel": 0},
                       "trunc": {"abs": 0, "rel": 0},
                       "deq": {"abs": 0, "rel": 0}},
        )
        self.desc = make_desc(self.raw)
        self.pack = make_pack(self.desc, [10, 10, 10, 10], [10, 10, 10, 10])


class TestLoadingValidation(unittest.TestCase):
    def test_valid_descriptor_and_pack(self):
        desc, dledger = load_descriptor(SAMPLES / "operator_desc.json")
        self.assertIsNotNone(dledger.__class__)
        self.assertEqual(len(dledger), 0)
        pack, pledger = load_tensor_pack(
            SAMPLES / "tensor_pack.jsonl", desc, expected_main_shapes(desc)
        )
        self.assertIsNotNone(pack)
        self.assertEqual(len(pledger), 0)
        calibration, cledger = load_calibration(SAMPLES / "calibration.json", desc)
        self.assertEqual(len(calibration), 3)
        self.assertEqual(len(cledger), 0)

    def test_invalid_enum_and_budget_codes(self):
        raw = base_descriptor(acc_bits=12, mac_budget=0, rounding="weird")
        path = Path(tempfile.mkdtemp()) / "bad.json"
        write_json(path, raw)
        desc, ledger = load_descriptor(path)
        self.assertIsNone(desc)
        codes = set(ledger.codes())
        self.assertIn(E_DESC_INVALID, codes)
        self.assertIn(E_BUDGET_INVALID, codes)

    def test_shape_mismatch(self):
        desc, _ = load_descriptor(SAMPLES / "operator_desc.json")
        bad_pack = Path(tempfile.mkdtemp()) / "pack.jsonl"
        act_shape = [desc.batch_n, desc.in_channels, desc.in_h, desc.in_w]
        weight_shape = [desc.out_channels, desc.in_channels, desc.k_h, desc.k_w]
        bad_pack.write_text(
            json.dumps({"name": "activation", "shape": act_shape,
                        "dtype": "int8", "signed": True,
                        "data": [0] * (desc.batch_n * desc.in_channels * desc.in_h * desc.in_w + 1)})
            + "\n",
            encoding="utf-8",
        )
        pack, ledger = load_tensor_pack(bad_pack, desc, expected_main_shapes(desc))
        self.assertIsNone(pack)
        self.assertIn(E_SHAPE_MISMATCH, ledger.codes())

    def test_illegal_wrap_combo_rejected(self):
        raw = base_descriptor(saturation="wrap", acc_bits=8, qbits=8)
        path = Path(tempfile.mkdtemp()) / "wrap.json"
        write_json(path, raw)
        desc, ledger = load_descriptor(path)
        self.assertIsNone(desc)
        self.assertIn(E_DESC_INVALID, ledger.codes())


class TestKernelDeterminism(unittest.TestCase):
    def test_fixed_accumulation_order_and_determinism(self):
        fixture = TinyFixture()
        first = compute_position(fixture.desc, fixture.pack, 0)
        second = compute_position(fixture.desc, fixture.pack, 0)
        self.assertEqual(first.raw_sum, 400)
        self.assertEqual(first.buffered, second.buffered)
        self.assertEqual(first.window_fingerprint, second.window_fingerprint)
        self.assertEqual(first.mac_count, 4)

    def test_overflow_first_coordinate_recorded(self):
        fixture = TinyFixture(acc_bits=8)
        result = compute_position(fixture.desc, fixture.pack, 0)
        self.assertIsNotNone(result.overflow)
        # 累加顺序 (oc,ic,kh,kw)，第一处溢出在坐标 (0,0,1)：100+100=200 > 127
        self.assertEqual(result.overflow.coord, (0, 0, 1))
        self.assertEqual(result.overflow.direction, "positive")
        self.assertEqual(result.overflow.partial_before, 100)
        self.assertEqual(result.buffered, 127)

    def test_wrap_overflow_changes_value(self):
        fixture = TinyFixture(acc_bits=16, qbits=16, saturation="wrap")
        result = compute_position(fixture.desc, fixture.pack, 0)
        self.assertIsNone(result.overflow)  # 16 位缓冲不溢出
        desc8 = make_desc(base_descriptor(
            batch_n=1, in_channels=1, in_h=2, in_w=2, out_channels=1,
            k_h=2, k_w=2, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
            acc_bits=8, qbits=8, saturation="saturate",
            weight_scales=[0.25],
        ))
        pack8 = make_pack(desc8, [10] * 4, [10] * 4)
        sat = compute_position(desc8, pack8, 0)
        self.assertEqual(sat.buffered, 127)


class TestSaturationAndRounding(unittest.TestCase):
    def test_saturate_vs_wrap_truncation(self):
        fixture = TinyFixture(acc_bits=16, qbits=8, saturation="saturate")
        position = compute_position(fixture.desc, fixture.pack, 0)
        sat = truncate(fixture.desc, position)
        self.assertEqual(sat.truncated, 127)
        self.assertTrue(sat.clamped)
        self.assertTrue(sat.boundary_sample)
        wrap = truncate(fixture.desc, position, strategy="wrap", qbits=8)
        self.assertEqual(wrap.truncated, -112)  # 400 的 int8 二补数回绕
        self.assertNotEqual(sat.truncated, wrap.truncated)

    def test_boundary_value_kept_in_range(self):
        fixture = TinyFixture()
        position = compute_position(fixture.desc, fixture.pack, 0)
        position.buffered = 127
        result = truncate(fixture.desc, position)
        self.assertEqual(result.truncated, 127)
        self.assertTrue(result.boundary_sample)
        self.assertFalse(result.clamped)

    def test_three_rounding_modes_and_step(self):
        fixture = TinyFixture(frac_bits=4)
        position = compute_position(fixture.desc, fixture.pack, 0)
        trunc = truncate(fixture.desc, position)
        values = {
            mode: dequantize(fixture.desc, trunc, 0, rounding=mode).value
            for mode in ("trunc_zero", "round_nearest_even", "floor")
        }
        self.assertEqual(dequantize(fixture.desc, trunc, 0).step, 2 ** -4)
        self.assertIsInstance(values["trunc_zero"], float)
        deq4 = dequantize(fixture.desc, trunc, 0, frac_bits=4)
        deq6 = dequantize(fixture.desc, trunc, 0, frac_bits=6)
        self.assertEqual(deq4.step, 2 ** -4)
        self.assertEqual(deq6.step, 2 ** -6)
        self.assertEqual(deq4.scale_source[:9], "act_scale"[:9])


class TestAttribution(unittest.TestCase):
    def test_first_fail_stage_unique_per_position(self):
        fixture = TinyFixture(acc_bits=16, qbits=8)
        position = compute_position(fixture.desc, fixture.pack, 0)
        trunc = truncate(fixture.desc, position)
        deq = dequantize(fixture.desc, trunc, 0)
        exact_sum, _, truth = reference_position(fixture.desc, fixture.pack, 0)
        attribution = attribute_position(
            fixture.desc, position, trunc, deq, exact_sum, truth
        )
        stages_in_fail = [
            stage for stage in ("accum", "trunc", "deq")
            if attribution.stage_errors[stage]["exceeds"]
        ]
        if attribution.first_fail_stage is not None:
            self.assertEqual(attribution.first_fail_stage, stages_in_fail[0])
        self.assertIn(attribution.cause, (
            "within_tolerance", "accum_overflow_saturation",
            "boundary_truncation", "dequant_rounding",
        ))

    def test_accum_overflow_cause(self):
        fixture = TinyFixture(acc_bits=8, qbits=8)
        position = compute_position(fixture.desc, fixture.pack, 0)
        trunc = truncate(fixture.desc, position)
        deq = dequantize(fixture.desc, trunc, 0)
        exact_sum, _, truth = reference_position(fixture.desc, fixture.pack, 0)
        attribution = attribute_position(
            fixture.desc, position, trunc, deq, exact_sum, truth
        )
        self.assertEqual(attribution.first_fail_stage, "accum")
        self.assertEqual(attribution.cause, "accum_overflow_saturation")


class TestScheduler(unittest.TestCase):
    def test_budget_split_and_exhaustion_code(self):
        ledger = FailureLedger()
        plan = plan_batches(total_positions=5, macs_per_position=18,
                            mac_budget=40, ledger=ledger)
        self.assertGreater(len(plan.batches), 1)
        first_batch = plan.batches[0]
        self.assertEqual(len({task.position for task in first_batch}), 2)
        self.assertTrue(all(task.sort_key[0] == 0 for task in first_batch))
        self.assertIn(E_BATCH_BUDGET_EXHAUSTED, ledger.codes())
        for batch in plan.budget_ledger:
            self.assertLessEqual(batch["allocated"], 40)

    def test_oversized_position_recorded(self):
        ledger = FailureLedger()
        plan = plan_batches(total_positions=2, macs_per_position=50,
                            mac_budget=20, ledger=ledger)
        self.assertIn(E_BATCH_BUDGET_EXHAUSTED, ledger.codes())
        computed = [task for batch in plan.batches for task in batch]
        self.assertEqual(computed, [])


class TestSnapshot(unittest.TestCase):
    def test_stale_snapshot_code(self):
        fixture = TinyFixture()
        snapshot_dir = Path(tempfile.mkdtemp()) / "snap"
        ledger = FailureLedger()
        engine = InjectionEngine([], fixture.desc, ledger)
        run_pipeline(fixture.desc, fixture.pack, ledger, engine,
                     RunConfig(snapshot_dir=str(snapshot_dir), run_id="a"))
        altered_raw = dict(fixture.raw)
        altered_raw["rounding"] = "floor"
        stale_desc = make_desc(altered_raw)
        stale_ledger = FailureLedger()
        stale_engine = InjectionEngine([], stale_desc, stale_ledger)
        run_pipeline(stale_desc, fixture.pack, stale_ledger, stale_engine,
                     RunConfig(snapshot_dir=str(snapshot_dir), run_id="b"))
        self.assertIn(E_SNAPSHOT_STALE, stale_ledger.codes())

    def test_resume_uses_snapshot(self):
        fixture = TinyFixture(snapshot_every=1)
        snapshot_dir = Path(tempfile.mkdtemp()) / "snap"
        ledger_a = FailureLedger()
        engine_a = InjectionEngine([], fixture.desc, ledger_a)
        first = run_pipeline(fixture.desc, fixture.pack, ledger_a, engine_a,
                             RunConfig(snapshot_dir=str(snapshot_dir), run_id="a"))
        ledger_b = FailureLedger()
        engine_b = InjectionEngine([], fixture.desc, ledger_b)
        second = run_pipeline(fixture.desc, fixture.pack, ledger_b, engine_b,
                              RunConfig(snapshot_dir=str(snapshot_dir), run_id="b"))
        self.assertTrue(second.resumed)
        self.assertEqual(
            sorted(first.positions.keys()), sorted(second.positions.keys())
        )


class TestInjectionIsolation(unittest.TestCase):
    def _load_engine(self, desc, ledger):
        injections = load_injections(SAMPLES / "injections.json", desc, ledger)
        return InjectionEngine(injections, desc, ledger)

    def test_all_four_business_codes_hit(self):
        desc, _ = load_descriptor(SAMPLES / "operator_desc.json")
        pack, _ = load_tensor_pack(
            SAMPLES / "tensor_pack.jsonl", desc, expected_main_shapes(desc)
        )
        ledger = FailureLedger()
        engine = self._load_engine(desc, ledger)
        snapshot_dir = Path(tempfile.mkdtemp()) / "snap"
        result = run_pipeline(
            desc, pack, ledger, engine,
            RunConfig(snapshot_dir=str(snapshot_dir), run_id="inject"),
        )
        codes = set(ledger.codes())
        for code in (E_INJECT_NONFINITE_SCALE, E_INJECT_QBITS_OUT_OF_RANGE,
                     E_INJECT_DATA_OUT_OF_SHAPE, E_INJECT_ILLEGAL_STRATEGY):
            self.assertIn(code, codes)
        # 单位置异常不中断其余位置。
        self.assertGreater(len(result.positions), 150)
        hit_kinds = {hit["kind"] for hit in result.injection_hits}
        self.assertEqual(len(hit_kinds), 4)

    def test_nonfinite_scale_isolates_only_target(self):
        fixture = TinyFixture()
        injections = [
            {"id": "X1", "kind": "nonfinite_scale", "target_position": 0,
             "target_tensor": "weight", "stage": "dequant",
             "params": {"value": float("inf")}},
        ]
        ledger = FailureLedger()
        engine = InjectionEngine(
            [type("I", (), {
                "injection_id": "X1", "kind": "nonfinite_scale",
                "target_position": 0, "target_tensor": "weight",
                "stage": "dequant", "params": {"value": float("inf")},
                "to_dict": lambda self: {},
            })()],
            fixture.desc, ledger,
        )
        cache = PositionCache()
        item = _process_position(
            fixture.desc, fixture.pack, 0, 0, cache, engine, ledger,
            scale_fingerprint(fixture.desc),
        )
        self.assertTrue(item["failed"])
        self.assertEqual(item["failure_code"], E_INJECT_NONFINITE_SCALE)
        self.assertIn(E_INJECT_NONFINITE_SCALE, ledger.codes())


class TestCacheInvalidation(unittest.TestCase):
    def test_input_replacement_only_recomputes_affected(self):
        fixture = TinyFixture()
        ledger = FailureLedger()
        engine = InjectionEngine([], fixture.desc, ledger)
        cache = PositionCache()
        scale_fp = scale_fingerprint(fixture.desc)
        for position in range(fixture.desc.total_positions):
            _process_position(
                fixture.desc, fixture.pack, position, 0, cache, engine,
                ledger, scale_fp,
            )
        cache.reused.clear()
        cache.recomputed.clear()
        for position in range(fixture.desc.total_positions):
            _process_position(
                fixture.desc, fixture.pack, position, 0, cache, engine,
                ledger, scale_fp,
            )
        self.assertEqual(cache.recomputed, [])

        activation = fixture.pack.get("activation")
        altered = list(activation.data)
        altered[0] = altered[0] + 3
        new_act = Tensor("activation", activation.shape, "int8", True,
                         tuple(altered), ())
        new_pack = TensorPack(
            tensors={"activation": new_act, "weight": fixture.pack.get("weight")},
            package_fingerprint="altered",
        )
        expected = positions_referencing_activation_cell(fixture.desc, 0, 0, 0, 0)
        cache.reused.clear()
        cache.recomputed.clear()
        for position in range(fixture.desc.total_positions):
            _process_position(
                fixture.desc, new_pack, position, 0, cache, engine,
                ledger, scale_fp,
            )
        self.assertEqual(set(cache.recomputed), expected)
        self.assertEqual(len(cache.reused), fixture.desc.total_positions - len(expected))

    def test_rounding_switch_partial_invalidation(self):
        desc, _ = load_descriptor(SAMPLES / "operator_desc.json")
        pack, _ = load_tensor_pack(
            SAMPLES / "tensor_pack.jsonl", desc, expected_main_shapes(desc)
        )
        ledger = FailureLedger()
        engine = InjectionEngine([], desc, ledger)
        cache = PositionCache()
        scale_fp = scale_fingerprint(desc)
        for position in range(desc.total_positions):
            _process_position(desc, pack, position,
                             position // desc.positions_per_sample,
                             cache, engine, ledger, scale_fp)
        floor_raw = dict(desc.raw)
        floor_raw["rounding"] = "floor"
        floor_desc = make_desc(floor_raw)
        cache.reused.clear()
        cache.recomputed.clear()
        for position in range(desc.total_positions):
            _process_position(floor_desc, pack, position,
                             position // floor_desc.positions_per_sample,
                             cache, engine, ledger, scale_fp)
        self.assertGreater(len(cache.reused), 0)
        self.assertGreater(len(cache.recomputed), 0)

    def test_int_range_enum(self):
        self.assertEqual(int_range(8), (-128, 127))
        self.assertEqual(int_range(16), (-32768, 32767))


if __name__ == "__main__":
    unittest.main(verbosity=2)

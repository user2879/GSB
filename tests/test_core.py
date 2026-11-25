import copy
import tempfile
import unittest
from pathlib import Path

from qos_diag.canonical import canonical_bytes
from qos_diag.diagnosis import diagnose, select_path
from qos_diag.engine import ChainEngine, StageState, evaluate_stage_transition, run_batched_event_path
from qos_diag.errors import SpecError
from qos_diag.fixedpoint import FixedPoint, classify_difference
from qos_diag.invalidation import derive_invalidation, recompute_incremental
from qos_diag.model import load_bundle, validate_pipelines, validate_shapers
from qos_diag.samplegen import build_deterministic_bundle_data, write_sample_dir


class CoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = build_deterministic_bundle_data(287)
        self.spec_dir = write_sample_dir(self.root / "spec", self.data)
        self.bundle = load_bundle(self.spec_dir)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_sample_and_diagnosis_parity(self) -> None:
        result = diagnose(self.bundle)
        self.assertTrue(result["path_parity"]["event_and_window_canonical_bytes_identical"])
        self.assertEqual(len(result["events"]), 36)
        self.assertIn(result["path_selection"]["selected_path"], {"event", "window"})
        self.assertIn("switch_boundary_advantage_gt", result["path_selection"]["estimate"])

    def test_arbitrary_batch_boundaries_are_byte_identical(self) -> None:
        full = diagnose(self.bundle, forced_path="event")["events"]
        full_bytes = canonical_bytes(full)
        for batch_size in (1, 2, 3, 5, 7, 12, 100):
            batched = run_batched_event_path(self.bundle, batch_size)
            self.assertEqual(canonical_bytes(batched), full_bytes, f"batch_size={batch_size}")

    def test_explicit_token_bucket_transitions(self) -> None:
        shaper = {
            "pipeline_id": "p",
            "stage": "ingress",
            "shaper_id": "s",
            "host": "127.0.0.1",
            "port": 1,
            "refill_rate_per_ms": 1,
            "burst_capacity_tokens": 5,
        }
        engine = ChainEngine(
            {"config": {"scale": 0, "slide_ms": 1}, "pipelines": [], "shapers": [], "events": []},
            FixedPoint(0),
        )
        state = StageState(5, 0)
        first, state = evaluate_stage_transition(
            shaper, 0, 2, state, FixedPoint(0)
        )
        self.assertEqual(first["verdict"], "allow")
        self.assertEqual(state.balance_q, 3)
        second, state = evaluate_stage_transition(
            shaper, 0, 4, state, FixedPoint(0)
        )
        self.assertEqual(second["verdict"], "shape")
        self.assertEqual(state.balance_q, 3)
        third, state = evaluate_stage_transition(
            shaper, 2, 4, state, FixedPoint(0)
        )
        self.assertEqual(third["verdict"], "allow")
        self.assertEqual(state.balance_q, 1)

    def test_spec_error_reports_first_bad_field(self) -> None:
        pipelines = validate_pipelines(self.data["pipelines_raw"])
        raw_shapers = self.data["shapers_raw"]["shapers"]
        raw_shapers[0]["host"] = "localhost"
        with self.assertRaises(SpecError) as context:
            validate_shapers(self.data["shapers_raw"], pipelines)
        self.assertEqual(context.exception.code, "E_SPEC")
        self.assertEqual(context.exception.field, "host")
        self.assertEqual(context.exception.object_id, raw_shapers[0]["shaper_id"])

    def test_duplicate_port_is_rejected(self) -> None:
        pipelines = validate_pipelines(self.data["pipelines_raw"])
        raw_shapers = self.data["shapers_raw"]["shapers"]
        raw_shapers[1]["port"] = raw_shapers[0]["port"]
        with self.assertRaises(SpecError) as context:
            validate_shapers(self.data["shapers_raw"], pipelines)
        self.assertEqual(context.exception.field, "port")

    def test_invalidation_for_all_edit_kinds(self) -> None:
        from qos_diag.samplegen import generate_all_samples

        paths = generate_all_samples(self.root / "samples")
        old = load_bundle(paths["base"])
        old_results = diagnose(old)["events"]
        for key in ("edited_param", "edited_arrival", "edited_burst"):
            new = load_bundle(paths[key])
            closure = derive_invalidation(old, new)
            incremental = recompute_incremental(old, new, old_results, closure)
            full = diagnose(new)["events"]
            self.assertEqual(canonical_bytes(incremental["results"]), canonical_bytes(full), key)
            self.assertTrue(incremental["unchanged_check"]["all_unaffected_byte_identical"], key)
            self.assertTrue(closure["affected_cells"], key)

    def test_difference_classification(self) -> None:
        self.assertEqual(classify_difference(100, 10, 1), "exceeds_tolerance")
        self.assertEqual(classify_difference(0, 10, 1), "within_tolerance")
        self.assertEqual(classify_difference(10, 10, 10), "indistinguishable")

    def test_chain_relation_edit(self) -> None:
        chain_data = copy.deepcopy(self.data)
        pipeline = next(item for item in chain_data["pipelines_raw"]["pipelines"] if item["pipeline_id"] == "p-01")
        ids = pipeline["stage_shaper_ids"]
        old_relay = ids["relay"]
        old_egress = ids["egress"]
        ids["relay"], ids["egress"] = old_egress, old_relay
        for shaper in chain_data["shapers_raw"]["shapers"]:
            if shaper["shaper_id"] == old_relay:
                shaper["stage"] = "egress"
            elif shaper["shaper_id"] == old_egress:
                shaper["stage"] = "relay"
        chain_dir = write_sample_dir(self.root / "chain", chain_data)
        new_bundle = load_bundle(chain_dir)
        old_results = diagnose(self.bundle)["events"]
        closure = derive_invalidation(self.bundle, new_bundle)
        incremental = recompute_incremental(self.bundle, new_bundle, old_results, closure)
        full = diagnose(new_bundle)["events"]
        self.assertTrue(incremental["unchanged_check"]["all_unaffected_byte_identical"])
        self.assertEqual(canonical_bytes(incremental["results"]), canonical_bytes(full))
        self.assertIn("relay", closure["restart_local_index"]["p-01"])
        self.assertEqual(closure["restart_local_index"]["p-01"]["relay"], 0)
        self.assertEqual(closure["restart_local_index"]["p-01"]["egress"], 0)

    def test_selection_contains_explicit_boundary(self) -> None:
        selection = select_path(self.bundle)
        estimate = selection["estimate"]
        self.assertIn("deadband_work", estimate)
        self.assertIn("required_advantage_at_least", estimate["boundary_value_if_not_switched"])


if __name__ == "__main__":
    unittest.main()

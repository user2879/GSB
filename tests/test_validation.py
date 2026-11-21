import json
import os
import tempfile
import unittest

from emergency_drill.errors import CycleError, PlanValidationError
from emergency_drill.loader import load_events, load_plan
from emergency_drill.sample_data import (
    bad_event_target_dict,
    cyclic_plan_dict,
    generate_samples,
    negative_duration_plan_dict,
    noncontinuous_order_plan_dict,
    valid_events_dict,
    valid_plan_dict,
)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        generate_samples(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def _dump(self, name, payload):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        return path

    def test_valid_bundle_loads(self):
        plan = load_plan(os.path.join(self.dir, "plan_valid.json"))
        events = load_events(os.path.join(self.dir, "events_valid.json"), plan)
        self.assertEqual(len(plan.stages), 5)
        self.assertEqual(len(events.events), 3)

    def test_corrupt_json_reports_first_location(self):
        with self.assertRaises(PlanValidationError) as caught:
            load_plan(os.path.join(self.dir, "plan_corrupt.json"))
        self.assertEqual(caught.exception.code, "E_PLAN")
        self.assertIn("line", caught.exception.location)

    def test_cycle_uses_e_cycle(self):
        with self.assertRaises(CycleError) as caught:
            load_plan(self._dump("cyc.json", cyclic_plan_dict()))
        self.assertEqual(caught.exception.code, "E_CYCLE")
        self.assertEqual(set(caught.exception.context["cycle"]), {"detect", "isolate", "failover",
                                                                   "verify", "recover"})

    def test_missing_reference_e_plan(self):
        path = self._dump("ref.json", {**valid_plan_dict()})
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                **valid_plan_dict(),
                "stages": [
                    stage if stage["stage_id"] != "isolate"
                    else {**stage, "prerequisites": ["detect", "ghost"]}
                    for stage in valid_plan_dict()["stages"]
                ],
            }, handle)
        with self.assertRaises(PlanValidationError) as caught:
            load_plan(path)
        self.assertEqual(caught.exception.code, "E_PLAN")
        self.assertIn("ghost", caught.exception.message)

    def test_order_gap_and_negative_duration(self):
        with self.assertRaises(PlanValidationError):
            load_plan(self._dump("gap.json", noncontinuous_order_plan_dict()))
        with self.assertRaises(PlanValidationError):
            load_plan(self._dump("neg.json", negative_duration_plan_dict()))

    def test_event_target_must_exist(self):
        plan = load_plan(os.path.join(self.dir, "plan_valid.json"))
        with self.assertRaises(PlanValidationError) as caught:
            load_events(self._dump("ev.json", bad_event_target_dict()), plan)
        self.assertEqual(caught.exception.code, "E_PLAN")

    def test_duplicate_stage_id_rejected(self):
        plan_dict = valid_plan_dict()
        plan_dict["stages"][1] = {**plan_dict["stages"][1], "stage_id": "detect"}
        with self.assertRaises(PlanValidationError) as caught:
            load_plan(self._dump("dup.json", plan_dict))
        self.assertIn("重复", caught.exception.message)


if __name__ == "__main__":
    unittest.main()

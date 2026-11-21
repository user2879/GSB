import tempfile
import unittest

from emergency_drill.editing import apply_changeset
from emergency_drill.errors import DependencyError, IncrementalMismatchError
from emergency_drill.engine import simulate, verify_recompute
from emergency_drill.loader import load_bundle
from emergency_drill.models import canonical_json
from emergency_drill.sample_data import generate_samples
import os


class EditingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        generate_samples(self.tmp.name)
        self.plan, self.events = load_bundle(
            os.path.join(self.tmp.name, "plan_valid.json"),
            os.path.join(self.tmp.name, "events_valid.json"),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_duration_change_invalidates_downstream_only(self):
        changeset = {"operations": [{"op": "update_duration", "stage_id": "detect",
                                     "duration_ms": 2000}]}
        edit = apply_changeset(self.plan, changeset)
        self.assertEqual(edit["invalid_stage_ids"],
                         ["detect", "isolate", "failover", "verify", "recover"])
        relations = {item["stage_id"]: item["relation"] for item in edit["propagation"]}
        self.assertEqual(relations["detect"], "changed")
        self.assertEqual(relations["isolate"], "downstream")

    def test_leaf_edit_freezes_upstream_byte_for_byte(self):
        changeset = {"operations": [{"op": "update_duration", "stage_id": "recover",
                                     "duration_ms": 1500}]}
        edit = apply_changeset(self.plan, changeset)
        self.assertEqual(edit["invalid_stage_ids"], ["recover"])
        baseline = simulate(self.plan, self.events)
        new_plan = edit["plan"]
        frozen = [sid for sid in new_plan.stage_ids_topo() if sid not in edit["invalid_stage_ids"]]
        fixed = {sid: baseline.outcomes[sid] for sid in frozen}
        partial = simulate(new_plan, self.events, fixed_outcomes=fixed,
                           invalid_stage_ids=edit["invalid_stage_ids"])
        for sid in frozen:
            before = {k: v for k, v in baseline.outcomes[sid].items() if k != "plan_version"}
            after = {k: v for k, v in partial.outcomes[sid].items() if k != "plan_version"}
            self.assertEqual(canonical_json(before), canonical_json(after))
        verify_recompute(new_plan, self.events, partial, frozen)

    def test_delete_referenced_stage_raises_e_depend(self):
        with self.assertRaises(DependencyError) as caught:
            apply_changeset(self.plan, {"operations": [{"op": "delete_stage",
                                                        "stage_id": "detect"}]})
        self.assertEqual(caught.exception.code, "E_DEPEND")
        self.assertIn("isolate", caught.exception.context["referrers"])

    def test_delete_leaf_and_prereq_adjustment(self):
        edit = apply_changeset(self.plan, {
            "operations": [
                {"op": "adjust_prereqs", "stage_id": "recover",
                 "prerequisites": ["failover"]},
                {"op": "delete_stage", "stage_id": "verify"},
            ]
        })
        self.assertEqual(edit["deleted_stage_ids"], ["verify"])
        self.assertIn("recover", edit["invalid_stage_ids"])

    def test_rewrite_criterion_can_flip_outcome(self):
        # 无注入事件的预案下，把 all_of 改成 any_of 并只要求一个动作。
        from emergency_drill.models import EventSet

        quiet = EventSet(events=())
        edit = apply_changeset(self.plan, {
            "operations": [{
                "op": "rewrite_criterion",
                "stage_id": "detect",
                "criterion": {"mode": "any_of", "required_actions": ["a_detect_link"]},
            }]
        })
        result = simulate(edit["plan"], quiet)
        self.assertEqual(result.outcomes["detect"]["state"], "succeeded")

    def test_incremental_mismatch_detects_first_field(self):
        edit = apply_changeset(self.plan, {
            "operations": [{"op": "update_duration", "stage_id": "recover",
                            "duration_ms": 3000}]
        })
        full = simulate(edit["plan"], self.events)
        tampered = simulate(edit["plan"], self.events)
        tampered.outcomes["recover"]["end_clock_ms"] += 999
        with self.assertRaises(IncrementalMismatchError) as caught:
            verify_recompute(edit["plan"], self.events, tampered,
                             [sid for sid in edit["plan"].stage_ids_topo() if sid != "recover"])
        self.assertEqual(caught.exception.code, "E_INCR")
        self.assertEqual(caught.exception.context["stage_id"], "recover")
        self.assertEqual(caught.exception.context["field"], "end_clock_ms")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from edge_ledger.engine import build_workspace
from edge_ledger.loader import prepare_workspace, load_raw_workspace
from edge_ledger.templates import parse_templates, render_template

from .helpers import make_sample_workspace, simple_sample, simple_template, write_custom_workspace


class ValidationAndRenderingTests(unittest.TestCase):
    def test_revision_chain_merge_sources_and_deterministic_render(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            templates = [
                simple_template(
                    revision=1,
                    instruction="Write {topic} in {style}",
                    answer="A {topic}/{style} answer",
                    slots=[{"name": "topic"}, {"name": "style"}],
                ),
                simple_template(
                    revision=2,
                    parent=1,
                    instruction="Write {topic} in {style} for {audience}",
                    answer="A {topic}/{style} answer for {audience}",
                    slots=[{"name": "audience"}, {"name": "style", "allowed_values": ["plain", "rich"]}],
                    constraints={"style": {"max_length": 10}},
                ),
            ]
            samples = [
                {
                    **simple_sample(revision=2),
                    "slot_fills": [
                        {"name": "topic", "value": "cache"},
                        {"name": "style", "value": "plain", "priority": 1},
                        {"name": "style", "value": "rich", "priority": 5},
                        {"name": "style", "value": "plain", "priority": 5},
                        {"name": "audience", "value": "engineers"},
                    ],
                }
            ]
            write_custom_workspace(root, templates=templates, samples=samples)
            loaded = prepare_workspace(load_raw_workspace(root))
            self.assertFalse(loaded.failures)
            effective = loaded.templates.get("tpl", 2)
            self.assertIsNotNone(effective)
            by_name = {slot.rule.name: slot for slot in effective.slots}
            self.assertEqual(by_name["style"].rule.source_revision, 2)
            actions = [item["action"] for item in by_name["style"].origin["overrides"]]
            self.assertEqual(actions, ["append", "override", "constraint"])
            sample = loaded.valid_samples[0]
            resolved = {slot.name: slot for slot in sample.resolved_slots}
            # Same priority: the later input wins deterministically.
            self.assertEqual(resolved["style"].value, "plain")
            self.assertEqual(len(resolved["style"].suppressed), 2)
            result = build_workspace(root)
            row = result.result["samples"][0]
            first = (row["instruction_text"], row["answer_text"], row["render_digest"])
            second_row = build_workspace(root).result["samples"][0]
            self.assertEqual(first, (second_row["instruction_text"], second_row["answer_text"], second_row["render_digest"]))
            self.assertEqual(sample.fill_order, ("audience", "style", "topic"))

    def test_cycle_missing_revision_missing_slot_and_nonfinite_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_sample_workspace(Path(temporary) / "workspace")
            result = build_workspace(root)
            codes = result.result["stats"]["code_counts"]
            self.assertEqual(codes["E_TEMPLATE_INVALID"], 1)
            self.assertEqual(codes["E_SAMPLE_INVALID"], 3)
            self.assertEqual(codes["E_SCORE_NONFINITE"], 1)
            self.assertEqual(result.result["stats"]["total"]["rendered_samples"], 17)
            hits = {item["injection_id"]: item for item in result.result["injection_hits"]}
            self.assertTrue(all(item["hit"] for item in hits.values()))
            self.assertEqual(hits["INJ-NONFINITE-004"]["observed_stages"], ["score"])
            failed_ids = {item["target_id"] for item in result.result["failures"]}
            self.assertIn("S-FAIL-CYCLE", failed_ids)
            self.assertIn("S0001", {item["sample_id"] for item in result.result["samples"]})

    def test_duplicate_ids_and_invalid_config_are_business_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "schema_version": 1,
                "dedupe": {"threshold_permille": 1001},
                "fixed_point": {"decimal_places": 5, "quantization": "Q7"},
            }
            templates = [simple_template(), simple_template()]
            samples = [simple_sample("DUP"), simple_sample("DUP")]
            write_custom_workspace(root, templates=templates, samples=samples, config=config)
            result = build_workspace(root)
            codes = result.result["stats"]["code_counts"]
            self.assertGreaterEqual(codes["E_CONFIG_INVALID"], 3)
            self.assertGreaterEqual(codes["E_TEMPLATE_INVALID"], 1)
            self.assertGreaterEqual(codes["E_SAMPLE_INVALID"], 1)

    def test_template_cycle_detection_directly(self) -> None:
        raw = {
            "templates": [
                simple_template(revision=1),
                simple_template(revision=2, parent=3),
                simple_template(revision=3, parent=2),
            ]
        }
        catalog = parse_templates(raw)
        self.assertTrue(any("cycle" in failure.message for failure in catalog.failures))
        self.assertNotIn(("tpl", 2), catalog.valid_revisions)
        self.assertNotIn(("tpl", 3), catalog.valid_revisions)

    def test_simple_placeholder_renderer_rejects_format_specs(self) -> None:
        self.assertEqual(render_template("A {x} B", {"x": "value"}), "A value B")
        catalog = parse_templates(
            {"templates": [simple_template(instruction="Bad {topic:>4}", answer="ok {topic}")]}
        )
        self.assertTrue(catalog.failures)
        self.assertFalse(catalog.valid_revisions)


if __name__ == "__main__":
    unittest.main()

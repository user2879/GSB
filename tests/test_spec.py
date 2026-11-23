from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from cacheproof.content import content_bytes, entry_digest
from cacheproof.errors import DiagnosticError
from cacheproof.spec import load_spec, parse_spec


ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "cache_spec.json"


class SpecAndContentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(SAMPLE.read_text(encoding="utf-8"))

    def assert_spec_error(self, document: dict, *, segment_id=None, field=None) -> None:
        with self.assertRaises(DiagnosticError) as caught:
            parse_spec(document)
        self.assertEqual(caught.exception.code, "E_SPEC")
        if segment_id is not None:
            self.assertEqual(caught.exception.segment_id, segment_id)
        if field is not None:
            self.assertEqual(caught.exception.field, field)

    def test_sample_loads_and_resolves_snapshot_digests(self) -> None:
        spec = load_spec(SAMPLE)
        self.assertEqual([segment.segment_id for segment in spec.segments], ["seg-alpha", "seg-beta"])
        self.assertEqual(spec.segments[0].declared_used_bytes, 112)
        audit = next(view for view in spec.views if view.view_id == "view-audit")
        alpha = next(segment for segment in audit.segments if segment.segment_id == "seg-alpha")
        self.assertTrue(all(entry.digest and entry.digest.startswith("sha256:") for entry in alpha.entries))

    def test_duplicate_segment_id_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        document["segments"].append(copy.deepcopy(document["segments"][0]))
        self.assert_spec_error(document, segment_id="seg-alpha", field="segment_id")

    def test_non_positive_capacity_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        document["segments"][0]["capacity_bytes"] = 0
        self.assert_spec_error(document, segment_id="seg-alpha", field="capacity_bytes")

    def test_boolean_capacity_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        document["segments"][0]["capacity_bytes"] = True
        self.assert_spec_error(document, segment_id="seg-alpha", field="capacity_bytes")

    def test_duplicate_entry_key_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        document["segments"][0]["entries"].append(
            copy.deepcopy(document["segments"][0]["entries"][0])
        )
        self.assert_spec_error(document, segment_id="seg-alpha", field="entries.key")

    def test_entry_bytes_exceeding_capacity_rejected(self) -> None:
        document = copy.deepcopy(self.document)
        document["segments"][0]["capacity_bytes"] = 100
        self.assert_spec_error(document, segment_id="seg-alpha", field="capacity_bytes")

    def test_view_version_must_increase_per_view(self) -> None:
        document = copy.deepcopy(self.document)
        document["views"].append(copy.deepcopy(document["views"][0]))
        document["views"][-1]["version"] = 1
        self.assert_spec_error(document, field="views[view-orders].version")

    def test_view_must_reference_existing_segment(self) -> None:
        document = copy.deepcopy(self.document)
        document["views"][0]["segments"][0]["segment_id"] = "missing"
        self.assert_spec_error(document, segment_id="missing", field="views[view-orders].segments.segment_id")

    def test_views_field_is_required(self) -> None:
        document = copy.deepcopy(self.document)
        document.pop("views")
        self.assert_spec_error(document, field="views")

    def test_content_and_digest_are_deterministic(self) -> None:
        first = content_bytes("seed-a", 100)
        second = content_bytes("seed-a", 100)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertNotEqual(first, content_bytes("seed-b", 100))
        self.assertEqual(
            entry_digest("key", 100, "seed-a"), entry_digest("key", 100, "seed-a")
        )
        self.assertNotEqual(
            entry_digest("key", 100, "seed-a"), entry_digest("other", 100, "seed-a")
        )


if __name__ == "__main__":
    unittest.main()

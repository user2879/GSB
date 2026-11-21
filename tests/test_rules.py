import unittest

from configlease.errors import (
    KeyNotCoveredError,
    LeaseExpiredError,
    RevisionOutOfRangeError,
    RollbackRevisionError,
)
from configlease.clock import now_ms
from configlease.engine import validate_request, verify_apply_result, content_hash
from configlease.manifest import Revision, fold_view, parse_manifest
from configlease.sample import sample_for_revision, summary_for_revision


def make_lease(**over):
    base = {
        "lease_id": "L1", "request_id": "R1", "node_id": "n1",
        "rev_seq": 2, "low_seq": 1, "high_seq": 3,
        "keys": ["a", "b"], "issued_ms": now_ms(),
        "deadline_ms": now_ms() + 60_000,
    }
    base.update(over)
    return base


class RuleTests(unittest.TestCase):
    def test_rev_below_window(self):
        with self.assertRaises(RevisionOutOfRangeError):
            validate_request({"rev_seq": 0, "changes": {"a": 1}},
                             make_lease(), set())

    def test_rev_above_window(self):
        with self.assertRaises(RevisionOutOfRangeError):
            validate_request({"rev_seq": 4, "changes": {"a": 1}},
                             make_lease(), set())

    def test_key_not_covered(self):
        with self.assertRaises(KeyNotCoveredError):
            validate_request({"rev_seq": 2, "changes": {"z": 1}},
                             make_lease(), set())

    def test_rollback_revision_rejected(self):
        with self.assertRaises(RollbackRevisionError):
            validate_request({"rev_seq": 2, "changes": {"a": 1}},
                             make_lease(), {2})

    def test_expired_lease_rejected(self):
        with self.assertRaises(LeaseExpiredError):
            validate_request(
                {"rev_seq": 2, "changes": {"a": 1}},
                make_lease(deadline_ms=now_ms() - 1), set(),
            )

    def test_verify_mismatch_detects_cursor_and_summary(self):
        lease = make_lease()
        response = {
            "applied_rev": 99,
            "view": {"a": 1},
            "summary": {"bogus": True},
        }
        with self.assertRaises(Exception) as ctx:
            verify_apply_result(
                {"rev_seq": 2, "changes": {"a": 1}}, lease, response, {}, 0,
            )
        fields = {m["field"] for m in ctx.exception.context["mismatches"]}
        self.assertEqual(fields, {"applied_rev", "summary"})

    def test_deterministic_summary_across_calls(self):
        a = summary_for_revision(5)
        b = summary_for_revision(5)
        self.assertEqual(a, b)
        self.assertEqual(sample_for_revision(5).shape[0], 64)

    def test_manifest_parse_and_fold(self):
        doc = {"revisions": [
            {"rev_seq": 1, "parent": 0, "changes": {"a": 1}},
            {"rev_seq": 2, "parent": 1, "changes": {"a": 2, "b": 7}},
        ]}
        revs = parse_manifest(doc)
        self.assertEqual(fold_view(revs), {"a": 2, "b": 7})
        self.assertEqual(fold_view(revs, upto_seq=1), {"a": 1})

    def test_manifest_corruption_raises_manifest_error(self):
        from configlease.errors import ManifestError
        with self.assertRaises(ManifestError):
            parse_manifest({"revisions": [{"rev_seq": 1, "parent": 0,
                                           "changes": {"a": "x"}}]})

    def test_content_hash_stable_and_sensitive(self):
        s1 = {"x": 1, "y": [1, 2]}
        s2 = {"y": [1, 2], "x": 1}
        s3 = {"x": 1, "y": [1, 3]}
        self.assertEqual(content_hash(s1), content_hash(s2))
        self.assertNotEqual(content_hash(s1), content_hash(s3))


if __name__ == "__main__":
    unittest.main()


class LeaseTtlTests(unittest.TestCase):
    def test_monotonic_expiry_using_real_clock(self):
        import asyncio
        from configlease.clock import deadline_from_ttl, is_expired, remaining_ms

        deadline = deadline_from_ttl(30)
        self.assertFalse(is_expired(deadline))
        self.assertGreater(remaining_ms(deadline), 0)

        async def wait_out():
            await asyncio.sleep(0.05)

        asyncio.run(wait_out())
        short = deadline_from_ttl(-10)
        self.assertTrue(is_expired(short))

from __future__ import annotations

import unittest

from batch_audit.dataset import generate_assets
from batch_audit.feed import FeedError, parse_feed


class FeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assets = generate_assets(seed=4, sample_count=8, input_dim=3, hidden_dim=4)
        self.ids = self.assets.sample_ids
        self.entries = [
            {"sample_id": self.ids[0], "arrival_seq": 4, "batch_slot": 0, "admitted": True},
            {"sample_id": self.ids[1], "arrival_seq": 1, "batch_slot": 1, "admitted": True},
            {"sample_id": self.ids[2], "arrival_seq": 3, "batch_slot": 0, "admitted": True},
            {"sample_id": self.ids[0], "arrival_seq": 8, "batch_slot": 0, "admitted": True},
            {"sample_id": self.ids[3], "arrival_seq": 2, "batch_slot": 2, "admitted": False, "reject_reason": "policy"},
        ]

    def _parse(self, entries, slots=(0, 1, 2, 3)):
        return parse_feed(
            {"schema": "batch-feed-v1", "entries": entries},
            supported_slots=slots,
            known_sample_ids=self.ids,
        )

    def test_grouping_uses_arrival_order_and_keeps_earliest_retransmission(self) -> None:
        first = self._parse(self.entries)
        second = self._parse(list(reversed(self.entries)))
        self.assertEqual(first.feed_digest, second.feed_digest)
        self.assertEqual(
            [(g.batch_slot, g.member_ids) for g in first.groups],
            [(g.batch_slot, g.member_ids) for g in second.groups],
        )
        slot0 = first.group_by_slot()[0]
        self.assertEqual(slot0.member_ids, (self.ids[2], self.ids[0]))
        self.assertEqual([member.rank for member in slot0.members], [1, 2])
        self.assertEqual(len(slot0.overwritten), 1)
        self.assertEqual(slot0.overwritten[0].arrival_seq, 8)
        self.assertEqual(slot0.overwritten[0].kept_arrival_seq, 4)
        self.assertEqual(first.empty_slots, (2, 3))

    def test_failed_entries_are_reported_by_slot_and_reason(self) -> None:
        plan = self._parse(self.entries)
        slot2 = plan.group_by_slot()[2]
        self.assertEqual(slot2.members, ())
        self.assertEqual(slot2.failed[0].reason, "policy")
        self.assertEqual(plan.failed_entries[0].batch_slot, 2)

    def test_duplicate_arrival_is_rejected_with_field_and_arrival(self) -> None:
        entries = [dict(item) for item in self.entries]
        entries[1]["arrival_seq"] = 4
        with self.assertRaises(FeedError) as caught:
            self._parse(entries)
        error = caught.exception
        self.assertEqual(error.code, "E_FEED")
        self.assertEqual(error.arrival_seq, 4)
        self.assertIn(".arrival_seq", error.field_path)

    def test_invalid_admitted_field_is_rejected(self) -> None:
        entries = [dict(item) for item in self.entries]
        entries[0]["admitted"] = 1
        with self.assertRaises(FeedError) as caught:
            self._parse(entries)
        self.assertEqual(caught.exception.arrival_seq, 4)
        self.assertIn(".admitted", caught.exception.field_path)

    def test_unknown_sample_is_rejected(self) -> None:
        entries = [dict(item) for item in self.entries]
        entries[0]["sample_id"] = "NOT_PRESENT"
        with self.assertRaises(FeedError) as caught:
            self._parse(entries)
        self.assertIn(".sample_id", caught.exception.field_path)

    def test_rejected_entry_may_use_external_identifier(self) -> None:
        entries = [dict(item) for item in self.entries]
        entries.append(
            {
                "sample_id": "EXTERNAL-REJECTED-1",
                "arrival_seq": 20,
                "batch_slot": 1,
                "admitted": False,
                "reject_reason": "external_schema",
            }
        )
        plan = self._parse(entries)
        self.assertEqual(plan.group_by_slot()[1].failed[0].sample_id, "EXTERNAL-REJECTED-1")

    def test_cross_slot_success_duplicate_is_rejected(self) -> None:
        entries = [dict(item) for item in self.entries]
        entries.append(
            {"sample_id": self.ids[0], "arrival_seq": 9, "batch_slot": 1, "admitted": True}
        )
        with self.assertRaises(FeedError) as caught:
            self._parse(entries)
        self.assertIn(".batch_slot", caught.exception.field_path)
        self.assertEqual(caught.exception.arrival_seq, 9)


if __name__ == "__main__":
    unittest.main()

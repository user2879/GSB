import unittest

from ffbudget import errors
from ffbudget.authority import adjudicate
from ffbudget.models import Receipt, SlotSpec
from ffbudget.view import (STATE_FRESH, STATE_FUTURE_FALLBACK,
                           STATE_STALE, STATE_UNAVAILABLE,
                           build_history_index, resolve_slot,
                           apply_staleness_cap)


def slot(retention=2):
    return SlotSpec(key="s", layer="item", dim=2, retention=retention,
                    weight=3, default_vector=(9, 9),
                    default_vector_source="test://default")


def receipt(batch, rev, effective, digest_tail=None, vector=(1, 1),
            group="s#item"):
    digest = digest_tail or f"d-{batch}-{rev}-{effective}"
    return Receipt(batch_seq=batch, rev_seq=rev, slot_key="s",
                   effective_ms=effective, points=2, vector=vector,
                   item_id="item", digest=digest, group_key=group)


class AuthorityTests(unittest.TestCase):
    def test_ranking_priority_rev_then_effective_then_batch(self):
        receipts = [
            receipt(1, 1, 100, "aaa"),
            receipt(2, 2, 50, "bbb"),
            receipt(3, 2, 80, "ccc"),
        ]
        authorities, rejections, ledger, duplicates = adjudicate(
            receipts, {"s": slot()})
        winner = authorities["s#item"].receipt
        self.assertEqual(winner.digest, "ccc")
        reasons = {item.digest: item.reason for item in rejections}
        self.assertEqual(reasons["bbb"], errors.E_EFFECTIVE_EARLIER)
        self.assertEqual(reasons["aaa"], errors.E_REV_SEQ_LOWER)
        self.assertEqual(duplicates, [])
        self.assertTrue(all(item["counted"] for item in ledger
                            if item["digest"] == winner.digest))

    def test_full_tie_broken_by_digest_and_flagged(self):
        receipts = [receipt(1, 1, 100, "zzz"),
                    receipt(1, 1, 100, "aaa")]
        authorities, rejections, _, _ = adjudicate(
            receipts, {"s": slot()})
        record = authorities["s#item"]
        self.assertTrue(record.tie_resolved)
        self.assertEqual(record.receipt.digest, "aaa")
        self.assertEqual(rejections[0].reason, errors.E_TIE_DIGEST)
        self.assertEqual(rejections[0].conflicting_digest, "aaa")

    def test_duplicate_redelivery_zero_points(self):
        one = receipt(1, 1, 100, "same")
        receipts = [one, Receipt(**one.__dict__)]
        authorities, _, ledger, duplicates = adjudicate(
            receipts, {"s": slot()})
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["code"],
                         errors.E_DUPLICATE_BATCH_DELIVERY)
        self.assertFalse(duplicates[0]["counted"])
        self.assertEqual(len(authorities), 1)
        billed = [item for item in ledger if item["counted"]]
        self.assertEqual(len(billed), 1)

    def test_retention_prunes_oldest_and_records_revisions(self):
        receipts = [receipt(b, b, b * 10, f"d{b}") for b in range(1, 6)]
        authorities, _, ledger, _ = adjudicate(
            receipts, {"s": slot(retention=2)})
        record = authorities["s#item"]
        self.assertEqual(len(record.history), 2)
        self.assertEqual(len(record.pruned), 3)
        self.assertTrue(all(item["code"] == errors.E_REVISION_PRUNED
                            for item in record.pruned))
        pruned_ledger = [item for item in ledger if not item["counted"]]
        self.assertEqual(len(pruned_ledger), 3)


class ViewTests(unittest.TestCase):
    def test_fresh_then_stale_then_future_fallback(self):
        spec = slot()
        history = [receipt(1, 1, 100, "old"),
                   receipt(2, 2, 300, "new")]
        authority = adjudicate(history, {"s": spec})[0]["s#item"]
        index = build_history_index(
            adjudicate(history, {"s": spec})[0], history)

        fresh = resolve_slot(300, spec, authority, index["s#item"])
        self.assertEqual(fresh.state, STATE_FRESH)
        self.assertEqual(fresh.staleness_ms, 0)

        stale = resolve_slot(450, spec, authority, index["s#item"])
        self.assertEqual(stale.state, STATE_STALE)
        self.assertEqual(stale.used_digest, "new")
        self.assertEqual(stale.staleness_ms, 150)

        early = resolve_slot(50, spec, authority, index["s#item"])
        self.assertEqual(early.state, STATE_FUTURE_FALLBACK)
        self.assertEqual(early.used_digest, "old")
        self.assertEqual(early.staleness_ms, -50)

    def test_missing_group_uses_default_vector(self):
        spec = slot()
        resolution = resolve_slot(100, spec, None, [])
        self.assertEqual(resolution.state, STATE_UNAVAILABLE)
        self.assertEqual(resolution.vector, (9, 9))
        self.assertTrue(resolution.vector_source.startswith("default:"))

    def test_staleness_cap_marks_unavailable_but_keeps_default_source(self):
        spec = slot()
        history = [receipt(1, 1, 100, "old")]
        authority = adjudicate(history, {"s": spec})[0]["s#item"]
        index = build_history_index(
            adjudicate(history, {"s": spec})[0], history)
        resolution = resolve_slot(5000, spec, authority, index["s#item"])
        capped = apply_staleness_cap(resolution, spec, max_staleness=1000)
        self.assertEqual(capped.state, STATE_UNAVAILABLE)
        self.assertEqual(capped.vector, (9, 9))
        self.assertEqual(capped.staleness_ms, 4900)
        self.assertTrue(capped.vector_source.startswith("default:"))


if __name__ == "__main__":
    unittest.main()

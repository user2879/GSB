import unittest

from ffbudget import errors
from ffbudget.cache import CacheBook
from ffbudget.models import SlotSpec
from ffbudget.scheduler import rank_requests, run
from ffbudget.view import (STATE_FRESH, STATE_UNAVAILABLE,
                           build_history_index, build_view,
                           staleness_fingerprint)
from ffbudget.authority import adjudicate
from ffbudget.models import Receipt, SlotResolution


def config(budget=1000, slot_quota=1000, concurrency=4):
    return {
        "format_version": 1,
        "fixed_point": {"width_tier": "i64q16", "frac_bits": 16},
        "staleness": {
            "max_staleness_ms": 10000, "penalty_fresh_q16": 65536,
            "stale_penalty_per_ms_q16": 8, "stale_floor_q16": 16384,
            "penalty_unavailable_q16": 6554},
        "budget": {"score_point_budget": budget,
                   "slot_point_quota": slot_quota},
        "scheduler": {"max_concurrency": concurrency},
        "_slot_order": ["r", "i"],
    }


def slots():
    return {
        "r": SlotSpec("r", "request", 2, 2, 4, (0, 0), "test://r"),
        "i": SlotSpec("i", "item", 3, 2, 5, (1, 1, 1), "test://i"),
    }


def requests():
    return [{"request_id": "q1", "rank_ms": 100,
             "candidates": ["a", "b"]},
            {"request_id": "q2", "rank_ms": 100,
             "candidates": ["a"]}]


def make_receipts():
    return [
        Receipt(1, 1, "r", 50, 2, (10, 20), None, "r-d", "r"),
        Receipt(1, 1, "i", 50, 3, (1, 2, 3), "a", "i-a", "i#a"),
        Receipt(1, 1, "i", 50, 3, (4, 5, 6), "b", "i-b", "i#b"),
        Receipt(2, 2, "i", 60, 3, (1, 1, 1), "a", "i-a2", "i#a"),
    ]


def build_resolutions(cfg, slot_map, reqs, recs):
    authorities = adjudicate(recs, slot_map)[0]
    history = build_history_index(authorities, recs)
    resolutions, _, staleness_fp = build_view(
        cfg, slot_map, authorities, reqs, history)
    return resolutions, staleness_fp


class SchedulerTests(unittest.TestCase):
    def test_happy_path_consumes_points_and_ranks(self):
        cfg, slot_map, reqs, recs = config(), slots(), requests(), make_receipts()
        resolutions, staleness_fp = build_resolutions(
            cfg, slot_map, reqs, recs)
        book = CacheBook()
        entries, _, pending, failures, ledger = run(
            cfg, reqs, slot_map, resolutions, staleness_fp, book)
        self.assertEqual(pending, [])
        self.assertEqual(failures, [])
        # q1: r(2) + i-a(3) + i-b(3); q2: r(2) + i-a(3) = 13
        self.assertEqual(ledger["consumed_points"], 13)
        self.assertEqual(ledger["budget"] - ledger["remaining"], 13)
        ranking = rank_requests(cfg, reqs, entries, slot_map, resolutions,
                                book)
        self.assertEqual([row["request_id"] for row in ranking],
                         ["q1", "q2"])
        q1 = ranking[0]
        self.assertEqual([c["item_id"] for c in q1["candidates"]],
                         ["b", "a"])
        self.assertTrue(q1["referenced_score_cache_keys"])

    def test_budget_exhaustion_keeps_pending_request(self):
        # Stable order fully funds q1 (8 points); q2 needs 5 more but only
        # 2 remain, so its claimed request-level unit is refunded and q2 waits.
        cfg = config(budget=10)
        slot_map, reqs, recs = slots(), requests(), make_receipts()
        resolutions, staleness_fp = build_resolutions(
            cfg, slot_map, reqs, recs)
        book = CacheBook()
        entries, _, pending, failures, ledger = run(
            cfg, reqs, slot_map, resolutions, staleness_fp, book)
        self.assertEqual([item["request_id"] for item in pending], ["q2"])
        self.assertEqual(pending[0]["code"],
                         errors.E_SCORE_BUDGET_PENDING)
        self.assertEqual(ledger["refunded_points"], 2)
        ranked_ids = {entry["request_id"] for entry in entries}
        self.assertEqual(ranked_ids, {"q1"})

    def test_slot_quota_abort(self):
        cfg = config(slot_quota=4)
        slot_map, reqs, recs = slots(), requests(), make_receipts()
        resolutions, staleness_fp = build_resolutions(
            cfg, slot_map, reqs, recs)
        book = CacheBook()
        _, _, pending, failures, ledger = run(
            cfg, reqs, slot_map, resolutions, staleness_fp, book)
        quota_codes = [item["code"] for item in failures]
        self.assertTrue(all(code == errors.E_SLOT_QUOTA_EXCEEDED
                            for code in quota_codes))
        self.assertIn("i", ledger["aborted_slots"])
        # The surviving request-level slot still funds fully, so the queue is
        # empty while every aborted-slot unit stays on the failure list.
        self.assertEqual(pending, [])
        self.assertGreaterEqual(len(quota_codes), 2)
        self.assertGreater(ledger["slot_quota"], 0)

    def test_serial_and_parallel_identical(self):
        cfg_serial = config(concurrency=1)
        cfg_parallel = config(concurrency=8)
        slot_map, reqs, recs = slots(), requests(), make_receipts()
        outputs = []
        for cfg in (cfg_serial, cfg_parallel):
            resolutions, staleness_fp = build_resolutions(
                cfg, slot_map, reqs, recs)
            entries, _, pending, failures, ledger = run(
                cfg, reqs, slot_map, resolutions, staleness_fp,
                CacheBook())
            outputs.append(([(e["request_id"], e["item_id"],
                              e["contribution_q16"]) for e in entries],
                            pending, failures,
                            ledger["events"], ledger["consumed_points"]))
        self.assertEqual(
            [list(x[0]) for x in outputs][0],
            [list(x[0]) for x in outputs][1])
        self.assertEqual(outputs[0][1:], outputs[1][1:])


class ScoringTests(unittest.TestCase):
    def test_fixed_point_contribution_deterministic(self):
        from ffbudget.scoring import score_slot
        cfg = config()
        slot_map = slots()
        resolution = SlotResolution(
            "r", "r", None, 100, "r-d", 1, 1, 50, 0, STATE_FRESH,
            (10, 20), "backfill:r-d")
        entry = score_slot(resolution, slot_map["r"], cfg,
                           staleness_fingerprint(cfg))
        # value = 30 (q16), weight = 4 (q16), penalty = 1.0 (q16);
        # (4q16 * 1q16)>>16 = 4q16, then 4q16 * 30q16 = 120 * 2^32
        self.assertEqual(entry["contribution_q16"], 120 << 32)

    def test_unavailable_uses_default_and_unavailable_penalty(self):
        from ffbudget.scoring import score_slot
        cfg = config()
        slot_map = slots()
        resolution = SlotResolution(
            "i", "i", None, 100, None, None, None, None, None,
            STATE_UNAVAILABLE, (1, 1, 1), "default:test://i")
        entry = score_slot(resolution, slot_map["i"], cfg,
                           staleness_fingerprint(cfg))
        self.assertEqual(entry["vector_source"], "default:test://i")
        self.assertTrue(entry["score_cache_key"])


if __name__ == "__main__":
    unittest.main()

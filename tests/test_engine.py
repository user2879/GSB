from __future__ import annotations

import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from streamagg.engine import Engine
from streamagg.fixed import INT64_MAX
from streamagg.generator import generate_session
from streamagg.ledger import Ledger
from streamagg.model import Event
from streamagg.spec import load_spec, parse_spec


ROOT = Path(__file__).resolve().parents[1]


def tiny_spec(*, budget_count: int = 2, budget_amount: int = 1000):
    payload = {
        "format_version": 1,
        "sources": [
            {
                "source_id": "tiny",
                "seed": 1,
                "event_count": 1,
                "out_of_order_ratio_bp": 0,
                "partition_keys": ["p0"],
            }
        ],
        "rules": [
            {
                "rule_id": "r",
                "width_ms": 100,
                "lateness_ms": 10,
                "partition_keys": ["p0"],
                "aggregation_functions": ["COUNT", "SUM", "MEAN"],
            }
        ],
        "session": {
            "fixed_scale": 1,
            "lateness_budget_count": budget_count,
            "lateness_budget_amount": budget_amount,
            "order_backward_limit_ms": 100000,
        },
    }
    return parse_spec(payload)


def event(arrival: int, timestamp: int, measure: int, *, event_id: str | None = None) -> Event:
    return Event(
        event_id=event_id or f"tiny:{arrival:08d}",
        source_id="tiny",
        source_seq=arrival,
        arrival_no=arrival,
        partition_key="p0",
        event_time_ms=timestamp,
        measure=measure,
    )


class EngineSemanticsTests(unittest.TestCase):
    def test_watermark_trace_and_finalized_window_immutability(self) -> None:
        engine = Engine(tiny_spec(), session_id="immutability")
        engine.ingest(event(0, 1050, 1))
        engine.ingest(event(1, 1200, 2))
        self.assertEqual(len(engine.finalized_windows), 1)
        finalized = next(iter(engine.finalized_windows.values()))
        self.assertEqual(finalized.sum_fixed, 1)
        engine.ingest(event(2, 1099, 99))
        self.assertEqual(next(iter(engine.finalized_windows.values())).sum_fixed, 1)
        dropped = [decision for decision in engine.decisions if decision.decision == "DROP"]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].reason, "WINDOW_ALREADY_FINALIZED")
        self.assertEqual(dropped[0].window_start_ms, 1000)
        self.assertEqual(engine.traces[0].watermark_before_ms, 1050)
        self.assertEqual(engine.traces[0].watermark_after_ms, 1050)
        self.assertEqual(engine.traces[1].watermark_before_ms, 1050)
        self.assertEqual(engine.traces[1].watermark_after_ms, 1200)

    def test_budget_exact_amount_boundary_has_unique_accept_then_drop(self) -> None:
        engine = Engine(tiny_spec(budget_count=2, budget_amount=10), session_id="budget-equality")
        engine.ingest(event(0, 1500, 1))
        engine.ingest(event(1, 1450, 10))
        late_accepts = [decision for decision in engine.decisions if decision.was_late and decision.decision == "ACCEPT"]
        self.assertEqual(len(late_accepts), 1)
        self.assertEqual(late_accepts[0].budget_amount_after, 0)
        self.assertEqual(late_accepts[0].reason, "LATE_WITHIN_BUDGET")
        engine.ingest(event(2, 1451, 1))
        last = engine.decisions[-1]
        self.assertEqual(last.decision, "DROP")
        self.assertEqual(last.reason, "LATE_BUDGET_AMOUNT_EXCEEDED")
        self.assertEqual(last.budget_amount_before, 0)
        self.assertEqual(last.arrival_no, 2)

    def test_budget_count_exhaustion_records_evidence(self) -> None:
        engine = Engine(tiny_spec(budget_count=1, budget_amount=1000), session_id="budget-count")
        engine.ingest(event(0, 1500, 1))
        engine.ingest(event(1, 1450, 3))
        engine.ingest(event(2, 1451, 4))
        last = engine.decisions[-1]
        self.assertEqual(last.reason, "LATE_BUDGET_COUNT_EXHAUSTED")
        self.assertEqual(last.budget_count_before, 0)
        self.assertEqual(last.budget_count_after, 0)
        self.assertEqual(last.cumulative_late_abs_amount, 3)
        self.assertEqual(last.arrival_no, 2)

    def test_shuffled_input_is_canonicalized_and_identical(self) -> None:
        spec = load_spec(ROOT / "samples" / "sample_spec.json")
        events, _ = generate_session(spec)
        shuffled = list(events)
        random.Random(2026).shuffle(shuffled)
        first = Engine(spec, session_id="order").run_events(events)
        second = Engine(spec, session_id="order").run_events(shuffled)
        self.assertEqual(first["finalized_window_keys"], second["finalized_window_keys"])
        self.assertEqual(first["drop_decisions"], second["drop_decisions"])
        self.assertEqual(first["watermarks"], second["watermarks"])
        self.assertEqual(first["ambiguity_groups"], second["ambiguity_groups"])
        for group in first["ambiguity_groups"]:
            expected_order = sorted(
                group["stable_order"],
                key=lambda key: (
                    key["partition_key"],
                    key["window_start_ms"],
                    key["window_end_ms"],
                    key["rule_id"],
                    key["rule_version"],
                ),
            )
            self.assertEqual(group["stable_order"], expected_order)

    def test_rule_version_change_marks_old_view_stale_and_replaces_view(self) -> None:
        with self.subTest("stale"):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                spec = load_spec(ROOT / "samples" / "scenario1_ordered.json")
                with Ledger(Path(directory) / "ledger.sqlite3") as ledger:
                    engine = Engine(spec, session_id="evolve", ledger=ledger)
                    engine.run_all()
                    with self.assertRaises(Exception) as unchanged:
                        engine.evolve_spec(spec, reason="no-op")
                    self.assertEqual(getattr(unchanged.exception, "code", None), "E_SPEC")
                    old_view = engine.view_id
                    old_rule = spec.rules[0]
                    evolved_rule = replace(
                        old_rule,
                        width_ms=300,
                        lateness_ms=20,
                        partition_keys=("p0", "p1"),
                        rule_version=2,
                    )
                    evolved_spec = replace(
                        spec,
                        rules=(evolved_rule,),
                        fixed_scale=50,
                    )
                    info = engine.evolve_spec(evolved_spec, reason="test evolution")
                    self.assertEqual(info["rules_version"], 2)
                    self.assertEqual(info["affected_partitions"], ["p0"])
                    self.assertEqual(
                        info["affected_windows"],
                        sorted(
                            info["affected_windows"],
                            key=lambda item: (
                                item["partition_key"],
                                item["window_start_ms"],
                                item["window_end_ms"],
                                item["rule_id"],
                                item["rule_version"],
                            ),
                        ),
                    )
                    self.assertGreater(len(info["affected_windows"]), 0)
                    self.assertFalse(ledger.current_windows("evolve", old_view))
                    stale_rows = ledger.all_windows("evolve", old_view, include_stale=True)
                    self.assertTrue(stale_rows)
                    self.assertTrue(all(row["stale"] == 1 for row in stale_rows))
                    self.assertNotEqual(old_view, engine.view_id)

    def test_extreme_fixed_values_saturate_without_sqlite_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = replace(tiny_spec(), fixed_scale=2)
            huge = 1 << 62
            with Ledger(Path(directory) / "ledger.sqlite3") as ledger:
                engine = Engine(spec, session_id="overflow", ledger=ledger)
                engine.ingest(event(0, 100, huge))
                engine.ingest(event(1, 101, huge))
                engine.ingest(event(2, 250, 0))
                result = next(iter(engine.finalized_windows.values()))
                self.assertEqual(result.sum_fixed, INT64_MAX)
                self.assertEqual(result.attribution.quantization_truncation, INT64_MAX)
                self.assertEqual(result.attribution.sum_bound, INT64_MAX)
                row = ledger.connection.execute(
                    "SELECT sum_exact, sum_fixed FROM window_state WHERE session_id = ? AND view_id = ? AND status = 'COMMITTED'",
                    ("overflow", engine.view_id),
                ).fetchone()
                self.assertEqual(row["sum_fixed"], INT64_MAX)
                self.assertEqual(int(row["sum_exact"]), 2 * INT64_MAX)


if __name__ == "__main__":
    unittest.main()

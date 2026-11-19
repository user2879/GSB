from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rts_pipeline.engine import synthesize_game
from rts_pipeline.errors import PipelineError
from rts_pipeline.events import normalize_events, read_events
from rts_pipeline.rng import derive_record, verify_records
from rts_pipeline.rules import initial_state, validate_rules

from .helpers import make_workspace


def _relative_bytes(root: Path, directories: tuple[str, ...]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for directory in directories:
        for path in sorted((root / directory).rglob("*")):
            if path.is_file():
                result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


class GeneratorTests(unittest.TestCase):
    def test_generation_is_byte_identical_and_reports_counts(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            one = make_workspace(Path(first), seed=77, ticks=18, faction_count=4, chunk_ticks=5)
            make_workspace(Path(second), seed=77, ticks=18, faction_count=4, chunk_ticks=5)
            one_files = _relative_bytes(Path(first), ("input", "truth", "report"))
            two_files = _relative_bytes(Path(second), ("input", "truth", "report"))
            self.assertEqual(one_files, two_files)
            report = json.loads((Path(first) / "report" / "generation.json").read_text())
            self.assertEqual(report["event_count"], len(one["events"]))
            self.assertEqual(report["tick_span"], 18)
            self.assertEqual(
                sum(report["faction_event_counts"].values()), report["event_count"]
            )
            for faction in ("A", "B", "C", "D"):
                self.assertTrue((Path(first) / "truth" / "factions" / f"{faction}.json").is_file())

    def test_overflow_injection_reaches_truth_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            generated = make_workspace(
                Path(root),
                seed=91,
                ticks=12,
                overflow_at=(6, "B"),
            )
            truth = generated["truth"]
            self.assertEqual(truth["result_codes"], ["E_RESOURCE_OVERFLOW"])
            self.assertEqual(len(truth["violations"]), 1)
            violation = truth["violations"][0]
            self.assertEqual(violation["reason"], "resource_cap")
            self.assertGreater(violation["attempted_resources"], violation["resource_cap"])

    def test_rng_records_recompute_from_seed_tick_and_delta(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            generated = make_workspace(Path(root), seed=123)
            records = generated["truth"]["rng_records"]
            self.assertEqual(verify_records(records, 123), len(records))
            for record in records[:10]:
                expected = derive_record(123, record["tick"], record["rng_seed_delta"])
                self.assertEqual(record["value_u64"], expected["value_u64"])

    def test_malformed_event_line_is_reported_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            generated = make_workspace(Path(root), seed=131, ticks=4)
            path = Path(root) / "input" / "events.jsonl"
            original = path.read_text(encoding="utf-8").splitlines()
            broken = json.loads(original[0])
            broken.pop("local_seq")
            path.write_text(
                "\n".join([json.dumps(broken), *original[1:]]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(PipelineError) as caught:
                read_events(path, generated["rules"])
            self.assertEqual(caught.exception.code, "E_MALFORMED_EVENT")
            self.assertEqual(caught.exception.details.get("line"), 1)


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = validate_rules(
            {
                "schema_version": 1,
                "factions": {
                    "A": {
                        "resource_cap": 100,
                        "collect_rate": 2,
                        "queue_capacity": 2,
                        "unit_costs": {"infantry": 5},
                        "build_ticks": {"infantry": 1},
                    }
                },
            }
        )
        self.events = [
            {
                "tick": 0,
                "actor": "A",
                "action": "collect",
                "target": "field",
                "amount": 3,
                "rng_seed_delta": 1,
                "local_seq": 0,
            },
            {
                "tick": 0,
                "actor": "A",
                "action": "produce",
                "target": "infantry",
                "amount": 1,
                "rng_seed_delta": 2,
                "local_seq": 1,
            },
            {
                "tick": 1,
                "actor": "A",
                "action": "dispatch",
                "target": "infantry",
                "amount": 1,
                "rng_seed_delta": 3,
                "local_seq": 2,
            },
        ]

    def _synthesize(self, events):
        return synthesize_game(
            events,
            self.rules,
            500,
            0,
            2,
            initial_state(self.rules),
        )

    def test_ticks_are_causal_and_random_sequence_is_repeatable(self) -> None:
        first = self._synthesize(self.events)
        second = self._synthesize(list(reversed(self.events)))
        self.assertEqual(first["event_records"], second["event_records"])
        self.assertEqual(first["tick_records"], second["tick_records"])
        self.assertEqual(first["rng_records"], second["rng_records"])
        tick0 = first["tick_records"][0]["snapshot"]["factions"]["A"]
        tick1 = first["tick_records"][1]["snapshot"]["factions"]["A"]
        self.assertEqual(len(tick0["queue"]), 1)
        self.assertEqual(tick0["queue"][0]["remaining"], 1)
        self.assertEqual(len(tick1["queue"]), 0)
        self.assertEqual(tick1["dispatched_total"], 1)

    def test_sequence_gap_is_fatal(self) -> None:
        broken = [dict(event) for event in self.events]
        broken[2]["local_seq"] = 3
        with self.assertRaises(PipelineError) as caught:
            self._synthesize(broken)
        self.assertEqual(caught.exception.code, "E_SEQ_GAP")

    def test_missing_initial_ledger_is_sequence_gap(self) -> None:
        with self.assertRaises(PipelineError) as caught:
            synthesize_game(self.events, self.rules, 500, 0, 2, None)
        self.assertEqual(caught.exception.code, "E_SEQ_GAP")


class OverflowAccountingTests(unittest.TestCase):
    def test_overflow_and_negative_budget_are_marked_per_event(self) -> None:
        rules = validate_rules(
            {
                "schema_version": 1,
                "factions": {
                    "A": {
                        "resource_cap": 10,
                        "collect_rate": 1,
                        "queue_capacity": 1,
                        "unit_costs": {"infantry": 5},
                        "build_ticks": {"infantry": 0},
                    }
                },
            }
        )
        events = normalize_events(
            [
                {
                    "tick": 0,
                    "actor": "A",
                    "action": "collect",
                    "target": "field",
                    "amount": 15,
                    "rng_seed_delta": 4,
                    "local_seq": 0,
                },
                {
                    "tick": 0,
                    "actor": "A",
                    "action": "produce",
                    "target": "infantry",
                    "amount": 1,
                    "rng_seed_delta": 5,
                    "local_seq": 1,
                },
            ],
            rules,
        )
        result = synthesize_game(events, rules, 9, 0, 0, initial_state(rules))
        self.assertEqual(result["result_codes"], ["E_RESOURCE_OVERFLOW"])
        self.assertEqual(len(result["violations"]), 2)
        self.assertEqual(result["violations"][0]["reason"], "resource_cap")
        self.assertEqual(result["violations"][1]["reason"], "resource_negative")
        self.assertTrue(all(record["outcome"] == "rejected" for record in result["event_records"]))


if __name__ == "__main__":
    unittest.main()

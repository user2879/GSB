from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from h2burst.engine import PROPORTIONAL, ROUND_ROBIN, Engine
from h2burst.model import load_session, parse_session
from tests.helpers import raw_session, spec_session, strategy, wu


ROOT = Path(__file__).resolve().parents[1]


def frame_signature(engine: Engine) -> list[tuple[object, ...]]:
    return [
        (
            frame.frame_type,
            frame.stream_id,
            frame.data_bytes,
            frame.event_seq,
            frame.strategy,
            frame.connection_window_before,
            frame.connection_window_after,
            frame.stream_window_before,
            frame.stream_window_after,
            frame.business_code,
        )
        for frame in engine.frames
    ]


class SchedulerTests(unittest.TestCase):
    def test_small_session_uses_round_robin(self) -> None:
        engine = Engine(load_session(ROOT / "samples/minimal.json")).run()
        self.assertEqual(engine.current_strategy, ROUND_ROBIN)

    def test_large_session_uses_proportional(self) -> None:
        engine = Engine(load_session(ROOT / "samples/scale_1000.json")).run()
        self.assertEqual(engine.current_strategy, PROPORTIONAL)
        self.assertGreater(engine.selection.params["stream_count"], 128)

    def test_explicit_strategy_event_pins_policy(self) -> None:
        spec = spec_session(
            streams=[{"stream_id": 1, "priority": 1, "request_body_bytes": 100000}],
            events=[strategy(0, "proportional")],
        )
        engine = Engine(spec).run()
        self.assertEqual(engine.current_strategy, PROPORTIONAL)
        self.assertEqual(engine.selection.mode, "pinned")
        self.assertEqual(engine.selection.reason_code, "explicit_strategy_event")

    def test_selection_contains_size_reason_and_tie_flag(self) -> None:
        engine = Engine(load_session(ROOT / "samples/minimal.json")).run()
        selection = engine.selection.to_dict()
        self.assertIn("reason", selection)
        self.assertIn("reason_code", selection)
        self.assertIn("stream_count", selection["params"])
        self.assertIn("pending_frames", selection["params"])
        self.assertIn("average_stream_window", selection["params"])
        self.assertIn("tie_affected", selection)
        self.assertTrue(selection["tie_affected"])

    def test_strategy_switch_records_expected_and_actual_deviations(self) -> None:
        engine = Engine(load_session(ROOT / "samples/burst_demo.json")).run()
        self.assertGreaterEqual(len(engine.strategy_switches), 1)
        for switch in engine.strategy_switches:
            self.assertNotEqual(switch.from_strategy, switch.to_strategy)
            for item in switch.stream_deviations:
                self.assertEqual(
                    item["deviation_bytes"],
                    item["actual_after_bytes"] - item["expected_before_bytes"],
                )

    def test_fairness_expected_bytes_reconcile_with_actual_data(self) -> None:
        engine = Engine(load_session(ROOT / "samples/burst_demo.json")).run()
        report = engine.fairness_report()
        expected = sum(item["expected_bytes"] for item in report)
        actual = sum(frame.data_bytes for frame in engine.frames)
        self.assertEqual(expected, actual)

    def test_scale_sample_detects_starvation_and_burst_preemption(self) -> None:
        engine = Engine(load_session(ROOT / "samples/scale_1000.json")).run()
        starvation = engine.starvation_warnings()
        preemption = engine.preemption_warnings()
        self.assertGreater(len(starvation), 0)
        self.assertGreater(len(preemption), 0)
        self.assertTrue(any(item["source_event_seq"] is not None for item in preemption))
        self.assertTrue(all(item["gap_rounds"] >= 5 for item in starvation))

    def test_repeated_runs_are_byte_for_byte_deterministic(self) -> None:
        first = Engine(load_session(ROOT / "samples/minimal.json")).run()
        second = Engine(load_session(ROOT / "samples/minimal.json")).run()
        self.assertEqual(first.run_digest(), second.run_digest())
        self.assertEqual(json.dumps(frame_signature(first)), json.dumps(frame_signature(second)))

    def test_diagnosis_is_independent_of_stream_declaration_order(self) -> None:
        raw = json.loads((ROOT / "samples/burst_demo.json").read_text(encoding="utf-8"))
        reversed_raw = copy.deepcopy(raw)
        reversed_raw["streams"] = list(reversed(reversed_raw["streams"]))
        first = Engine(parse_session(raw)).run()
        second = Engine(parse_session(reversed_raw)).run()
        self.assertEqual(frame_signature(first), frame_signature(second))
        self.assertEqual(first.starvation_warnings(), second.starvation_warnings())

    def test_event_object_key_order_does_not_change_content_digest(self) -> None:
        raw = raw_session(events=[wu(0, 1000), strategy(1, "proportional")])
        reordered = copy.deepcopy(raw)
        reordered["events"] = [
            {key: event[key] for key in reversed(list(event))} for event in reordered["events"]
        ]
        self.assertEqual(parse_session(raw).content_digest, parse_session(reordered).content_digest)

    def test_round_robin_respects_priority_order(self) -> None:
        spec = spec_session(
            streams=[
                {"stream_id": 1, "priority": 5, "request_body_bytes": 100000},
                {"stream_id": 3, "priority": 1, "request_body_bytes": 100000},
            ],
            max_frame_size=16384,
            scheduling_quantum=2,
            events=[wu(0, 0)],
        )
        engine = Engine(spec).run(drain=False)
        data = [frame for frame in engine.frames if frame.frame_type == "DATA"]
        self.assertEqual([frame.stream_id for frame in data[:2]], [3, 1])

    def test_proportional_allocates_frame_slots_by_window_ratio(self) -> None:
        spec = spec_session(
            streams=[
                {"stream_id": 1, "priority": 1, "request_body_bytes": 200000},
                {"stream_id": 3, "priority": 2, "request_body_bytes": 200000},
            ],
            initial_window=200000,
            events=[
                strategy(0, "proportional"),
                wu(1, 100000, stream_id=1),
            ],
            scheduling_quantum=4,
        )
        engine = Engine(spec)
        engine.step()
        engine.step()
        round_frames = [
            frame for frame in engine.frames if frame.frame_type == "DATA" and frame.event_seq == 1
        ]
        self.assertEqual([frame.stream_id for frame in round_frames], [1, 3, 1, 1])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from dialogwb.config import load_config, normalize_config
from dialogwb.edits import apply_batch, load_edit_batches
from dialogwb.engine import Engine, core_view
from dialogwb.lexicon import load_lexicon
from dialogwb.loader import load_jsonl, normalize_records
from dialogwb.workbench import ConfigMismatchError, Workbench


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lexicon = load_lexicon(SAMPLES / "lexicon.json")
        cls.config = load_config(SAMPLES / "config.json")
        cls.loaded = load_jsonl(SAMPLES / "events.jsonl", cls.lexicon)
        cls.batches = load_edit_batches(SAMPLES / "edits.json")

    def test_each_edit_batch_incremental_matches_full(self) -> None:
        seed_engine = Engine(
            config=self.config,
            lexicon=self.lexicon,
            loaded=self.loaded,
            mode="full",
        )
        baseline = seed_engine.run()
        full_baseline = Engine(
            config=self.config,
            lexicon=self.lexicon,
            loaded=self.loaded,
            mode="full",
        ).run()
        self.assertEqual(core_view(baseline), core_view(full_baseline))

        records = copy.deepcopy(self.loaded.records)
        config = copy.deepcopy(self.config)
        order = max(
            (int(item.get("_order", 0)) for item in records),
            default=0,
        )
        previous = seed_engine
        for batch in self.batches:
            records, config, edit_failures, order = apply_batch(
                records, config, batch, order
            )
            loaded = normalize_records(
                records,
                self.lexicon,
                parse_failures=self.loaded.parse_failures,
            )
            incremental = Engine(
                config=config,
                lexicon=self.lexicon,
                loaded=loaded,
                mode="incremental",
                seed_sessions=previous.export_seed(),
                initial_failures=edit_failures,
            ).run()
            full = Engine(
                config=config,
                lexicon=self.lexicon,
                loaded=loaded,
                mode="full",
                initial_failures=edit_failures,
            ).run()
            self.assertEqual(
                core_view(incremental),
                core_view(full),
                msg=f"mismatch after batch {batch['batch_id']}",
            )
            if batch["batch_id"] == "replace-and-reweight":
                s01_reuse = incremental["sessions"]["s01_plan"]["incremental"]
                s02_reuse = incremental["sessions"]["s02_delivery"]["incremental"]
                self.assertIn(1, s01_reuse["reused_stages"]["extraction"])
                self.assertIn(5, s01_reuse["reused_stages"]["extraction"])
                self.assertEqual(
                    s02_reuse["reused_rounds"],
                    list(range(1, 10)),
                )
            previous = Engine(
                config=config,
                lexicon=self.lexicon,
                loaded=loaded,
                mode="incremental",
                seed_sessions=previous.export_seed(),
                initial_failures=edit_failures,
            )
            self.assertEqual(
                core_view(previous.run()),
                core_view(incremental),
            )

    def test_add_in_middle_keeps_edited_records_contiguous(self) -> None:
        batch = {
            "batch_id": "middle-add",
            "operations": [
                {
                    "op": "add",
                    "session_id": "s00_short",
                    "turn": 2,
                    "speaker": "system",
                    "text": "团队确认交付。",
                    "timestamp": "2026-11-10T00:00:00+00:00",
                }
            ],
        }
        records, config, failures, order = apply_batch(
            copy.deepcopy(self.loaded.records),
            copy.deepcopy(self.config),
            batch,
            100,
        )
        loaded = normalize_records(
            records,
            self.lexicon,
            parse_failures=self.loaded.parse_failures,
        )
        self.assertEqual(failures, [])
        self.assertFalse(
            [
                failure
                for failure in loaded.failures
                if failure["session_id"] == "s00_short"
                and failure["code"] == "E_TURN_GAP"
            ]
        )
        self.assertEqual(
            [event["turn"] for event in loaded.events if event["session_id"] == "s00_short"],
            [1, 2, 3],
        )
        self.assertEqual(
            next(
                event["text"]
                for event in loaded.events
                if event["session_id"] == "s00_short" and event["turn"] == 2
            ),
            "团队确认交付。",
        )

    def test_concurrency_has_identical_core_and_work_units(self) -> None:
        serial = Engine(
            config=normalize_config({**self.config, "concurrency": 1}),
            lexicon=self.lexicon,
            loaded=self.loaded,
            mode="full",
        ).run()
        concurrent = Engine(
            config=normalize_config({**self.config, "concurrency": 4}),
            lexicon=self.lexicon,
            loaded=self.loaded,
            mode="full",
        ).run()
        self.assertEqual(
            core_view(serial, include_stage_timing=True),
            core_view(concurrent, include_stage_timing=True),
        )

    def test_candidate_order_window_eviction_and_truncation(self) -> None:
        result = Engine(
            config=self.config,
            lexicon=self.lexicon,
            loaded=self.loaded,
            mode="full",
        ).run()
        session = result["sessions"]["s01_plan"]
        round_four = session["rounds"][3]
        self.assertEqual(round_four["extraction"]["evicted_turns"], [1])
        candidates = round_four["extraction"]["candidates"]
        expected = sorted(
            candidates,
            key=lambda item: (
                -float(item["weight"]),
                int(item["source_turn"]),
                int(item["start"]),
            ),
        )
        self.assertEqual(candidates, expected)

        truncated = [
            entry
            for session_result in result["sessions"].values()
            for round_item in session_result["rounds"]
            for entry in round_item["summary_entries"]
            if entry["truncation"]["truncated"]
        ]
        self.assertTrue(truncated)
        for entry in truncated:
            detail = entry["truncation"]
            self.assertLessEqual(detail["kept_length"], self.config["max_summary_length"])
            self.assertEqual(
                detail["discarded_chars"],
                detail["original_length"] - detail["kept_length"],
            )
            self.assertIsNotNone(detail["truncation_position"])

    def test_stage_exception_blocks_later_references(self) -> None:
        def fault(stage: str, session_id: str, turn: int) -> None:
            if stage == "merge" and session_id == "s01_plan" and turn == 1:
                raise RuntimeError("injected merge failure")

        result = Engine(
            config=self.config,
            lexicon=self.lexicon,
            loaded=self.loaded,
            mode="full",
            fault_hook=fault,
        ).run()
        failures = [
            item
            for item in result["failures"]
            if item["code"] == "E_STAGE_EXCEPTION"
            and item["session_id"] == "s01_plan"
            and item["turn"] == 1
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["stage"], "merge")
        self.assertEqual(failures[0]["exception_type"], "RuntimeError")
        for round_item in result["sessions"]["s01_plan"]["rounds"]:
            if round_item["turn"] == 1:
                continue
            considered_turns = {
                item["turn"]
                for item in round_item["extraction"]["considered_events"]
            }
            source_turns = {
                source["turn"]
                for source in round_item["source_map"]
                for source in source["sources"]
            }
            self.assertNotIn(1, considered_turns)
            self.assertNotIn(1, source_turns)

    def test_snapshot_forced_exit_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = Engine(
                config=self.config,
                lexicon=self.lexicon,
                loaded=self.loaded,
                mode="full",
            ).run()
            partial = Workbench(
                events_path=SAMPLES / "events.jsonl",
                lexicon_path=SAMPLES / "lexicon.json",
                config_path=SAMPLES / "config.json",
                out_dir=root / "partial_out",
                snapshot_dir=root / "snapshot",
            )
            marker = partial.run(crash_after_snapshots=4)
            self.assertTrue(marker["__crash_requested__"])
            pointer = partial.current_snapshot_info()
            self.assertIsNotNone(pointer)
            self.assertEqual(pointer["status"], "running")
            self.assertEqual(pointer["snapshot_version"], 4)

            resumed = Workbench(
                events_path=SAMPLES / "events.jsonl",
                lexicon_path=SAMPLES / "lexicon.json",
                config_path=SAMPLES / "config.json",
                out_dir=root / "resumed_out",
                snapshot_dir=root / "snapshot",
            ).run(resume=True)
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["resumed_from"], 4)
            self.assertEqual(core_view(reference), core_view(resumed))

    def test_snapshot_config_mismatch_rejects_without_pointer_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbench = Workbench(
                events_path=SAMPLES / "events.jsonl",
                lexicon_path=SAMPLES / "lexicon.json",
                config_path=SAMPLES / "config.json",
                out_dir=root / "out",
                snapshot_dir=root / "snapshot",
            )
            workbench.run(crash_after_snapshots=3)
            before = workbench.current_snapshot_info()
            changed = json.loads((SAMPLES / "config.json").read_text(encoding="utf-8"))
            changed["compression_window"] += 1
            changed_path = root / "changed.json"
            changed_path.write_text(
                json.dumps(changed, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigMismatchError) as context:
                Workbench(
                    events_path=SAMPLES / "events.jsonl",
                    lexicon_path=SAMPLES / "lexicon.json",
                    config_path=changed_path,
                    out_dir=root / "mismatch_out",
                    snapshot_dir=root / "snapshot",
                ).run(resume=True)
            self.assertEqual(context.exception.code, "E_CONFIG_MISMATCH")
            after = workbench.current_snapshot_info()
            self.assertEqual(
                before["snapshot_version"], after["snapshot_version"]
            )

    def test_graceful_cancellation_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = Engine(
                config=self.config,
                lexicon=self.lexicon,
                loaded=self.loaded,
                mode="full",
            ).run()
            workbench = Workbench(
                events_path=SAMPLES / "events.jsonl",
                lexicon_path=SAMPLES / "lexicon.json",
                config_path=SAMPLES / "config.json",
                out_dir=root / "out",
                snapshot_dir=root / "snapshot",
            )
            interrupted = workbench.run(stop_after_rounds=3)
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertTrue((root / "out" / "result.json").exists())
            self.assertTrue((root / "out" / "report.json").exists())
            resumed = Workbench(
                events_path=SAMPLES / "events.jsonl",
                lexicon_path=SAMPLES / "lexicon.json",
                config_path=SAMPLES / "config.json",
                out_dir=root / "resumed_out",
                snapshot_dir=root / "snapshot",
            ).run(resume=True)
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(core_view(reference), core_view(resumed))


if __name__ == "__main__":
    unittest.main()

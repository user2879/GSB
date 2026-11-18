from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from session_pipeline.engine import compare_state_projection
from session_pipeline.loading import (
    load_scenario_base,
    normalize_machine,
    validate_events,
)
from session_pipeline.merge import merge_slot_update
from session_pipeline.runner import (
    projection_from_report,
    run_full_build,
    run_incremental,
)
from session_pipeline.util import fixed_to_string, stable_hash


SCENARIO = ROOT / "samples" / "scenario_01"


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="session-pipeline-tests-")
        self.workspace = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _copy_scenario(self) -> Path:
        destination = self.workspace / "scenario"
        shutil.copytree(SCENARIO, destination)
        return destination

    def test_loading_validation_keeps_other_events(self) -> None:
        base = load_scenario_base(SCENARIO)
        machine = base["machine"]
        config = base["config"]
        events = [
            {
                "session_id": "SX",
                "turn": 1,
                "role": "user",
                "text": "ok",
                "slot_updates": [],
                "timestamp_ms": 1000,
                "requested_target": "unknown-state",
            },
            {
                "session_id": "SX",
                "turn": 3,
                "role": "user",
                "text": "gap",
                "slot_updates": [],
                "timestamp_ms": 1100,
            },
            {
                "session_id": "SX",
                "turn": 2,
                "role": "user",
                "text": "clock goes backwards",
                "slot_updates": [],
                "timestamp_ms": 900,
            },
            {
                "session_id": "",
                "turn": 1,
                "role": "user",
                "text": "missing id",
                "slot_updates": [],
                "timestamp_ms": 1200,
            },
            {
                "session_id": "SX",
                "turn": 1,
                "role": "user",
                "text": "valid",
                "slot_updates": [],
                "timestamp_ms": 1300,
            },
        ]
        accepted, failures = validate_events(events, machine, config)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["turn"], 1)
        self.assertTrue(failures)
        self.assertEqual({failure["code"] for failure in failures}, {"E_INPUT_INVALID"})

    def test_fixed_seed_sample_generation_is_byte_deterministic(self) -> None:
        generated = self.workspace / "generated"
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_samples.py"), "--output", str(generated)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        names = [
            "events.jsonl",
            "state_machine.yaml",
            "pipeline_config.json",
            "edit_plan.json",
            "fault_injections.json",
            "meta.json",
        ]
        for name in names:
            self.assertEqual(
                (SCENARIO / name).read_bytes(),
                (generated / name).read_bytes(),
                msg=f"non-deterministic sample file: {name}",
            )

    def test_machine_rejects_unknown_transition_state_and_continues(self) -> None:
        base = load_scenario_base(SCENARIO)
        raw_machine = {
            "machine_id": "bad",
            "initial_state": "a",
            "states": ["a", "b"],
            "slots": {"x": {"type": "int", "merge": "overwrite"}},
            "merge_strategies": {"x": "overwrite"},
            "conflict_priority": ["explicit"],
            "transitions": [
                {"id": "BAD", "from": "a", "to": "missing"},
                {"id": "OK", "from": "a", "to": "b"},
            ],
        }
        machine, failures = normalize_machine(raw_machine, base["config"])
        self.assertEqual([rule["id"] for rule in machine["transitions"]], ["OK"])
        self.assertTrue(any("unknown target state" in failure["message"] for failure in failures))

    def test_merge_strategies_are_deterministic(self) -> None:
        base = load_scenario_base(SCENARIO)
        machine = base["machine"]
        config = base["config"]
        event = {
            "session_id": "S",
            "turn": 2,
            "role": "user",
            "text": "",
            "slot_updates": [],
            "timestamp_ms": 2,
        }
        budget_schema = machine["slots"]["budget"]
        old_budget = {
            "value": "100.00",
            "confidence_fixed": 500000,
            "confidence": fixed_to_string(500000, 6),
            "merge_strategy": "weighted",
            "source": "explicit",
            "updated_turn": 1,
            "history_hash": "",
            "cache_key": "",
        }
        merged, change = merge_slot_update(
            old_state=old_budget,
            update={"slot": "budget", "value": "200.00", "confidence": "0.5", "source": "explicit"},
            slot_name="budget",
            slot_schema=budget_schema,
            strategy="weighted",
            machine=machine,
            config=config,
            event=event,
            turn=2,
        )
        self.assertTrue(change["accepted"])
        self.assertEqual(merged["value"], "150.00")
        self.assertEqual(merged["confidence_fixed"], 750000)

        member_schema = machine["slots"]["member"]
        old_member = {
            "value": True,
            "confidence_fixed": 700000,
            "confidence": fixed_to_string(700000, 6),
            "merge_strategy": "reject_overwrite",
            "source": "explicit",
            "updated_turn": 1,
            "history_hash": "",
            "cache_key": "",
        }
        merged_member, member_change = merge_slot_update(
            old_state=old_member,
            update={"slot": "member", "value": False, "confidence": "0.9", "source": "inferred"},
            slot_name="member",
            slot_schema=member_schema,
            strategy="reject_overwrite",
            machine=machine,
            config=config,
            event=event,
            turn=2,
        )
        self.assertFalse(member_change["accepted"])
        self.assertTrue(merged_member["value"])

    def test_full_state_and_transition_trace(self) -> None:
        report = run_full_build(
            scenario_dir=SCENARIO,
            report_path=self.workspace / "full.json",
            serial=True,
        )
        self.assertEqual(report["summary"]["session_count"], 5)
        self.assertEqual(report["sessions"]["S001"]["current_state"], "ready")
        self.assertEqual(report["sessions"]["S002"]["current_state"], "closed")
        self.assertEqual(report["sessions"]["S004"]["current_state"], "closed")
        collecting = [
            item
            for item in report["sessions"]["S001"]["transition_trace"]
            if item["rule_id"] == "R030"
        ]
        self.assertEqual(collecting[0]["candidate_rule_ids"], ["R030", "R031"])
        self.assertGreaterEqual(report["summary"]["rejected_transition_count"], 3)
        self.assertEqual(report["sessions"]["S003"]["slots"]["member"]["value"], True)

    def test_every_incremental_batch_matches_full_rebuild(self) -> None:
        workdir = self.workspace / "incremental"
        base = load_scenario_base(SCENARIO)
        batch_count = len(base["edit_plan"]["batches"])
        previous = None
        for batch_number in range(batch_count + 1):
            incremental, _ = run_incremental(
                scenario_dir=SCENARIO,
                workdir=workdir,
                report_path=self.workspace / f"incremental-{batch_number}.json",
                reset=False,
                max_batch=batch_number,
            )
            full = run_full_build(
                scenario_dir=SCENARIO,
                report_path=self.workspace / f"full-{batch_number}.json",
                serial=True,
                max_edit_batches=batch_number,
            )
            self.assertEqual(
                projection_from_report(incremental),
                projection_from_report(full),
                msg=f"state mismatch at batch {batch_number}",
            )
            if previous is not None:
                reused = incremental["reuse"]["reused_sessions"]
                for session_id in reused:
                    before = projection_from_report(previous)[session_id]
                    after = projection_from_report(incremental)[session_id]
                    self.assertEqual(before, after)
                    self.assertEqual(
                        incremental["sessions"][session_id]["reuse"]["replayed_events"], 0
                    )
            previous = incremental

    def test_state_budget_exhaustion_is_explicit_and_not_truncated(self) -> None:
        scenario = self._copy_scenario()
        config_path = scenario / "pipeline_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["global_state_budget"] = 6
        config_path.write_text(json.dumps(config), encoding="utf-8")
        report = run_full_build(
            scenario_dir=scenario,
            report_path=self.workspace / "budget.json",
            serial=True,
        )
        self.assertEqual(report["budget"]["limit"], 6)
        self.assertEqual(report["budget"]["committed_sessions"], ["S001"])
        exhausted = report["budget"]["exhausted_sessions"]
        self.assertTrue(exhausted)
        for session_id in exhausted:
            session = report["sessions"][session_id]
            self.assertEqual(session["status"], "budget_exhausted")
            self.assertEqual(session["state_entries"], 0)
            self.assertIsNotNone(session["pending_cursor"])
            self.assertIn(
                "E_STATE_BUDGET_EXHAUSTED",
                {failure["code"] for failure in session["failures"]},
            )

    def test_session_quota_stops_only_the_offending_session(self) -> None:
        scenario = self._copy_scenario()
        config_path = scenario / "pipeline_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["session_state_quota"] = 2
        config_path.write_text(json.dumps(config), encoding="utf-8")
        report = run_full_build(
            scenario_dir=scenario,
            report_path=self.workspace / "quota.json",
            serial=True,
        )
        self.assertEqual(report["sessions"]["S001"]["status"], "quota_exceeded")
        self.assertIsNotNone(report["sessions"]["S001"]["pending_cursor"])
        self.assertTrue(
            any(
                failure["code"] == "E_SESSION_QUOTA_EXCEEDED"
                for failure in report["sessions"]["S001"]["failures"]
            )
        )
        self.assertGreaterEqual(report["summary"]["committed_session_count"], 1)

    def test_fault_injection_hits_are_reported(self) -> None:
        report, exit_code = run_incremental(
            scenario_dir=SCENARIO,
            workdir=self.workspace / "fault-work",
            report_path=self.workspace / "fault.json",
            reset=True,
            enable_injections=True,
        )
        self.assertEqual(exit_code, 0)
        injection_ids = {hit["injection_id"] for hit in report["injection_hits"]}
        self.assertEqual(
            injection_ids,
            {"INJ-UNKNOWN-TARGET", "INJ-ILLEGAL-SLOT", "INJ-TURN-JUMP", "INJ-QUOTA"},
        )
        codes = {failure["code"] for failure in report["failures"]}
        self.assertIn("E_INPUT_INVALID", codes)
        self.assertIn("E_SLOT_VALUE_INVALID", codes)
        self.assertIn("E_SESSION_QUOTA_EXCEEDED", codes)
        self.assertEqual(report["sessions"]["S005"]["status"], "quota_exceeded")

    def test_graceful_stop_finishes_current_batch_and_returns_zero(self) -> None:
        calls = {"count": 0}

        def stop_requested() -> bool:
            calls["count"] += 1
            return calls["count"] >= 2

        report, exit_code = run_incremental(
            scenario_dir=SCENARIO,
            workdir=self.workspace / "stop-work",
            report_path=self.workspace / "stop.json",
            reset=True,
            stop_requested=stop_requested,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["run_status"], "interrupted")
        self.assertEqual(report["batch_number"], 1)
        self.assertEqual(report["snapshot"]["completed_batch"], 1)

    def test_forced_termination_resumes_from_complete_snapshot(self) -> None:
        workdir = self.workspace / "crash-work"
        report_path = self.workspace / "crash-report.json"
        script = (
            "from pathlib import Path;"
            "from session_pipeline.runner import run_incremental;"
            f"run_incremental(scenario_dir=Path({str(SCENARIO)!r}),"
            f"workdir=Path({str(workdir)!r}),"
            f"report_path=Path({str(report_path)!r}),reset=True,crash_after_batch=1)"
        )
        child = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(child.returncode, 137)
        self.assertFalse(report_path.exists(), "forced termination must not promise an exit report")
        pointer = json.loads((workdir / "CURRENT").read_text(encoding="utf-8"))
        self.assertGreaterEqual(pointer["global_version"], 2)
        resumed, exit_code = run_incremental(
            scenario_dir=SCENARIO,
            workdir=workdir,
            report_path=self.workspace / "resumed.json",
            reset=False,
        )
        self.assertEqual(exit_code, 0)
        self.assertTrue(resumed["recovery"]["resumed"])
        self.assertEqual(resumed["recovery"]["recovery_point"]["completed_batch"], 1)
        full = run_full_build(
            scenario_dir=SCENARIO,
            report_path=self.workspace / "crash-full.json",
            serial=True,
        )
        self.assertEqual(projection_from_report(resumed), projection_from_report(full))

        snapshot = Path(resumed["snapshot"]["path"])
        manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["complete"])
        self.assertEqual(set(manifest["stages"]), {"loading", "advance", "summary"})
        self.assertTrue(all(stage["complete"] for stage in manifest["stages"].values()))
        self.assertEqual(
            manifest["file_hashes"]["state/session_states.json"],
            stable_hash(
                json.loads((snapshot / "state" / "session_states.json").read_text(encoding="utf-8"))
            ),
        )

    @unittest.skipIf(os.name == "nt", "Popen.terminate on Windows is not a catchable SIGTERM signal")
    def test_actual_sigterm_finishes_current_batch_with_exit_zero(self) -> None:
        workdir = self.workspace / "signal-work"
        report_path = self.workspace / "signal-report.json"
        script = (
            "import signal,time;"
            "from pathlib import Path;"
            "from session_pipeline import scheduler;"
            "from session_pipeline.runner import run_incremental;"
            "original=scheduler.evaluate_session;"
            "scheduler.evaluate_session=lambda *a,**k:(time.sleep(0.15),original(*a,**k))[1];"
            "STOP=False;"
            "handler=(lambda signum,frame:globals().__setitem__('STOP',True));"
            "signal.signal(signal.SIGTERM,handler);"
            f"run_incremental(scenario_dir=Path({str(SCENARIO)!r}),"
            f"workdir=Path({str(workdir)!r}),report_path=Path({str(report_path)!r}),"
            "reset=True,stop_requested=lambda:STOP)"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.time() + 15
        pointer = workdir / "CURRENT"
        while time.time() < deadline and process.poll() is None and not pointer.exists():
            time.sleep(0.01)
        self.assertIsNone(process.poll(), "child exited before the baseline snapshot")
        self.assertTrue(pointer.exists(), "baseline snapshot was not committed")
        time.sleep(0.25)
        process.terminate()
        stdout, stderr = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 0, msg=stdout + stderr)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["run_status"], "interrupted")
        self.assertGreaterEqual(report["batch_number"], 1)
        self.assertEqual(report["snapshot"]["completed_batch"], report["batch_number"])


if __name__ == "__main__":
    unittest.main()

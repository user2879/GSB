import json
import tempfile
import unittest
from pathlib import Path

from netdiag import samples
from netdiag.gaps import analyze_model
from netdiag.ledger import verify_entry, read_entries
from netdiag.merge import merge_sources
from netdiag.pipeline import (persist_session, restore_session,
                              run_diagnosis)
from netdiag.replay import replay_and_verify
from netdiag.snapshot import compare_versions
from netdiag.logging_utils import StructuredLogger


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        meta = samples.generate_workspace(self.root / "sample")
        self.capture = self.root / "sample" / "capture.bin"
        self.events = [self.root / "sample" / f"events_v{v}.json"
                       for v in (1, 2, 3)]

    def tearDown(self):
        self.tmp.cleanup()

    def test_replay_matches_conclusions(self):
        session = run_diagnosis(wire_paths=[str(self.capture)],
                                event_paths=[str(p) for p in self.events],
                                logger=StructuredLogger(stream=None))
        frames_by_flow = {f: list(o.frames)
                          for f, o in session.model.flows.items()}
        report = replay_and_verify(frames_by_flow, session.conclusions,
                                   port=41801)
        self.assertEqual(report.matched_gaps, 4)
        self.assertTrue(report.repeatable)

    def test_replay_tamper_raises(self):
        from netdiag.errors import ReplayMismatchError
        session = run_diagnosis(wire_paths=[str(self.capture)],
                                event_paths=[str(self.events[2])],
                                logger=StructuredLogger(stream=None))
        frames_by_flow = {f: list(o.frames)
                          for f, o in session.model.flows.items()}
        with self.assertRaises(ReplayMismatchError) as ctx:
            replay_and_verify(frames_by_flow, session.conclusions,
                              port=41802, tamper=True)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertIn("flow", ctx.exception.context)

    def test_snapshot_restore_and_append(self):
        logger = StructuredLogger(stream=None)
        session = run_diagnosis(wire_paths=[str(self.capture)],
                                event_paths=[str(p) for p in self.events],
                                logger=logger)
        snap = self.root / "snap.json"
        ledger = self.root / "ledger.jsonl"
        persist_session(session, snap, ledger, entry_id="e1")
        restored, report = restore_session(snap, logger=logger)
        self.assertFalse(report.replaced)
        restored.append_capture(self.capture)
        self.assertEqual(len(restored.model.flows), 2)

    def test_version_mismatch_triggers_full_replace(self):
        session = run_diagnosis(wire_paths=[str(self.capture)],
                                event_paths=[str(p) for p in self.events],
                                logger=StructuredLogger(stream=None))
        snap = self.root / "snap.json"
        persist_session(session, snap, self.root / "l.jsonl", entry_id="e2")
        data = json.loads(snap.read_text(encoding="utf-8"))
        data["model_version"] = "2.9.0"
        snap.write_text(json.dumps(data), encoding="utf-8")
        report = compare_versions(data)
        self.assertTrue(report.replaced)
        self.assertTrue(any("model_version" in d for d in
                            report.differences))

    def test_ledger_verification_roundtrip(self):
        logger = StructuredLogger(stream=None)
        session = run_diagnosis(wire_paths=[str(self.capture)],
                                event_paths=[str(p) for p in self.events],
                                logger=logger)
        ledger = self.root / "ledger.jsonl"
        persist_session(session, self.root / "snap.json", ledger,
                        entry_id="e3")
        entries = read_entries(ledger)
        result = verify_entry(entries[0], workspace=self.root,
                              logger=logger)
        self.assertTrue(result["verified"])

    def test_ledger_digest_tamper_detected(self):
        from netdiag.errors import ReplayMismatchError
        logger = StructuredLogger(stream=None)
        session = run_diagnosis(wire_paths=[str(self.capture)],
                                event_paths=[str(p) for p in self.events],
                                logger=logger)
        ledger = self.root / "ledger.jsonl"
        persist_session(session, self.root / "snap.json", ledger,
                        entry_id="e4")
        entries = read_entries(ledger)
        entries[0]["input_summary"]["wire"][0]["sha256"] = "0" * 64
        with self.assertRaises(ReplayMismatchError) as ctx:
            verify_entry(entries[0], workspace=self.root, logger=logger)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.context["reason"],
                         "digest_mismatch")


if __name__ == "__main__":
    unittest.main()

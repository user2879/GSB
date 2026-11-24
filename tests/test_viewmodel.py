from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from h2burst.model import load_session
from h2burst.viewmodel import WorkbenchViewModel


ROOT = Path(__file__).resolve().parents[1]


class ViewModelTests(unittest.TestCase):
    def test_render_is_stable_and_does_not_mutate_ledger(self) -> None:
        spec = load_session(ROOT / "samples/burst_demo.json")
        with tempfile.TemporaryDirectory() as temp:
            view_model = WorkbenchViewModel(spec, snapshot_db=Path(temp) / "ui.db")
            try:
                before = view_model.engine.run_digest()
                first = view_model.render_signature()
                second = view_model.render_signature()
                self.assertEqual(first, second)
                self.assertEqual(view_model.engine.run_digest(), before)
            finally:
                view_model.close()

    def test_rows_are_stably_sorted(self) -> None:
        spec = load_session(ROOT / "samples/burst_demo.json")
        with tempfile.TemporaryDirectory() as temp:
            view_model = WorkbenchViewModel(spec, snapshot_db=Path(temp) / "ui.db")
            try:
                frame_numbers = [row["frame"] for row in view_model.frame_rows()]
                stream_numbers = [row["stream"] for row in view_model.stream_rows()]
                self.assertEqual(frame_numbers, sorted(frame_numbers))
                self.assertEqual(stream_numbers, sorted(stream_numbers))
                self.assertEqual(
                    [row["index"] for row in view_model.transition_rows()],
                    list(range(1, len(view_model.engine.transitions) + 1)),
                )
            finally:
                view_model.close()

    def test_step_snapshot_restore_round_trip(self) -> None:
        spec = load_session(ROOT / "samples/minimal.json")
        with tempfile.TemporaryDirectory() as temp:
            view_model = WorkbenchViewModel(
                spec,
                snapshot_db=Path(temp) / "ui.db",
                auto_run=False,
            )
            try:
                view_model.step()
                view_model.step()
                checkpoint_frames = len(view_model.engine.frames)
                info = view_model.save_snapshot("ui-checkpoint")
                view_model.run()
                self.assertGreaterEqual(len(view_model.engine.frames), checkpoint_frames)
                restored = view_model.restore_snapshot(info.snapshot_id)
                self.assertFalse(restored.whole_replacement)
                self.assertEqual(len(view_model.engine.frames), checkpoint_frames)
                self.assertEqual(len(view_model.snapshot_rows()), 1)
            finally:
                view_model.close()

    def test_snapshot_rows_show_read_only_status(self) -> None:
        spec = load_session(ROOT / "samples/minimal.json")
        with tempfile.TemporaryDirectory() as temp:
            view_model = WorkbenchViewModel(spec, snapshot_db=Path(temp) / "ui.db")
            try:
                info = view_model.save_snapshot("readonly-check")
                rows = view_model.snapshot_rows()
                self.assertTrue(any(row["id"] == info.snapshot_id and row["read_only"] == "no" for row in rows))
            finally:
                view_model.close()

    def test_status_exposes_strategy_windows_and_next_event(self) -> None:
        spec = load_session(ROOT / "samples/minimal.json")
        with tempfile.TemporaryDirectory() as temp:
            view_model = WorkbenchViewModel(spec, snapshot_db=Path(temp) / "ui.db")
            try:
                status = view_model.status_rows()
                self.assertIn("connection_window", status)
                self.assertIn("selection", status)
                self.assertIn("next_event_seq", status)
                self.assertEqual(status["status_code"], "OK")
            finally:
                view_model.close()


if __name__ == "__main__":
    unittest.main()

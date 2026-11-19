import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from subtitle_workbench.controller import WorkbenchController  # noqa: E402
from subtitle_workbench.gui import TimelineWidget, WorkbenchWindow  # noqa: E402
from subtitle_workbench.layout import solve_layout  # noqa: E402
from subtitle_workbench.sample import generate_sample  # noqa: E402
from subtitle_workbench.snapshot import (POINTER_NAME, SnapshotStore,  # noqa: E402
                                         StaleVersionError)

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class GuiInteractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(Path(self.tmp.name))
        project = generate_sample()
        self.project = project
        self.store.commit(project.items, solve_layout(project.items))
        self.controller = WorkbenchController(self.store)
        self.widget = TimelineWidget(self.controller)
        self.widget.resize(1600, 700)
        self.window = WorkbenchWindow(self.controller)

    def tearDown(self):
        self.tmp.cleanup()

    def _cue_point(self, item_id, *, edge=None):
        snapshot = self.store.load_current()
        cue = next(c for c in snapshot.layout.cues
                   if c.item.item_id == item_id)
        x = self.widget._frame_to_x(
            cue.item.end_frame - 4 if edge == "end"
            else cue.item.start_frame + 2 if edge == "start"
            else (cue.item.start_frame + cue.item.end_frame) // 2)
        y = 40 + cue.lane_index * 64 + 20
        return QtCore.QPoint(x, y)

    def test_lane_drag_preview_then_commit(self):
        item_id = self.project.items[0].item_id
        point = self._cue_point(item_id)
        QtCore.QTimer.singleShot(0, lambda: None)
        self.widget.mousePressEvent(
            QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, point,
                              QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
                              QtCore.Qt.NoModifier))
        target_y = 40 + 5 * 64 + 20
        move = QtGui.QMouseEvent(QtCore.QEvent.MouseMove,
                                 QtCore.QPoint(point.x(), target_y),
                                 QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
                                 QtCore.Qt.NoModifier)
        self.widget.mouseMoveEvent(move)
        self.assertIsNotNone(self.widget._preview)
        self.assertEqual(self.widget._preview.target_lane, 5)
        self.assertIsNone(self.widget.last_error)
        self.assertTrue(self.widget.commit_drag())
        self.assertEqual(self.store.load_current().version, 2)

    def test_illegal_drag_paints_rejection_and_keeps_state(self):
        item_id = self.project.items[0].item_id
        point = self._cue_point(item_id)
        self.widget.mousePressEvent(
            QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, point,
                              QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
                              QtCore.Qt.NoModifier))
        # drag far below every lane -> E_DRAG_BOUNDS
        bad = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPoint(point.x(), 40 + 40 * 64),
            QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier)
        self.widget.mouseMoveEvent(bad)
        self.assertIsNotNone(self.widget.last_error)
        self.assertEqual(self.widget.last_error.code, "E_DRAG_BOUNDS")
        self.assertFalse(self.widget.commit_drag())
        self.widget.cancel_drag()
        self.assertEqual(self.store.load_current().version, 1)

    def test_escape_cancels_and_restores_geometry(self):
        item_id = self.project.items[0].item_id
        point = self._cue_point(item_id)
        self.widget.mousePressEvent(
            QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, point,
                              QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
                              QtCore.Qt.NoModifier))
        self.widget.mouseMoveEvent(QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPoint(point.x(), 40 + 3 * 64 + 20),
            QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier))
        before = [item.to_dict() for item in self.controller.committed_items]
        self.widget.keyPressEvent(
            QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape,
                            QtCore.Qt.NoModifier))
        self.assertIsNone(self.widget._preview)
        self.assertEqual(
            before, [item.to_dict()
                     for item in self.controller.committed_items])

    def test_end_edge_drag_updates_interval(self):
        item_id = self.project.items[0].item_id
        point = self._cue_point(item_id, edge="end")
        self.widget.mousePressEvent(
            QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, point,
                              QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
                              QtCore.Qt.NoModifier))
        snapshot = self.store.load_current()
        cue = next(c for c in snapshot.layout.cues
                   if c.item.item_id == item_id)
        new_x = self.widget._frame_to_x(cue.item.end_frame + 25)
        expected_end = self.widget._x_to_frame(new_x)
        self.widget.mouseMoveEvent(QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, QtCore.QPoint(new_x, point.y()),
            QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier))
        self.assertEqual(self.widget._preview.candidate.end_frame,
                         expected_end)
        self.widget.cancel_drag()


class RecoveryTests(unittest.TestCase):
    def test_reload_after_simulated_kill_uses_last_commit_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(tmp)
            project = generate_sample()
            v1 = store.commit(project.items, solve_layout(project.items))
            moved = list(project.items)
            first = moved[0]
            moved[0] = type(first)(first.item_id, first.start_frame + 5,
                                   first.end_frame + 5, first.text,
                                   first.speaker, first.language,
                                   first.priority, first.is_ad)
            v2 = store.commit(tuple(moved), solve_layout(tuple(moved)))
            # simulate a fresh process after forced termination
            rebooted = SnapshotStore(tmp).load_current()
            self.assertEqual(rebooted.version, v2.version)
            self.assertEqual(rebooted.layout_hash, v2.layout_hash)
            self.assertEqual(rebooted.items, tuple(moved))

    def test_missing_snapshot_file_never_fabricates_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(tmp)
            project = generate_sample()
            store.commit(project.items, solve_layout(project.items))
            os.remove(store._snapshot_path(1))
            self.assertIsNone(SnapshotStore(tmp).load_current())

    def test_pointer_hash_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(tmp)
            project = generate_sample()
            store.commit(project.items, solve_layout(project.items))
            pointer = Path(tmp) / POINTER_NAME
            data = json.loads(pointer.read_text())
            data["layout_hash"] = "0" * 64
            pointer.write_text(json.dumps(data, sort_keys=True),
                               encoding="utf-8")
            with self.assertRaises(StaleVersionError):
                SnapshotStore(tmp).load_current()


if __name__ == "__main__":
    unittest.main()

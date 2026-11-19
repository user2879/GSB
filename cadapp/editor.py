"""Control point editing with snapshot-based undo.

Each edit records a deep snapshot of the affected patch's control point
grid plus an operation-log entry used by the version-comparison
attribution.  Undo restores the snapshot and invalidates exactly one
patch's display cache, so the model hash and the rendered image return
bit-for-bit to their pre-edit state.
"""

from __future__ import annotations

import copy

from .errors import CadError, ErrorCode


class EditorSession:
    def __init__(self, model, display=None):
        self.model = model
        self.display = display
        self._undo_stack: list[tuple[str, list]] = []
        self.op_log: list[dict] = []

    def edit_control_point(self, patch_id: str, i: int, j: int, new_value) -> None:
        patch = self.model.patch_by_id(patch_id)
        if not (0 <= i < len(patch.control_points)):
            raise CadError(ErrorCode.E_FORMAT_HEADER,
                           f"control point row {i} out of range",
                           f"{patch_id}.control_points[{i}]")
        if not (0 <= j < len(patch.control_points[0])):
            raise CadError(ErrorCode.E_FORMAT_HEADER,
                           f"control point column {j} out of range",
                           f"{patch_id}.control_points[{i}][{j}]")
        new_value = [float(c) for c in new_value]
        if len(new_value) != 3:
            raise CadError(ErrorCode.E_FORMAT_HEADER,
                           "control point must have 3 coordinates",
                           f"{patch_id}.control_points[{i}][{j}]")
        self._undo_stack.append((patch_id, copy.deepcopy(patch.control_points)))
        patch.control_points[i][j] = new_value
        self.op_log.append({"op": "edit", "patch": patch_id, "index": [i, j],
                            "new_value": new_value})
        if self.display is not None:
            self.display.invalidate(patch_id)

    def undo(self) -> None:
        if not self._undo_stack:
            raise CadError(ErrorCode.E_UNDO_EMPTY,
                           "undo requested with an empty undo stack",
                           "undo_stack")
        patch_id, snapshot = self._undo_stack.pop()
        patch = self.model.patch_by_id(patch_id)
        patch.control_points = snapshot
        self.op_log.append({"op": "undo", "patch": patch_id})
        if self.display is not None:
            self.display.invalidate(patch_id)

    @property
    def undo_depth(self) -> int:
        return len(self._undo_stack)

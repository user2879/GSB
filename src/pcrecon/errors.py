"""Business error codes for the point-cloud reconstruction application.

Every domain failure carries a stable string code so that reports and the
diagnostics entry point can refer to failures deterministically.
"""
from __future__ import annotations


class PcreconError(Exception):
    code = "E_UNKNOWN"

    def __init__(self, message: str = "", *, context: dict | None = None):
        super().__init__(message or self.code)
        self.message = message or self.code
        self.context = dict(context or {})

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "context": self.context}


class PointCloudFormatError(PcreconError):
    code = "E_POINT_CLOUD_FORMAT"


class NeighborRadiusError(PcreconError):
    code = "E_NEIGHBOR_RADIUS"


class SnapshotStaleError(PcreconError):
    code = "E_SNAPSHOT_STALE"


class MeshMismatchError(PcreconError):
    code = "E_MESH_MISMATCH"


class BudgetExceededError(PcreconError):
    code = "E_BUDGET_EXCEEDED"


class UndoEmptyError(PcreconError):
    code = "E_UNDO_EMPTY"


class PatchDependencyError(PcreconError):
    code = "E_PATCH_DEPENDENCY"


class GLContextError(PcreconError):
    code = "E_GL_CONTEXT"


CODE_TO_CLASS = {
    cls.code: cls
    for cls in (
        PointCloudFormatError,
        NeighborRadiusError,
        SnapshotStaleError,
        MeshMismatchError,
        BudgetExceededError,
        UndoEmptyError,
        PatchDependencyError,
        GLContextError,
    )
}

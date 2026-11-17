"""Business error codes and the pipeline's deterministic exception type."""

E_CAMERA_INVALID = "E_CAMERA_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_PNG_INVALID = "E_PNG_INVALID"
E_VIEW_MISSING = "E_VIEW_MISSING"
E_MATCH_TOO_FEW_INLIERS = "E_MATCH_TOO_FEW_INLIERS"
E_PARALLAX_INVALID = "E_PARALLAX_INVALID"
E_FEATURE_NONFINITE = "E_FEATURE_NONFINITE"
E_VOXEL_INVALID = "E_VOXEL_INVALID"
E_SLOT_BUDGET_EXHAUSTED = "E_SLOT_BUDGET_EXHAUSTED"
E_POINT_BUDGET_EXHAUSTED = "E_POINT_BUDGET_EXHAUSTED"
E_GROUP_FAILED = "E_GROUP_FAILED"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"


class PipelineError(Exception):
    """An exception carrying a stable business code and structured context."""

    def __init__(self, code, message, context=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = dict(context or {})

    def to_record(self):
        record = {"code": self.code, "message": self.message}
        record.update(self.context)
        return record

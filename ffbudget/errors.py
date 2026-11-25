"""Business error codes for the feature-freshness budget component."""
from __future__ import annotations

E_UNKNOWN_LAYER = "E_UNKNOWN_LAYER"
E_DUPLICATE_SLOT_KEY = "E_DUPLICATE_SLOT_KEY"
E_BAD_RETENTION = "E_BAD_RETENTION"
E_BAD_DIMENSION = "E_BAD_DIMENSION"
E_BATCH_SEQ_NOT_STRICT = "E_BATCH_SEQ_NOT_STRICT"
E_BAD_EFFECTIVE_MS = "E_BAD_EFFECTIVE_MS"
E_REV_SEQ_NOT_MONOTONIC = "E_REV_SEQ_NOT_MONOTONIC"
E_BAD_POINTS = "E_BAD_POINTS"
E_UNKNOWN_SLOT = "E_UNKNOWN_SLOT"
E_VECTOR_DIM_MISMATCH = "E_VECTOR_DIM_MISMATCH"
E_DUPLICATE_BATCH_DELIVERY = "E_DUPLICATE_BATCH_DELIVERY"
E_REV_SEQ_LOWER = "E_REV_SEQ_LOWER"
E_EFFECTIVE_EARLIER = "E_EFFECTIVE_EARLIER"
E_BATCH_SEQ_LOWER = "E_BATCH_SEQ_LOWER"
E_TIE_DIGEST = "E_TIE_DIGEST"
E_SCORE_BUDGET_PENDING = "E_SCORE_BUDGET_PENDING"
E_SLOT_QUOTA_EXCEEDED = "E_SLOT_QUOTA_EXCEEDED"
E_REVISION_PRUNED = "E_REVISION_PRUNED"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"


class BusinessFailure(Exception):
    """A mapped business validation failure (recorded, never raised silently)."""

    def __init__(self, code: str, message: str, context: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}

    def as_record(self) -> dict:
        record = {"code": self.code, "message": self.message}
        record.update(self.context)
        return record

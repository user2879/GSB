"""Business error codes and typed exceptions.

Error code registry (frozen, part of the public spec):

    E_FRAME_FORMAT     frame file shape / dtype / fp16 decode is invalid
    E_STATE_VERSION    processing state / ledger version is unsupported
    E_HASH_MISMATCH    recorded content hash disagrees with recomputed data
    E_CLOSURE_DIVERGE  actual recomputed set differs from the invalidation closure
    E_INJECT_BOUNDS    an injected parameter or value is outside its legal domain
    E_DETECT_MISS      a fault that should have been detected was missed
    E_SNAPSHOT_STALE   recovery snapshot is older than the last committed state
    E_TONEMAP_DOMAIN   tone mapping input is not finite or outside the clamp domain
"""


class ErrorCode:
    FRAME_FORMAT = "E_FRAME_FORMAT"
    STATE_VERSION = "E_STATE_VERSION"
    HASH_MISMATCH = "E_HASH_MISMATCH"
    CLOSURE_DIVERGE = "E_CLOSURE_DIVERGE"
    INJECT_BOUNDS = "E_INJECT_BOUNDS"
    DETECT_MISS = "E_DETECT_MISS"
    SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
    TONEMAP_DOMAIN = "E_TONEMAP_DOMAIN"

    ALL = frozenset(
        [
            FRAME_FORMAT,
            STATE_VERSION,
            HASH_MISMATCH,
            CLOSURE_DIVERGE,
            INJECT_BOUNDS,
            DETECT_MISS,
            SNAPSHOT_STALE,
            TONEMAP_DOMAIN,
        ]
    )


class PipelineError(Exception):
    """Base class for all domain errors; carries a fixed business error code."""

    code = "E_PIPELINE"

    def __init__(self, message, *, code=None, details=None):
        super().__init__(message)
        if code is not None:
            self.code = code
        self.details = dict(details or {})

    def to_dict(self):
        return {"code": self.code, "message": str(self), "details": self.details}


class FrameFormatError(PipelineError):
    code = ErrorCode.FRAME_FORMAT


class StateVersionError(PipelineError):
    code = ErrorCode.STATE_VERSION


class HashMismatchError(PipelineError):
    code = ErrorCode.HASH_MISMATCH


class ClosureDivergeError(PipelineError):
    code = ErrorCode.CLOSURE_DIVERGE


class InjectBoundsError(PipelineError):
    code = ErrorCode.INJECT_BOUNDS


class DetectMissError(PipelineError):
    code = ErrorCode.DETECT_MISS


class SnapshotStaleError(PipelineError):
    code = ErrorCode.SNAPSHOT_STALE


class TonemapDomainError(PipelineError):
    code = ErrorCode.TONEMAP_DOMAIN

"""slowprof - local slow-request stack profiling and hotspot localization."""

from .errors import (
    LedgerError,
    ProfilerError,
    RangeError,
    ReplayError,
    ReplayMismatch,
    SampleError,
)
from .folding import (
    DEFAULT_MAX_TOTAL_WEIGHT,
    DEFAULT_MAX_UNIQUE_STACKS,
    FoldResult,
    cross_check,
    fold_adaptive,
    fold_full,
    fold_streaming,
)
from .fixedpoint import QuantParams, build_share_report
from .ranking import FrameRank, build_contexts, rank_frames
from .samples import Interval, Sample, iter_samples, load_samples, make_interval
from .session import ProfileSession, SessionConfig

__all__ = [
    "ProfilerError",
    "SampleError",
    "RangeError",
    "LedgerError",
    "ReplayError",
    "ReplayMismatch",
    "Interval",
    "Sample",
    "iter_samples",
    "load_samples",
    "make_interval",
    "FoldResult",
    "fold_full",
    "fold_streaming",
    "fold_adaptive",
    "cross_check",
    "QuantParams",
    "build_share_report",
    "FrameRank",
    "rank_frames",
    "build_contexts",
    "ProfileSession",
    "SessionConfig",
    "DEFAULT_MAX_UNIQUE_STACKS",
    "DEFAULT_MAX_TOTAL_WEIGHT",
]

__version__ = "1.0.0"

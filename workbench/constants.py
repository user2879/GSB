"""Stable identifiers and declared measurement semantics."""

SCHEMA_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1
ESTIMATOR_VERSION = "fixed-window-v1"
REPLAY_MODEL_VERSION = "batch-barrier-v1"

CLOCK_BASIS = (
    "virtual monotonic clock (integer nanoseconds); clock_base_ms is the "
    "session origin for both master commit timestamps and replica completion"
)
TRUTH_SOURCE = (
    "per-record replay completion persisted in replay_samples.replay_completed_ns"
)
TRUNCATION_DIRECTION = "floor"
P95_METHOD = "nearest_rank_ceil"

SEGMENT_ORDER = (
    "sampling_truncation",
    "fixed_point_truncation",
    "replay_rate_discretization",
)

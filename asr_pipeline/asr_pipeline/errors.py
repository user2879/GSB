"""Business error codes and the pipeline exception type.

Every failure captured by the pipeline is expressed with one of the stable
business codes below so that diagnostic reports and exit codes stay
machine-readable across runs.
"""

E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_SHARD_BUDGET_EXHAUSTED = "E_SHARD_BUDGET_EXHAUSTED"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_DECODE_FAILED = "E_DECODE_FAILED"

ALL_CODES = (
    E_AUDIO_FORMAT,
    E_CONFIG_INVALID,
    E_SHARD_BUDGET_EXHAUSTED,
    E_POINT_BUDGET_PENDING,
    E_SNAPSHOT_STALE,
    E_DECODE_FAILED,
)


class PipelineError(Exception):
    """Raised by stage functions; carries a business code and stage name."""

    def __init__(self, code, message, stage=None, injection_id=None):
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.injection_id = injection_id


def failure_record(audio, shard_index, stage, code, message, injection_id=None):
    """Build a canonical failure record (stable key order, JSON friendly)."""
    return {
        "audio": audio,
        "shard_index": shard_index,
        "stage": stage,
        "code": code,
        "message": message,
        "injection_id": injection_id,
        "sort_key": (audio, shard_index if shard_index is not None else -1, stage or ""),
    }

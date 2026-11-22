"""Business error codes and the :class:`PipelineError` carrier.

Error codes are stable strings because they appear in failure lists, diagnostic
reports and snapshot manifests; tests and external tooling rely on them.
"""

ERROR_CODES = {
    "E_STREAM_INVALID": "event stream / structural validation failure",
    "E_FSM_INVALID": "state machine definition validation failure",
    "E_CONFIG_INVALID": "pipeline configuration validation failure",
    "E_SLOT_INVALID": "slot value failed type / range validation",
    "E_TRANSITION_REJECTED": "transition not permitted by the rule table",
    "E_QUOTA_EXCEEDED": "per-session state entry quota exhausted",
    "E_BUDGET_PENDING": "global state entry budget exhausted, work retained",
    "E_SNAPSHOT_STALE": "snapshot version or configuration hashes do not match",
}


class PipelineError(Exception):
    """Raised inside a processing stage; the runner converts it into a
    :class:`~dialog_state_pipeline.models.FailureRecord` and keeps going."""

    def __init__(self, code, message, stage, injection_id=None):
        if code not in ERROR_CODES:  # pragma: no cover - defensive
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.injection_id = injection_id


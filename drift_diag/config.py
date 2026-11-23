"""Session-level deterministic evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import format_error


@dataclass(frozen=True)
class EvaluationConfig:
    scale_exp: int = 3
    scale_version: str = "fixed-point-v1"
    tolerance_num: int = 0
    tolerance_den: int = 1
    global_limit: int = 3
    per_workload_quota: int = 1
    task_delay_ms: float = 0.0
    batch_events: int = 4

    def validate(self) -> "EvaluationConfig":
        if isinstance(self.scale_exp, bool) or not isinstance(self.scale_exp, int):
            raise format_error("scale_exp must be an integer", field="scale_exp")
        if not 0 <= self.scale_exp <= 18:
            raise format_error("scale_exp must be between 0 and 18", field="scale_exp")
        if not isinstance(self.scale_version, str) or not self.scale_version:
            raise format_error("scale_version must be a non-empty string", field="scale_version")
        if (
            isinstance(self.tolerance_num, bool)
            or not isinstance(self.tolerance_num, int)
            or isinstance(self.tolerance_den, bool)
            or not isinstance(self.tolerance_den, int)
        ):
            raise format_error("tolerance numerator and denominator must be integers", field="tolerance")
        if self.tolerance_num < 0 or self.tolerance_den <= 0:
            raise format_error(
                "tolerance must be a non-negative rational with positive denominator",
                field="tolerance",
            )
        if isinstance(self.global_limit, bool) or not isinstance(self.global_limit, int) or self.global_limit <= 0:
            raise format_error("global_limit must be a positive integer", field="global_limit")
        if (
            isinstance(self.per_workload_quota, bool)
            or not isinstance(self.per_workload_quota, int)
            or self.per_workload_quota <= 0
        ):
            raise format_error("per_workload_quota must be a positive integer", field="per_workload_quota")
        if isinstance(self.task_delay_ms, bool) or not isinstance(self.task_delay_ms, (int, float)):
            raise format_error("task_delay_ms must be numeric", field="task_delay_ms")
        if self.task_delay_ms < 0:
            raise format_error("task_delay_ms cannot be negative", field="task_delay_ms")
        if isinstance(self.batch_events, bool) or not isinstance(self.batch_events, int) or self.batch_events <= 0:
            raise format_error("batch_events must be a positive integer", field="batch_events")
        return self

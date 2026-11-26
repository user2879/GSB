from __future__ import annotations


class StageInputError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        business_code: str = "E_SCHEMA_INVALID",
        stage_code: str = "STAGE_INPUT_INVALID",
        injection_id: str | None = None,
        fault_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.business_code = business_code
        self.stage_code = stage_code
        self.injection_id = injection_id
        self.fault_type = fault_type


class BudgetExceeded(RuntimeError):
    pass

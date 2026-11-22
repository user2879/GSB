"""Stable error codes emitted by the pipeline."""

E_BASE_INVALID = "E_BASE_INVALID"
E_TASK_INVALID = "E_TASK_INVALID"
E_QUANT_INVALID = "E_QUANT_INVALID"
E_BASE_VERSION_MISMATCH = "E_BASE_VERSION_MISMATCH"
E_IRREVERSIBLE_MERGE = "E_IRREVERSIBLE_MERGE"
E_MERGE_BUDGET_EXHAUSTED = "E_MERGE_BUDGET_EXHAUSTED"
E_MERGE_PENDING = "E_MERGE_PENDING"


def failure(code, reason, task=None, layer=None):
    return {"code": code, "task": task, "layer": layer, "reason": reason}

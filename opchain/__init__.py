"""Deterministic incremental activation/permutation operator-chain editor."""

from .chain import Chain, Node, parse_chain
from .config import ExecutionConfig, parse_config
from .edits import apply_edit, apply_edit_files
from .engine import OperatorChainEngine, compare_execution_artifacts
from .errors import (
    E_ACTIVATION_FAILED,
    E_CHAIN_INVALID,
    E_CYCLE_DETECTED,
    E_INPLACE_CONFLICT,
    E_PERM_INVALID,
    E_PERMUTE_FAILED,
    E_RECOMPUTE_BUDGET_PENDING,
    E_SNAPSHOT_STALE,
    BusinessError,
)
from .input_spec import InputSpec, generate_input, parse_input_spec

__all__ = [
    "BusinessError",
    "Chain",
    "ExecutionConfig",
    "InputSpec",
    "Node",
    "OperatorChainEngine",
    "apply_edit",
    "apply_edit_files",
    "compare_execution_artifacts",
    "generate_input",
    "parse_chain",
    "parse_config",
    "parse_input_spec",
    "E_ACTIVATION_FAILED",
    "E_CHAIN_INVALID",
    "E_CYCLE_DETECTED",
    "E_INPLACE_CONFLICT",
    "E_PERM_INVALID",
    "E_PERMUTE_FAILED",
    "E_RECOMPUTE_BUDGET_PENDING",
    "E_SNAPSHOT_STALE",
]


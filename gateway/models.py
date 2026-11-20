"""Domain model for the inference gateway routing diagnostics console."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Explicit, documented bounds for validated integer fields.
MIN_PORT = 1024
MAX_PORT = 65535
MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 64
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 256
MIN_WEIGHT_BP = 0
MAX_WEIGHT_BP = 10000
LOOPBACK_HOST = "127.0.0.1"

# Ledger integer format version (bump on schema change).
LEDGER_FORMAT_VERSION = 1

# Explicit tie-break threshold (in estimated batch units) for strategy choice.
STRATEGY_TIE_THRESHOLD = 1


@dataclass(frozen=True)
class Backend:
    model_id: str
    host: str
    port: int
    params: Tuple[float, ...]
    max_concurrency: int
    batch_size: int


@dataclass(frozen=True)
class RouteRule:
    rule_id: str
    prefix: str
    stable_backend: str
    canary_backend: str
    canary_weight_bp: int
    batch_size: int


@dataclass(frozen=True)
class Request:
    seq: int
    client_key: str
    inputs: Tuple[float, ...]
    expected_stage: int


@dataclass(frozen=True)
class Declaration:
    """A fully validated set of backends, rules and requests."""

    backends: Tuple[Backend, ...]          # canonical order: sorted by model_id
    rules: Tuple[RouteRule, ...]           # canonical order: sorted by rule_id
    requests: Tuple[Request, ...]          # canonical order: sorted by seq
    param_dim: int


@dataclass(frozen=True)
class RoutedRequest:
    request: Request
    rule_id: str
    backend_id: str
    bucket: int
    variant: str  # "stable" | "canary"


@dataclass(frozen=True)
class Batch:
    batch_id: str          # deterministic: f"{backend_id}#{index:04d}"
    backend_id: str
    rule_id: str
    batch_index: int
    request_seqs: Tuple[int, ...]


@dataclass(frozen=True)
class BatchResult:
    batch: Batch
    fingerprint: str
    wait_start_ms: int
    wait_end_ms: int
    start_ms: int
    end_ms: int
    duration_ms: int
    output_digest: str
    attempt: int

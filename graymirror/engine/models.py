"""Typed in memory models for rules and request stream entries."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    rule_id: str
    prefix: str
    stable_target: str
    gray_target: str
    gray_weight: int          # per-10000 integer, 0..10000
    mirror_rate: int          # per-10000 integer, 0..10000
    gray_retries: int         # non negative integer, retries for gray target
    mirror_target: str
    mirror_timeout_ms: int    # non negative integer
    source_line: int          # physical line in the rule JSON file


@dataclass(frozen=True)
class Request:
    request_id: str
    path: str
    client_key: str
    timestamp_ms: int
    source_line: int         # physical line in the NDJSON stream

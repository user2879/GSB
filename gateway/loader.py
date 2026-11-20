"""Loading and strict validation of backends, rules (JSON) and requests (NDJSON).

Any violation rejects the whole input and reports the first offending field
path in dotted/indexed form, e.g. ``backends[1].port``.
"""

from __future__ import annotations

import json
import math
from typing import List

from .errors import ValidationError
from .models import (
    LOOPBACK_HOST,
    MAX_BATCH_SIZE,
    MAX_CONCURRENCY,
    MAX_PORT,
    MAX_WEIGHT_BP,
    MIN_BATCH_SIZE,
    MIN_CONCURRENCY,
    MIN_PORT,
    MIN_WEIGHT_BP,
    Backend,
    Declaration,
    Request,
    RouteRule,
)


def _require(cond: bool, path: str, message: str) -> None:
    if not cond:
        raise ValidationError(path, message)


def _require_str(value, path: str) -> str:
    _require(isinstance(value, str) and value != "", path, "expected non-empty string")
    return value


def _require_int(value, path: str, lo: int, hi: int) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), path,
             "expected integer")
    _require(lo <= value <= hi, path, f"expected integer in [{lo}, {hi}]")
    return value


def _require_finite_float(value, path: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             path, "expected number")
    fvalue = float(value)
    _require(math.isfinite(fvalue), path, "expected finite number")
    return fvalue


def _parse_vector(value, path: str) -> tuple:
    _require(isinstance(value, list) and len(value) > 0, path,
             "expected non-empty array of finite numbers")
    return tuple(
        _require_finite_float(item, f"{path}[{i}]") for i, item in enumerate(value)
    )


def load_backends(path: str) -> List[Backend]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    _require(isinstance(raw, dict) and isinstance(raw.get("backends"), list),
             "backends", "expected object with a 'backends' array")
    backends: List[Backend] = []
    seen_ids = set()
    seen_ports = set()
    for i, item in enumerate(raw["backends"]):
        base = f"backends[{i}]"
        _require(isinstance(item, dict), base, "expected object")
        model_id = _require_str(item.get("model_id"), f"{base}.model_id")
        _require(model_id not in seen_ids, f"{base}.model_id",
                 f"duplicate model_id {model_id!r}")
        seen_ids.add(model_id)
        host = _require_str(item.get("host"), f"{base}.host")
        _require(host == LOOPBACK_HOST, f"{base}.host",
                 f"expected loopback host {LOOPBACK_HOST!r}")
        port = _require_int(item.get("port"), f"{base}.port", MIN_PORT, MAX_PORT)
        _require(port not in seen_ports, f"{base}.port",
                 f"duplicate port {port}")
        seen_ports.add(port)
        params = _parse_vector(item.get("params"), f"{base}.params")
        max_conc = _require_int(item.get("max_concurrency"),
                                f"{base}.max_concurrency",
                                MIN_CONCURRENCY, MAX_CONCURRENCY)
        batch_size = _require_int(item.get("batch_size"), f"{base}.batch_size",
                                  MIN_BATCH_SIZE, MAX_BATCH_SIZE)
        backends.append(Backend(model_id, host, port, params, max_conc, batch_size))
    _require(len(backends) > 0, "backends", "expected at least one backend")
    return backends


def load_rules(path: str, backend_ids: set) -> List[RouteRule]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    _require(isinstance(raw, dict) and isinstance(raw.get("rules"), list),
             "rules", "expected object with a 'rules' array")
    rules: List[RouteRule] = []
    seen_ids = set()
    for i, item in enumerate(raw["rules"]):
        base = f"rules[{i}]"
        _require(isinstance(item, dict), base, "expected object")
        rule_id = _require_str(item.get("rule_id"), f"{base}.rule_id")
        _require(rule_id not in seen_ids, f"{base}.rule_id",
                 f"duplicate rule_id {rule_id!r}")
        seen_ids.add(rule_id)
        prefix = _require_str(item.get("prefix"), f"{base}.prefix")
        stable = _require_str(item.get("stable_backend"), f"{base}.stable_backend")
        _require(stable in backend_ids, f"{base}.stable_backend",
                 f"unknown backend {stable!r}")
        canary = _require_str(item.get("canary_backend"), f"{base}.canary_backend")
        _require(canary in backend_ids, f"{base}.canary_backend",
                 f"unknown backend {canary!r}")
        _require(canary != stable, f"{base}.canary_backend",
                 "canary backend must differ from stable backend")
        weight = _require_int(item.get("canary_weight_bp"),
                              f"{base}.canary_weight_bp",
                              MIN_WEIGHT_BP, MAX_WEIGHT_BP)
        batch_size = _require_int(item.get("batch_size"), f"{base}.batch_size",
                                  MIN_BATCH_SIZE, MAX_BATCH_SIZE)
        rules.append(RouteRule(rule_id, prefix, stable, canary, weight, batch_size))
    _require(len(rules) > 0, "rules", "expected at least one rule")
    return rules


def load_requests(path: str, param_dim: int) -> List[Request]:
    requests: List[Request] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            base = f"requests[{line_no}]"
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(base, f"invalid JSON: {exc}") from exc
            _require(isinstance(item, dict), base, "expected object")
            seq = _require_int(item.get("seq"), f"{base}.seq", 0, 2 ** 31 - 1)
            client_key = _require_str(item.get("client_key"), f"{base}.client_key")
            inputs = _parse_vector(item.get("inputs"), f"{base}.inputs")
            _require(len(inputs) == param_dim, f"{base}.inputs",
                     f"expected dimension {param_dim}, got {len(inputs)}")
            stage = _require_int(item.get("expected_stage"),
                                 f"{base}.expected_stage", 0, 2 ** 31 - 1)
            requests.append(Request(seq, client_key, inputs, stage))
    _require(len(requests) > 0, "requests", "expected at least one request")
    seqs = sorted(r.seq for r in requests)
    for i, seq in enumerate(seqs):
        _require(seq == i, "requests",
                 f"arrival sequence must be consecutive and unique from 0; "
                 f"missing seq {i}")
    return requests


def load_declaration(backends_path: str, rules_path: str,
                     requests_path: str) -> Declaration:
    """Load and validate all inputs; canonical ordering is by sorted id/seq so
    the result never depends on file ordering of declarations."""
    backends = load_backends(backends_path)
    dims = {len(b.params) for b in backends}
    _require(len(dims) == 1, "backends",
             f"all param vectors must share one fixed length, got {sorted(dims)}")
    param_dim = dims.pop()
    backend_ids = {b.model_id for b in backends}
    rules = load_rules(rules_path, backend_ids)
    requests = load_requests(requests_path, param_dim)
    return Declaration(
        backends=tuple(sorted(backends, key=lambda b: b.model_id)),
        rules=tuple(sorted(rules, key=lambda r: r.rule_id)),
        requests=tuple(sorted(requests, key=lambda r: r.seq)),
        param_dim=param_dim,
    )

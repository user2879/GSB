"""Gateway declaration loading with strict whole-document validation.

Any violation rejects the whole declaration with ``E_SPEC`` and reports the
first offending JSON-style field path (e.g. ``backends[2].port``).
"""

import json
import os

from .errors import SpecError

REQUIRED_TOP = ("backends", "rules", "request_stream")


def _fail(path, reason):
    raise SpecError(f"{reason} at {path}", field=path)


def _is_pos_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def load_declaration(path):
    if not os.path.isfile(path):
        raise SpecError(f"declaration file not found: {path}", field=path)
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SpecError(f"invalid JSON: {exc}", field=f"line {exc.lineno}")
    validate_declaration(data)
    return data


def validate_declaration(data):
    if not isinstance(data, dict):
        _fail("$", "declaration root must be an object")
    for key in REQUIRED_TOP:
        if key not in data:
            _fail(key, "missing required section")

    backends = data["backends"]
    if not isinstance(backends, list) or not backends:
        _fail("backends", "must be a non-empty array")

    seen_ids, seen_ports = set(), set()
    for idx, backend in enumerate(backends):
        base = f"backends[{idx}]"
        if not isinstance(backend, dict):
            _fail(base, "backend entry must be an object")
        for key in ("backend_id", "model_version", "host", "port", "model_dim",
                    "hidden_dim", "seed", "quant_version"):
            if key not in backend:
                _fail(f"{base}.{key}", "missing required field")
        bid = backend["backend_id"]
        if not isinstance(bid, str) or not bid:
            _fail(f"{base}.backend_id", "must be a non-empty string")
        if bid in seen_ids:
            _fail(f"{base}.backend_id", "duplicate backend identifier")
        seen_ids.add(bid)
        if not isinstance(backend["model_version"], str) or not backend["model_version"]:
            _fail(f"{base}.model_version", "must be a non-empty string")
        if backend["host"] != "127.0.0.1":
            _fail(f"{base}.host", "only literal loopback host 127.0.0.1 is allowed")
        port = backend["port"]
        if not _is_pos_int(port) or not 1024 <= port <= 65535:
            _fail(f"{base}.port", "must be an integer port in [1024, 65535]")
        if port in seen_ports:
            _fail(f"{base}.port", "duplicate loopback port across backends")
        seen_ports.add(port)
        for dim_key in ("model_dim", "hidden_dim"):
            if not _is_pos_int(backend[dim_key]):
                _fail(f"{base}.{dim_key}", "must be a positive integer")
        if not _is_int(backend["seed"]) or backend["seed"] < 0:
            _fail(f"{base}.seed", "must be a non-negative integer")
        if not _is_pos_int(backend["quant_version"]):
            _fail(f"{base}.quant_version", "must be a positive integer")

    rules = data["rules"]
    if not isinstance(rules, list) or not rules:
        _fail("rules", "must be a non-empty array")
    seen_rule_ids = set()
    for idx, rule in enumerate(rules):
        base = f"rules[{idx}]"
        if not isinstance(rule, dict):
            _fail(base, "rule entry must be an object")
        for key in ("rule_id", "prefix", "stable_backend", "gray_backend",
                    "gray_weight_bp", "batch_size"):
            if key not in rule:
                _fail(f"{base}.{key}", "missing required field")
        rid = rule["rule_id"]
        if not isinstance(rid, str) or not rid:
            _fail(f"{base}.rule_id", "must be a non-empty string")
        if rid in seen_rule_ids:
            _fail(f"{base}.rule_id", "duplicate rule identifier")
        seen_rule_ids.add(rid)
        if not isinstance(rule["prefix"], str) or not rule["prefix"]:
            _fail(f"{base}.prefix", "must be a non-empty string")
        for ref_key in ("stable_backend", "gray_backend"):
            ref = rule[ref_key]
            if ref not in seen_ids:
                _fail(f"{base}.{ref_key}", "references an undeclared backend")
        if rule["stable_backend"] == rule["gray_backend"]:
            _fail(f"{base}.gray_backend", "stable and gray backends must differ")
        weight = rule["gray_weight_bp"]
        if not _is_int(weight) or not 0 <= weight <= 10000:
            _fail(f"{base}.gray_weight_bp", "must be an integer in [0, 10000]")
        if not _is_pos_int(rule["batch_size"]):
            _fail(f"{base}.batch_size", "must be a positive integer")

    stream = data["request_stream"]
    if not isinstance(stream, list) or not stream:
        _fail("request_stream", "must be a non-empty array")
    seen_samples = set()
    for idx, req in enumerate(stream):
        base = f"request_stream[{idx}]"
        if not isinstance(req, dict):
            _fail(base, "request entry must be an object")
        for key in ("sample_id", "rule_id", "client_key"):
            if key not in req:
                _fail(f"{base}.{key}", "missing required field")
        sid = req["sample_id"]
        if not isinstance(sid, str) or not sid:
            _fail(f"{base}.sample_id", "must be a non-empty string")
        if sid in seen_samples:
            _fail(f"{base}.sample_id", "duplicate sample identifier")
        seen_samples.add(sid)
        if req["rule_id"] not in seen_rule_ids:
            _fail(f"{base}.rule_id", "references an undeclared rule")
        if not isinstance(req["client_key"], str) or not req["client_key"]:
            _fail(f"{base}.client_key", "must be a non-empty string")
    return True

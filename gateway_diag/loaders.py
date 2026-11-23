from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .errors import E_REF, E_SPEC, GatewayError
from .graph import ensure_acyclic
from .models import (
    SUPPORTED_QUANTIZATION,
    TARGET_SELECTS,
    Backend,
    Request,
    Rule,
    Spec,
    TargetRef,
)
from .routing import validate_request_routes
from .util import is_int, is_positive_int


def _fail(message: str, *, field: str, entry: int | str | None = None, **details: Any) -> None:
    raise GatewayError(
        E_SPEC,
        message,
        field=field,
        entry=entry,
        details={key: value for key, value in details.items() if value is not None},
    )


def _read_json(path: str | Path, label: str) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise GatewayError(E_SPEC, f"{label} file does not exist", field=label, entry=str(path))
    except json.JSONDecodeError as exc:
        raise GatewayError(
            E_SPEC,
            f"{label} file is not valid JSON",
            field=label,
            entry=str(path),
            details={"line": exc.lineno, "column": exc.colno, "reason": exc.msg},
        )
    except OSError as exc:
        raise GatewayError(E_SPEC, f"cannot read {label} file", field=label, entry=str(path), details={"reason": str(exc)})


def _as_sequence(value: Any, field: str, label: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            item = value[key]
            if not isinstance(item, dict):
                _fail(f"{field} mapping value must be an object", field=field, entry=key)
            copy = dict(item)
            copy.setdefault("id", key)
            result.append(copy)
        return result
    _fail(f"{field} must be an array or object", field=field, entry=label)


def _extract_document(document: Any, label: str) -> dict[str, Any]:
    if isinstance(document, list):
        return {label: document}
    if not isinstance(document, dict):
        _fail(f"{label} document must be a JSON object", field=label, entry=label)
    return document


def _merge_batch_limit(existing: int | None, value: Any, entry: str) -> int:
    if value is None:
        if existing is None:
            return 0
        return existing
    if not is_positive_int(value):
        _fail("batch_limit must be a positive integer", field="batch_limit", entry=entry)
    if existing not in (None, 0) and existing != value:
        _fail("conflicting batch_limit declarations", field="batch_limit", entry=entry)
    return value


def _parse_backend(raw: Any, index: int) -> Backend:
    if not isinstance(raw, dict):
        _fail("backend entry must be an object", field="backends", entry=index)
    backend_id = raw.get("id", raw.get("backend_id"))
    if not isinstance(backend_id, str) or not backend_id.strip():
        _fail("backend id must be a non-empty string", field="id", entry=index)
    host = raw.get("host")
    if host != "127.0.0.1":
        _fail(
            "backend host must be the loopback literal 127.0.0.1",
            field="host",
            entry=index,
            backend_id=backend_id,
        )
    port = raw.get("port")
    if not is_positive_int(port) or port > 65535:
        _fail("backend port must be an integer in 1..65535", field="port", entry=index, backend_id=backend_id)
    model_dim = raw.get("model_dim")
    if not is_positive_int(model_dim):
        _fail("model_dim must be a positive integer", field="model_dim", entry=index, backend_id=backend_id)
    hidden_dim = raw.get("hidden_dim")
    if not is_positive_int(hidden_dim):
        _fail("hidden_dim must be a positive integer", field="hidden_dim", entry=index, backend_id=backend_id)
    seed = raw.get("seed")
    if not is_int(seed) or seed < 0 or seed > 2**63 - 1:
        _fail("seed must be a non-negative integer", field="seed", entry=index, backend_id=backend_id)
    quantization = raw.get("quantization")
    if not isinstance(quantization, str) or quantization.lower() not in SUPPORTED_QUANTIZATION:
        _fail(
            "quantization must be one of fp32, fp16, int8",
            field="quantization",
            entry=index,
            backend_id=backend_id,
        )
    return Backend(
        id=backend_id,
        host=host,
        port=port,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        seed=seed,
        quantization=quantization.lower(),
    )


def _parse_backend_ref(raw: Any, rule_id: str, field: str) -> str:
    if isinstance(raw, str) and raw:
        return raw
    if isinstance(raw, dict):
        value = raw.get("backend_id", raw.get("backend", raw.get("id")))
        if isinstance(value, str) and value:
            return value
    _fail("backend target must be a non-empty backend id", field=field, entry=rule_id, rule_id=rule_id)


def _parse_canary_target(raw: Any, rule_id: str) -> TargetRef:
    if isinstance(raw, str) and raw:
        return TargetRef(kind="backend", backend_id=raw)
    if not isinstance(raw, dict):
        _fail(
            "canary_backend must be a backend id or a derivation object",
            field="canary_backend",
            entry=rule_id,
            rule_id=rule_id,
        )
    reference = raw.get("reference")
    reference = reference if isinstance(reference, dict) else {}
    from_rule = raw.get("from_rule", raw.get("rule_id", reference.get("from_rule", reference.get("rule_id"))))
    backend_id = raw.get("backend_id", raw.get("backend"))
    if from_rule is not None:
        if not isinstance(from_rule, str) or not from_rule:
            _fail(
                "canary derivation from_rule must be a non-empty rule id",
                field="canary_backend.from_rule",
                entry=rule_id,
                rule_id=rule_id,
            )
        select = raw.get("select", "conclusion")
        aliases = {
            "conclusion": "conclusion",
            "selected": "conclusion",
            "stable": "stable_target",
            "stable_target": "stable_target",
            "canary": "canary_target",
            "canary_target": "canary_target",
        }
        normalized = aliases.get(select) if isinstance(select, str) else None
        if normalized is None or normalized not in TARGET_SELECTS:
            _fail(
                "canary derivation select must be conclusion, stable_target or canary_target",
                field="canary_backend.select",
                entry=rule_id,
                rule_id=rule_id,
            )
        return TargetRef(kind="derived", from_rule=from_rule, select=normalized)
    if backend_id is not None:
        if not isinstance(backend_id, str) or not backend_id:
            _fail(
                "canary backend id must be a non-empty string",
                field="canary_backend",
                entry=rule_id,
                rule_id=rule_id,
            )
        return TargetRef(kind="backend", backend_id=backend_id)
    _fail(
        "canary_backend object requires from_rule or backend_id",
        field="canary_backend",
        entry=rule_id,
        rule_id=rule_id,
    )


def _parse_depends_on(raw: Any, rule_id: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        _fail("depends_on must be an array of rule ids", field="depends_on", entry=rule_id, rule_id=rule_id)
    result: set[str] = set()
    for index, item in enumerate(raw):
        if isinstance(item, str) and item:
            result.add(item)
        elif isinstance(item, dict):
            target = item.get("rule_id", item.get("from_rule", item.get("rule")))
            if not isinstance(target, str) or not target:
                _fail(
                    "dependency reference must contain a rule id",
                    field=f"depends_on[{index}]",
                    entry=rule_id,
                    rule_id=rule_id,
                )
            result.add(target)
        else:
            _fail(
                "dependency reference must be a rule id or object",
                field=f"depends_on[{index}]",
                entry=rule_id,
                rule_id=rule_id,
            )
    return tuple(sorted(result))


def _parse_rule(raw: Any, index: int, batch_limit: int) -> Rule:
    if not isinstance(raw, dict):
        _fail("rule entry must be an object", field="rules", entry=index)
    rule_id = raw.get("id", raw.get("rule_id"))
    if not isinstance(rule_id, str) or not rule_id.strip():
        _fail("rule id must be a non-empty string", field="id", entry=index)
    prefix = raw.get("request_prefix", raw.get("prefix"))
    if not isinstance(prefix, str) or not prefix:
        _fail("request_prefix must be a non-empty string", field="request_prefix", entry=rule_id, rule_id=rule_id)
    stable = _parse_backend_ref(
        raw.get("stable_backend", raw.get("stable_target")),
        rule_id,
        "stable_backend",
    )
    canary_raw = raw.get(
        "canary_backend",
        raw.get("canary_target", raw.get("gray_backend", raw.get("gray_target"))),
    )
    canary = _parse_canary_target(canary_raw, rule_id)
    weight = raw.get("canary_weight_bp", raw.get("gray_weight_bp", raw.get("weight_bp")))
    if not is_int(weight) or not 0 <= weight <= 10000:
        _fail(
            "canary_weight_bp must be an integer in 0..10000",
            field="canary_weight_bp",
            entry=rule_id,
            rule_id=rule_id,
        )
    batch_size = raw.get("batch_size")
    if not is_positive_int(batch_size) or batch_size > batch_limit:
        _fail(
            "batch_size must be a positive integer not greater than batch_limit",
            field="batch_size",
            entry=rule_id,
            rule_id=rule_id,
            details={"batch_limit": batch_limit, "actual": batch_size},
        )
    version = raw.get("version", raw.get("rule_version"))
    if not is_int(version) or version < 0:
        _fail(
            "version must be a non-negative integer",
            field="version",
            entry=rule_id,
            rule_id=rule_id,
        )
    depends_on = _parse_depends_on(raw.get("depends_on", raw.get("dependencies")), rule_id)
    return Rule(
        id=rule_id,
        request_prefix=prefix,
        stable_backend=stable,
        canary_target=canary,
        canary_weight_bp=weight,
        batch_size=batch_size,
        version=version,
        depends_on=depends_on,
    )


def _validate_references(backends: dict[str, Backend], rules: dict[str, Rule]) -> None:
    missing: dict[str, set[str]] = {}
    for rule_id in sorted(rules):
        rule = rules[rule_id]
        direct_refs = {rule.stable_backend}
        if rule.canary_target.direct_backend_id:
            direct_refs.add(rule.canary_target.direct_backend_id)
        for backend_id in direct_refs:
            if backend_id not in backends:
                missing.setdefault(f"backend:{backend_id}", set()).add(rule_id)
        for dep in rule.dependency_ids:
            if dep not in rules:
                missing.setdefault(f"rule:{dep}", set()).add(rule_id)
    if missing:
        referrers = sorted({referrer for values in missing.values() for referrer in values})
        raise GatewayError(
            E_REF,
            "rule references a missing target",
            rule_id=referrers[0] if referrers else None,
            field="reference",
            details={
                "missing_targets": sorted(missing),
                "referrers": referrers,
                "references": {key: sorted(values) for key, values in sorted(missing.items())},
            },
        )


def _parse_requests(path: str | Path, spec: Spec) -> tuple[Request, ...]:
    file_path = Path(path)
    requests: list[Request] = []
    seen_ids: set[str] = set()
    seen_seq: set[int] = set()
    try:
        handle = file_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise GatewayError(E_SPEC, "cannot read requests file", field="requests", entry=str(path), details={"reason": str(exc)})
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GatewayError(
                    E_SPEC,
                    "request line is not valid JSON",
                    field="request",
                    entry=line_number,
                    details={"line": line_number, "column": exc.colno, "reason": exc.msg},
                )
            if not isinstance(raw, dict):
                _fail("request entry must be an object", field="request", entry=line_number)
            request_id = raw.get("request_id", raw.get("req_id", raw.get("id")))
            if not isinstance(request_id, str) or not request_id:
                _fail("request_id must be a non-empty string", field="request_id", entry=line_number)
            if request_id in seen_ids:
                _fail(
                    "request_id must be unique",
                    field="request_id",
                    entry=line_number,
                    rule_id=raw.get("rule_id"),
                    request_id=request_id,
                )
            arrival_seq = raw.get("arrival_seq", raw.get("seq"))
            if not is_int(arrival_seq) or arrival_seq < 0:
                _fail(
                    "arrival_seq must be a non-negative integer",
                    field="arrival_seq",
                    entry=line_number,
                    rule_id=raw.get("rule_id"),
                    request_id=request_id,
                )
            if arrival_seq in seen_seq:
                _fail(
                    "arrival_seq must be unique",
                    field="arrival_seq",
                    entry=line_number,
                    rule_id=raw.get("rule_id"),
                    request_id=request_id,
                )
            client_key = raw.get("client_key", raw.get("client"))
            if not isinstance(client_key, str) or not client_key:
                _fail(
                    "client_key must be a non-empty string",
                    field="client_key",
                    entry=line_number,
                    rule_id=raw.get("rule_id"),
                    request_id=request_id,
                )
            prefix = raw.get("prefix", raw.get("path"))
            if not isinstance(prefix, str) or not prefix:
                _fail(
                    "prefix must be a non-empty string",
                    field="prefix",
                    entry=line_number,
                    rule_id=raw.get("rule_id"),
                    request_id=request_id,
                )
            vector = raw.get("vector", raw.get("features"))
            if not isinstance(vector, list):
                _fail(
                    "vector must be an array of finite numbers",
                    field="vector",
                    entry=line_number,
                    rule_id=raw.get("rule_id"),
                    request_id=request_id,
                )
            converted: list[float] = []
            for vector_index, value in enumerate(vector):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    _fail(
                        "vector values must be finite numbers",
                        field=f"vector[{vector_index}]",
                        entry=line_number,
                        rule_id=raw.get("rule_id"),
                        request_id=request_id,
                    )
                converted.append(float(value))
            explicit_rule = raw.get("rule_id")
            if explicit_rule is not None and (not isinstance(explicit_rule, str) or not explicit_rule):
                _fail(
                    "rule_id must be a non-empty string when supplied",
                    field="rule_id",
                    entry=line_number,
                    request_id=request_id,
                )
            seen_ids.add(request_id)
            seen_seq.add(arrival_seq)
            requests.append(
                Request(
                    request_id=request_id,
                    arrival_seq=arrival_seq,
                    client_key=client_key,
                    prefix=prefix,
                    vector=tuple(converted),
                    rule_id=explicit_rule,
                )
            )
    return tuple(requests)


def load_spec(
    *,
    config_path: str | Path | None = None,
    backends_path: str | Path | None = None,
    rules_path: str | Path | None = None,
    requests_path: str | Path,
) -> Spec:
    if config_path is None and backends_path is None and rules_path is None:
        raise GatewayError(E_SPEC, "one of --config or --backends/--rules is required", field="config")

    backends_raw: Any = None
    rules_raw: Any = None
    batch_limit = 0
    documents: list[tuple[str, Any]] = []
    if config_path is not None:
        documents.append(("config", _read_json(config_path, "config")))
    if backends_path is not None:
        documents.append(("backends", _read_json(backends_path, "backends")))
    if rules_path is not None:
        documents.append(("rules", _read_json(rules_path, "rules")))

    for label, document in documents:
        doc = _extract_document(document, label)
        schema_version = doc.get("schema_version")
        if schema_version is not None and schema_version != 1:
            _fail("unsupported schema_version", field="schema_version", entry=label)
        settings = doc.get("settings") if isinstance(doc.get("settings"), dict) else {}
        raw_limit = doc.get(
            "batch_limit",
            doc.get("max_batch_size", settings.get("batch_limit", settings.get("max_batch_size"))),
        )
        batch_limit = _merge_batch_limit(batch_limit, raw_limit, label)
        document_backends = doc.get("backends")
        document_rules = doc.get("rules")
        if label == "backends" and isinstance(document, list):
            document_backends = document
        if label == "rules" and isinstance(document, list):
            document_rules = document
        if document_backends is not None:
            if backends_raw is not None:
                _fail("backends are declared in more than one input", field="backends", entry=label)
            backends_raw = document_backends
        if document_rules is not None:
            if rules_raw is not None:
                _fail("rules are declared in more than one input", field="rules", entry=label)
            rules_raw = document_rules

    if batch_limit <= 0:
        _fail("batch_limit must be declared as a positive integer", field="batch_limit")
    if backends_raw is None:
        _fail("backends declaration is missing", field="backends")
    if rules_raw is None:
        _fail("rules declaration is missing", field="rules")

    backend_entries = _as_sequence(backends_raw, "backends", "config")
    backends: dict[str, Backend] = {}
    ports: dict[int, str] = {}
    for index, raw in enumerate(backend_entries):
        backend = _parse_backend(raw, index)
        if backend.id in backends:
            _fail(
                "duplicate backend id",
                field="id",
                entry=index,
                backend_id=backend.id,
                first_entry=next(i for i, item in enumerate(backend_entries) if (item.get("id", item.get("backend_id")) if isinstance(item, dict) else None) == backend.id),
            )
        if backend.port in ports:
            _fail(
                "backend port must be globally unique",
                field="port",
                entry=index,
                backend_id=backend.id,
                details={"port": backend.port, "first_backend": ports[backend.port]},
            )
        backends[backend.id] = backend
        ports[backend.port] = backend.id

    rule_entries = _as_sequence(rules_raw, "rules", "config")
    rules: dict[str, Rule] = {}
    for index, raw in enumerate(rule_entries):
        rule = _parse_rule(raw, index, batch_limit)
        if rule.id in rules:
            _fail("duplicate rule id", field="id", entry=rule.id, rule_id=rule.id)
        rules[rule.id] = rule

    _validate_references(backends, rules)
    provisional = Spec(batch_limit=batch_limit, backends=backends, rules=rules, requests=())
    ensure_acyclic(provisional)
    requests = _parse_requests(requests_path, provisional)
    spec = Spec(batch_limit=batch_limit, backends=backends, rules=rules, requests=requests)
    validate_request_routes(spec)
    return spec


def load_backends_file(path: str | Path) -> list[Backend]:
    document = _read_json(path, "backends")
    doc = _extract_document(document, "backends")
    raw = doc.get("backends", document if isinstance(document, list) else None)
    if raw is None:
        raise GatewayError(E_SPEC, "backends declaration is missing", field="backends", entry=str(path))
    result: list[Backend] = []
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, entry in enumerate(_as_sequence(raw, "backends", "backends")):
        backend = _parse_backend(entry, index)
        if backend.id in seen_ids:
            _fail("duplicate backend id", field="id", entry=index, backend_id=backend.id)
        if backend.port in seen_ports:
            _fail("backend port must be globally unique", field="port", entry=index, backend_id=backend.id)
        seen_ids.add(backend.id)
        seen_ports.add(backend.port)
        result.append(backend)
    return result


def spec_declarations_from_payload(payload: Any, requests: tuple[Request, ...] = ()) -> Spec:
    """Rebuild a validated declaration-only Spec from a canonical snapshot."""
    if not isinstance(payload, dict):
        _fail("declaration payload must be an object", field="declaration")
    batch_limit = payload.get("batch_limit")
    if not is_positive_int(batch_limit):
        _fail("batch_limit must be a positive integer", field="batch_limit")
    backends_raw = payload.get("backends")
    rules_raw = payload.get("rules")
    if not isinstance(backends_raw, list):
        _fail("backends must be an array", field="backends")
    if not isinstance(rules_raw, list):
        _fail("rules must be an array", field="rules")
    backends: dict[str, Backend] = {}
    ports: set[int] = set()
    for index, raw in enumerate(backends_raw):
        backend = _parse_backend(raw, index)
        if backend.id in backends:
            _fail("duplicate backend id", field="id", entry=index, backend_id=backend.id)
        if backend.port in ports:
            _fail("backend port must be globally unique", field="port", entry=index, backend_id=backend.id)
        backends[backend.id] = backend
        ports.add(backend.port)
    rules: dict[str, Rule] = {}
    for index, raw in enumerate(rules_raw):
        rule = _parse_rule(raw, index, batch_limit)
        if rule.id in rules:
            _fail("duplicate rule id", field="id", entry=rule.id, rule_id=rule.id)
        rules[rule.id] = rule
    _validate_references(backends, rules)
    spec = Spec(batch_limit=batch_limit, backends=backends, rules=rules, requests=requests)
    ensure_acyclic(spec)
    if requests:
        validate_request_routes(spec)
    return spec

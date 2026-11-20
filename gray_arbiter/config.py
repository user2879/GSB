"""Configuration and NDJSON loading with whole-file validation.

All malformed input is rejected before any request is forwarded.  The error
code is ``E_POOL`` and the ``field`` member points at the first invalid field
in deterministic declaration order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import BusinessError
from .model import Config, PoolSpec, RequestSpec, RuleSpec


POOL_ALIASES = {
    "pool_id": ("pool_id", "id"),
    "max_connections": (
        "max_connections",
        "capacity",
        "max_concurrent_connections",
        "max_concurrent",
    ),
    "max_requests_per_connection": (
        "max_requests_per_connection",
        "reuse_limit",
        "max_reuse_requests",
        "max_requests",
        "max_reuse_per_connection",
        "reuse_request_limit",
    ),
    "idle_timeout_ms": (
        "idle_timeout_ms",
        "idle_reclaim_ms",
        "idle_recycle_ms",
        "idle_ms",
    ),
    "lendable_to": ("lendable_to", "can_lend_to", "lendable_pool_ids"),
}

RULE_ALIASES = {
    "rule_id": ("rule_id", "id"),
    "stable_pool": ("stable_pool", "stable_pool_id"),
    "gray_pool": ("gray_pool", "gray_pool_id", "canary_pool"),
    "mirror_pool": ("mirror_pool", "mirror_pool_id"),
    "gray_weight_bp": (
        "gray_weight_bp",
        "gray_weight",
        "gray_weight_basis_points",
        "gray_ratio_bp",
        "gray_permyriad",
    ),
    "mirror_sample_bp": (
        "mirror_sample_bp",
        "mirror_sample_rate_bp",
        "sample_rate_bp",
        "mirror_sampling_rate_bp",
        "mirror_sampling_bp",
    ),
}


def _fail(field: str, message: str, *, code: str = "E_POOL", **details: Any) -> None:
    raise BusinessError(
        code=code,
        message=message,
        field=field,
        phase="configuration",
        details=details,
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _pick(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _require_object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(field, "must be a JSON object")
    return value


def _required_string(mapping: Mapping[str, Any], key: str, field: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        _fail(field, "must be a non-empty string")
    return value.strip()


def _required_int(
    mapping: Mapping[str, Any],
    key: str,
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = mapping.get(key)
    if not _is_int(value):
        _fail(field, "must be an integer")
    if value < minimum or value > maximum:
        _fail(field, f"must be in [{minimum}, {maximum}]", actual=value)
    return value


def _parse_endpoint(raw: Mapping[str, Any], field_prefix: str) -> tuple[str, int]:
    endpoint = raw.get("endpoint")
    if endpoint is not None:
        if isinstance(endpoint, str):
            endpoint = {"target": endpoint}
        else:
            endpoint = _require_object(endpoint, f"{field_prefix}.endpoint")
        raw = endpoint
        field_prefix = f"{field_prefix}.endpoint"
    if raw.get("target") is not None or raw.get("url") is not None:
        target = raw.get("target", raw.get("url"))
        if not isinstance(target, str):
            _fail(f"{field_prefix}.target", "must be a loopback HTTP URL string")
        prefix = "http://"
        if not target.lower().startswith(prefix):
            _fail(f"{field_prefix}.target", "must start with http://")
        authority = target[len(prefix) :].split("/", 1)[0]
        if ":" not in authority:
            _fail(f"{field_prefix}.target", "must include an explicit port")
        host, port_text = authority.rsplit(":", 1)
        try:
            port = int(port_text)
        except ValueError:
            _fail(f"{field_prefix}.target", "contains a non-integer port")
        port_field = f"{field_prefix}.target"
        host_field = f"{field_prefix}.target"
    else:
        host = raw.get("host", "127.0.0.1")
        port = raw.get("port")
        port_field = f"{field_prefix}.port"
        host_field = f"{field_prefix}.host"
    if host != "127.0.0.1":
        _fail(host_field, "host must be 127.0.0.1", actual=host)
    if not _is_int(port):
        _fail(port_field, "must be an integer")
    if port < 1024 or port > 65535:
        _fail(port_field, "must be a high loopback port in [1024, 65535]", actual=port)
    return host, port


def _parse_pools(items: Any) -> dict[str, PoolSpec]:
    if isinstance(items, dict):
        normalized = []
        for key in sorted(items):
            value = items[key]
            if not isinstance(value, dict):
                _fail(f"pools.{key}", "must be a JSON object")
            declared = _pick(value, POOL_ALIASES["pool_id"])
            if declared is not None and declared != key:
                _fail(
                    f"pools.{key}.pool_id",
                    "mapping key and pool_id must agree",
                    pool_id=key,
                )
            normalized.append({**value, "pool_id": key})
        items = normalized
    if not isinstance(items, list) or not items:
        _fail("pools", "must be a non-empty array")
    pools: dict[str, PoolSpec] = {}
    for index, raw_item in enumerate(items):
        raw = _require_object(raw_item, f"pools[{index}]")
        pool_id_value = _pick(raw, POOL_ALIASES["pool_id"])
        if not isinstance(pool_id_value, str) or not pool_id_value.strip():
            _fail(f"pools[{index}].pool_id", "must be a non-empty string")
        pool_id = pool_id_value.strip()
        if pool_id in pools:
            _fail(f"pools[{index}].pool_id", "duplicate pool_id", pool_id=pool_id)
        max_connections = _pick(raw, POOL_ALIASES["max_connections"])
        if not _is_int(max_connections) or max_connections <= 0:
            _fail(
                f"pools[{index}].max_connections",
                "must be a positive integer",
                pool_id=pool_id,
            )
        max_requests = _pick(raw, POOL_ALIASES["max_requests_per_connection"])
        if not _is_int(max_requests) or max_requests <= 0:
            _fail(
                f"pools[{index}].max_requests_per_connection",
                "must be a positive integer",
                pool_id=pool_id,
            )
        idle_timeout = _pick(raw, POOL_ALIASES["idle_timeout_ms"])
        if not _is_int(idle_timeout) or idle_timeout < 0:
            _fail(
                f"pools[{index}].idle_timeout_ms",
                "must be a non-negative integer",
                pool_id=pool_id,
            )
        lendable_raw = _pick(raw, POOL_ALIASES["lendable_to"])
        if lendable_raw is None:
            lendable_raw = []
        if not isinstance(lendable_raw, list) or any(
            not isinstance(item, str) or not item.strip() for item in lendable_raw
        ):
            _fail(
                f"pools[{index}].lendable_to",
                "must be an array of non-empty pool identifiers",
                pool_id=pool_id,
            )
        if len(set(lendable_raw)) != len(lendable_raw):
            _fail(
                f"pools[{index}].lendable_to",
                "must not contain duplicate pool identifiers",
                pool_id=pool_id,
            )
        if pool_id in lendable_raw:
            _fail(
                f"pools[{index}].lendable_to",
                "a pool must not lend to itself (self-loop)",
                pool_id=pool_id,
            )
        host, port = _parse_endpoint(raw, f"pools[{index}]")
        pools[pool_id] = PoolSpec(
            pool_id=pool_id,
            host=host,
            port=port,
            max_connections=max_connections,
            max_requests_per_connection=max_requests,
            idle_timeout_ms=idle_timeout,
            lendable_to=tuple(lendable_raw),
        )
    for index, pool in enumerate(pools.values()):
        for lendee in pool.lendable_to:
            if lendee not in pools:
                _fail(
                    f"pools[{index}].lendable_to",
                    "references an unknown pool identifier",
                    pool_id=pool.pool_id,
                    referenced_pool=lendee,
                )
    return pools


def _parse_rules(items: Any, pools: Mapping[str, PoolSpec]) -> dict[str, RuleSpec]:
    if isinstance(items, dict):
        normalized = []
        for key in sorted(items):
            value = items[key]
            if not isinstance(value, dict):
                _fail(f"rules.{key}", "must be a JSON object")
            declared = _pick(value, RULE_ALIASES["rule_id"])
            if declared is not None and declared != key:
                _fail(
                    f"rules.{key}.rule_id",
                    "mapping key and rule_id must agree",
                    rule_id=key,
                )
            normalized.append({**value, "rule_id": key})
        items = normalized
    if not isinstance(items, list) or not items:
        _fail("rules", "must be a non-empty array")
    rules: dict[str, RuleSpec] = {}
    for index, raw_item in enumerate(items):
        raw = _require_object(raw_item, f"rules[{index}]")
        rule_id_value = _pick(raw, RULE_ALIASES["rule_id"])
        if not isinstance(rule_id_value, str) or not rule_id_value.strip():
            _fail(f"rules[{index}].rule_id", "must be a non-empty string")
        rule_id = rule_id_value.strip()
        if rule_id in rules:
            _fail(f"rules[{index}].rule_id", "duplicate rule_id", rule_id=rule_id)
        prefix = raw.get("prefix")
        if (
            not isinstance(prefix, str)
            or not prefix.startswith("/")
            or prefix.startswith("//")
            or "\r" in prefix
            or "\n" in prefix
        ):
            _fail(f"rules[{index}].prefix", "must be a safe absolute path prefix")
        stable_pool = _pick(raw, RULE_ALIASES["stable_pool"])
        gray_pool = _pick(raw, RULE_ALIASES["gray_pool"])
        mirror_pool = _pick(raw, RULE_ALIASES["mirror_pool"])
        for field, value in (
            ("stable_pool", stable_pool),
            ("gray_pool", gray_pool),
            ("mirror_pool", mirror_pool),
        ):
            if not isinstance(value, str) or not value.strip():
                _fail(f"rules[{index}].{field}", "must be a non-empty pool identifier")
            if value not in pools:
                _fail(
                    f"rules[{index}].{field}",
                    "references an unknown pool identifier",
                    rule_id=rule_id,
                    referenced_pool=value,
                )
        stable_pool = stable_pool.strip()
        gray_pool = gray_pool.strip()
        mirror_pool = mirror_pool.strip()
        if mirror_pool in (stable_pool, gray_pool):
            _fail(
                f"rules[{index}].mirror_pool",
                "mirror_pool must be independent from stable_pool and gray_pool",
                rule_id=rule_id,
            )
        gray_weight = _pick(raw, RULE_ALIASES["gray_weight_bp"])
        if not _is_int(gray_weight) or not 0 <= gray_weight <= 10000:
            _fail(
                f"rules[{index}].gray_weight_bp",
                "must be an integer in [0, 10000]",
                rule_id=rule_id,
            )
        mirror_sample = _pick(raw, RULE_ALIASES["mirror_sample_bp"])
        if not _is_int(mirror_sample) or not 0 <= mirror_sample <= 10000:
            _fail(
                f"rules[{index}].mirror_sample_bp",
                "must be an integer in [0, 10000]",
                rule_id=rule_id,
            )
        rules[rule_id] = RuleSpec(
            rule_id=rule_id,
            prefix=prefix,
            stable_pool=stable_pool,
            gray_pool=gray_pool,
            mirror_pool=mirror_pool,
            gray_weight_bp=gray_weight,
            mirror_sample_bp=mirror_sample,
        )
    return rules


def _find_cycles(pools: Mapping[str, PoolSpec]) -> list[list[str]]:
    adjacency = {pool_id: list(pool.lendable_to) for pool_id, pool in pools.items()}
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []
    cycles: list[list[str]] = []

    def visit(node: str) -> None:
        if node in visiting:
            start = stack.index(node)
            cycles.append(stack[start:] + [node])
            return
        if node in visited:
            return
        visiting.add(node)
        stack.append(node)
        for neighbour in sorted(adjacency.get(node, [])):
            visit(neighbour)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for node in sorted(pools):
        visit(node)
    return cycles


def validate_config(
    config: Config | Mapping[str, Any],
    *,
    detect_borrow_cycles: bool = True,
    cycle_code: str = "E_POOL",
) -> Config:
    """Validate a programmatically constructed Config as well as parsed JSON."""

    if isinstance(config, Mapping):
        return parse_config(
            dict(config),
            detect_borrow_cycles=detect_borrow_cycles,
            cycle_code=cycle_code,
        )

    # Reusing the JSON parser keeps validation rules in one place while also
    # ensuring aliases cannot hide malformed values.
    raw = {
        "version": config.version,
        "pools": [
            {
                "pool_id": pool.pool_id,
                "host": pool.host,
                "port": pool.port,
                "max_connections": pool.max_connections,
                "max_requests_per_connection": pool.max_requests_per_connection,
                "idle_timeout_ms": pool.idle_timeout_ms,
                "lendable_to": list(pool.lendable_to),
            }
            for pool in config.pools.values()
        ],
        "rules": [
            {
                "rule_id": rule.rule_id,
                "prefix": rule.prefix,
                "stable_pool": rule.stable_pool,
                "gray_pool": rule.gray_pool,
                "mirror_pool": rule.mirror_pool,
                "gray_weight_bp": rule.gray_weight_bp,
                "mirror_sample_bp": rule.mirror_sample_bp,
            }
            for rule in config.rules.values()
        ],
        "borrow_ttl_ms": config.borrow_ttl_ms,
        "queue_timeout_ms": config.queue_timeout_ms,
        "main_timeout_ms": config.main_timeout_ms,
        "mirror_timeout_ms": config.mirror_timeout_ms,
    }
    rebuilt = parse_config(raw, detect_borrow_cycles=detect_borrow_cycles, cycle_code=cycle_code)
    return rebuilt


def parse_config(
    raw: Any,
    *,
    detect_borrow_cycles: bool = True,
    cycle_code: str = "E_POOL",
    source: dict[str, Any] | None = None,
) -> Config:
    data = _require_object(raw, "$")
    version = data.get("version", 1)
    if not _is_int(version) or version < 1:
        _fail("version", "must be a positive integer")
    pools = _parse_pools(data.get("pools"))
    rules = _parse_rules(data.get("rules"), pools)
    mirror_pools = {rule.mirror_pool for rule in rules.values()}
    for index, pool in enumerate(pools.values()):
        if pool.pool_id in mirror_pools and pool.lendable_to:
            _fail(
                f"pools[{index}].lendable_to",
                "borrow relationships must not cross a mirror pool",
                pool_id=pool.pool_id,
            )
        for lendee in pool.lendable_to:
            if lendee in mirror_pools:
                _fail(
                    f"pools[{index}].lendable_to",
                    "borrow relationships must not cross a mirror pool",
                    pool_id=pool.pool_id,
                    referenced_pool=lendee,
                )
    cycles = _find_cycles(pools)
    if cycles:
        if detect_borrow_cycles:
            cycle = cycles[0]
            _fail(
                "pools.lendable_to",
                "borrow relationship cycle detected",
                code=cycle_code,
                cycle=cycle,
            )
    borrow_ttl = data.get("borrow_ttl_ms", 5000)
    queue_timeout = data.get("queue_timeout_ms", 30000)
    main_timeout = data.get("main_timeout_ms", 2000)
    mirror_timeout = data.get("mirror_timeout_ms", 500)
    for field, value, minimum in (
        ("borrow_ttl_ms", borrow_ttl, 1),
        ("queue_timeout_ms", queue_timeout, 1),
        ("main_timeout_ms", main_timeout, 1),
        ("mirror_timeout_ms", mirror_timeout, 1),
    ):
        if not _is_int(value) or value < minimum:
            _fail(field, f"must be an integer >= {minimum}", actual=value)
    return Config(
        version=version,
        pools=pools,
        rules=rules,
        borrow_ttl_ms=borrow_ttl,
        queue_timeout_ms=queue_timeout,
        main_timeout_ms=main_timeout,
        mirror_timeout_ms=mirror_timeout,
        source=dict(source or {}),
    )


def load_config(
    path: str | Path,
    *,
    rules_path: str | Path | None = None,
    detect_borrow_cycles: bool = True,
    cycle_code: str = "E_POOL",
) -> Config:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8-sig") as fh:
            raw = json.load(fh)
    except OSError as exc:
        _fail("$", f"cannot read config: {exc}")
    except json.JSONDecodeError as exc:
        _fail("$", f"invalid JSON: {exc.msg}", line=exc.lineno, column=exc.colno)
    if rules_path is not None:
        raw = _require_object(raw, "$")
        try:
            with Path(rules_path).open("r", encoding="utf-8-sig") as fh:
                rules_raw = json.load(fh)
        except OSError as exc:
            _fail("$", f"cannot read rules config: {exc}")
        except json.JSONDecodeError as exc:
            _fail("$", f"invalid rules JSON: {exc.msg}", line=exc.lineno, column=exc.colno)
        if isinstance(rules_raw, list):
            raw = {**raw, "rules": rules_raw}
        else:
            rules_raw = _require_object(rules_raw, "$rules")
            raw = {**raw, **rules_raw}
    return parse_config(
        raw,
        detect_borrow_cycles=detect_borrow_cycles,
        cycle_code=cycle_code,
        source={"config_path": str(config_path), "rules_path": str(rules_path) if rules_path else None},
    )


def load_request_stream(path: str | Path) -> list[RequestSpec]:
    stream_path = Path(path)
    try:
        lines = stream_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        _fail("requests", f"cannot read request stream: {exc}")
    requests: list[RequestSpec] = []
    seen: set[str] = set()
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        field_prefix = f"requests[line={line_no}]"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(field_prefix, f"invalid NDJSON: {exc.msg}", line=line_no)
        if not isinstance(raw, dict):
            _fail(field_prefix, "must be a JSON object")
        request_id = raw.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            _fail(f"{field_prefix}.request_id", "must be a non-empty string")
        request_id = request_id.strip()
        if request_id in seen:
            _fail(f"{field_prefix}.request_id", "duplicate request_id", request_id=request_id)
        seen.add(request_id)
        path_value = raw.get("path")
        if (
            not isinstance(path_value, str)
            or not path_value.startswith("/")
            or path_value.startswith("//")
            or "\r" in path_value
            or "\n" in path_value
        ):
            _fail(f"{field_prefix}.path", "must be a safe absolute path beginning with /")
        client_key = raw.get("client_key")
        if not isinstance(client_key, str) or not client_key:
            _fail(f"{field_prefix}.client_key", "must be a non-empty string")
        timestamp_ms = raw.get("timestamp_ms")
        if not _is_int(timestamp_ms) or timestamp_ms < 0:
            _fail(f"{field_prefix}.timestamp_ms", "must be a non-negative integer")
        requests.append(
            RequestSpec(
                request_id=request_id,
                path=path_value,
                client_key=client_key,
                timestamp_ms=timestamp_ms,
                line_no=line_no,
            )
        )
    if not requests:
        _fail("requests", "request stream contains no requests")
    ordered = sorted(requests, key=lambda item: (item.timestamp_ms, item.line_no))
    return [
        RequestSpec(
            request_id=item.request_id,
            path=item.path,
            client_key=item.client_key,
            timestamp_ms=item.timestamp_ms,
            ordinal=index,
            line_no=item.line_no,
        )
        for index, item in enumerate(ordered, start=1)
    ]


def match_rule(path: str, rules: Mapping[str, RuleSpec]) -> RuleSpec | None:
    matches = [rule for rule in rules.values() if path.startswith(rule.prefix)]
    if not matches:
        return None
    return sorted(matches, key=lambda rule: (-len(rule.prefix), rule.rule_id))[0]


def config_digest_material(config: Config) -> dict[str, Any]:
    return config.digest_material()

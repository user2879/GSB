from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .errors import ErrorRecord, SpecValidationError
from .models import BorrowRule, ParameterSnapshot, PoolConfig, TraceEvent


CANONICAL_EVENT_TYPES = {
    "establish": "establish",
    "establishment": "establish",
    "建立": "establish",
    "request": "request",
    "请求": "request",
    "idle_release": "idle_release",
    "idle-release": "idle_release",
    "空闲释放": "idle_release",
    "close": "close",
    "关闭": "close",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _line_for_text(text: str, token: str | None) -> int:
    if not token:
        return 1
    quoted = json.dumps(token, ensure_ascii=False)
    for number, line in enumerate(text.splitlines(), start=1):
        if quoted in line:
            return number
    return 1


def _error(
    message: str,
    path: str,
    line: int | None,
    *,
    business_code: str | None = None,
    details: dict[str, Any] | None = None,
) -> ErrorRecord:
    return ErrorRecord(
        code="E_SPEC",
        message=message,
        path=path,
        line=line,
        business_code=business_code,
        details=details or {},
    )


def _read_json(path: str | Path, label: str) -> tuple[dict[str, Any] | None, str, list[ErrorRecord]]:
    source = Path(path)
    errors: list[ErrorRecord] = []
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(_error(f"cannot read {label}: {exc}", label, 1))
        return None, "", errors
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        errors.append(_error(f"invalid JSON: {exc.msg}", label, exc.lineno))
        return None, text, errors
    if not isinstance(data, dict):
        errors.append(_error(f"{label} root must be an object", label, 1))
        return None, text, errors
    return data, text, errors


def _positive_int(
    value: Any,
    *,
    path: str,
    line: int | None,
    errors: list[ErrorRecord],
    allow_zero: bool = False,
) -> int | None:
    if not _is_int(value):
        errors.append(_error("must be an integer", path, line))
        return None
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparator = "non-negative" if allow_zero else "positive"
        errors.append(_error(f"must be a {comparator} integer", path, line))
        return None
    return value


def _validate_pool_block(
    pools_doc: dict[str, Any],
    pools_text: str,
    pools_label: str,
    borrows_doc: dict[str, Any] | None,
    borrows_text: str,
    borrows_label: str,
    *,
    require_borrows: bool,
) -> ParameterSnapshot | None:
    errors: list[ErrorRecord] = []

    version = _positive_int(
        pools_doc.get("version"),
        path=f"{pools_label}.version",
        line=_line_for_text(pools_text, "version"),
        errors=errors,
    )
    alert_threshold = _positive_int(
        pools_doc.get("alert_threshold"),
        path=f"{pools_label}.alert_threshold",
        line=_line_for_text(pools_text, "alert_threshold"),
        errors=errors,
        allow_zero=True,
    )
    rate_window_ms = _positive_int(
        pools_doc.get("rate_window_ms"),
        path=f"{pools_label}.rate_window_ms",
        line=_line_for_text(pools_text, "rate_window_ms"),
        errors=errors,
    )
    percentage_scale = _positive_int(
        pools_doc.get("percentage_scale"),
        path=f"{pools_label}.percentage_scale",
        line=_line_for_text(pools_text, "percentage_scale"),
        errors=errors,
    )

    raw_pools = pools_doc.get("pools")
    if not isinstance(raw_pools, list) or not raw_pools:
        errors.append(
            _error(
                "pools must be a non-empty array",
                f"{pools_label}.pools",
                _line_for_text(pools_text, "pools"),
            )
        )
        raw_pools = []

    parsed_pools: dict[str, PoolConfig] = {}
    for index, raw_pool in enumerate(raw_pools):
        base_path = f"{pools_label}.pools[{index}]"
        if not isinstance(raw_pool, dict):
            errors.append(_error("pool entry must be an object", base_path, _line_for_text(pools_text, "pools")))
            continue
        pool_id = raw_pool.get("id")
        line = _line_for_text(pools_text, pool_id if isinstance(pool_id, str) else "pools")
        if not isinstance(pool_id, str) or not pool_id.strip():
            errors.append(_error("pool id must be a non-empty string", f"{base_path}.id", line))
            continue
        if pool_id in parsed_pools:
            errors.append(
                _error(
                    "pool id must be unique",
                    f"{base_path}.id",
                    line,
                    details={"pool_id": pool_id},
                )
            )
            continue
        max_connections = _positive_int(
            raw_pool.get("max_connections"),
            path=f"{base_path}.max_connections",
            line=line,
            errors=errors,
        )
        request_limit = _positive_int(
            raw_pool.get("per_connection_request_limit"),
            path=f"{base_path}.per_connection_request_limit",
            line=line,
            errors=errors,
        )
        idle_release_ms = _positive_int(
            raw_pool.get("idle_release_ms"),
            path=f"{base_path}.idle_release_ms",
            line=line,
            errors=errors,
            allow_zero=True,
        )
        read_buffer_limit = _positive_int(
            raw_pool.get("read_buffer_limit_bytes"),
            path=f"{base_path}.read_buffer_limit_bytes",
            line=line,
            errors=errors,
            allow_zero=True,
        )
        if None in (max_connections, request_limit, idle_release_ms, read_buffer_limit):
            continue
        parsed_pools[pool_id] = PoolConfig(
            pool_id=pool_id,
            max_connections=max_connections,
            per_connection_request_limit=request_limit,
            idle_release_ms=idle_release_ms,
            read_buffer_limit_bytes=read_buffer_limit,
        )

    raw_rules: Any
    if borrows_doc is None:
        if require_borrows:
            errors.append(_error("borrows document is required", borrows_label, 1))
        raw_rules = []
    else:
        borrow_version = _positive_int(
            borrows_doc.get("version"),
            path=f"{borrows_label}.version",
            line=_line_for_text(borrows_text, "version"),
            errors=errors,
        )
        if version is not None and borrow_version is not None and version != borrow_version:
            errors.append(
                _error(
                    "pools.version and borrows.version must match",
                    f"{borrows_label}.version",
                    _line_for_text(borrows_text, "version"),
                    details={"pools_version": version, "borrows_version": borrow_version},
                )
            )
        raw_rules = borrows_doc.get("borrows")
        if not isinstance(raw_rules, list):
            errors.append(
                _error(
                    "borrows must be an array",
                    f"{borrows_label}.borrows",
                    _line_for_text(borrows_text, "borrows"),
                )
            )
            raw_rules = []

    parsed_rules: list[BorrowRule] = []
    seen_pairs: set[tuple[str, str]] = set()
    rule_index_by_pair: dict[tuple[str, str], int] = {}
    for index, raw_rule in enumerate(raw_rules):
        base_path = f"{borrows_label}.borrows[{index}]"
        if not isinstance(raw_rule, dict):
            errors.append(_error("borrow rule must be an object", base_path, _line_for_text(borrows_text, "borrows")))
            continue
        lender = raw_rule.get("lender_pool")
        borrower = raw_rule.get("borrower_pool")
        line = _line_for_text(borrows_text, lender if isinstance(lender, str) else "borrows")
        if not isinstance(lender, str) or not lender.strip():
            errors.append(_error("lender_pool must be a non-empty string", f"{base_path}.lender_pool", line))
            continue
        if not isinstance(borrower, str) or not borrower.strip():
            errors.append(_error("borrower_pool must be a non-empty string", f"{base_path}.borrower_pool", line))
            continue
        if lender not in parsed_pools:
            errors.append(_error("lender_pool does not exist", f"{base_path}.lender_pool", line, details={"pool_id": lender}))
            continue
        if borrower not in parsed_pools:
            errors.append(
                _error("borrower_pool does not exist", f"{base_path}.borrower_pool", line, details={"pool_id": borrower})
            )
            continue
        if lender == borrower:
            errors.append(
                _error(
                    "self-borrow creates a cycle",
                    f"{base_path}.borrower_pool",
                    line,
                    business_code="E_BORROW",
                )
            )
            continue
        pair = (lender, borrower)
        if pair in seen_pairs:
            errors.append(
                _error(
                    "duplicate lender/borrower relation",
                    base_path,
                    line,
                    details={"lender_pool": lender, "borrower_pool": borrower},
                )
            )
            continue
        seen_pairs.add(pair)
        max_borrowed = _positive_int(
            raw_rule.get("max_borrowed"),
            path=f"{base_path}.max_borrowed",
            line=line,
            errors=errors,
        )
        deadline_ms = _positive_int(
            raw_rule.get("return_deadline_ms"),
            path=f"{base_path}.return_deadline_ms",
            line=line,
            errors=errors,
        )
        if max_borrowed is None or deadline_ms is None:
            continue
        lender_config = parsed_pools[lender]
        if max_borrowed > lender_config.max_connections:
            errors.append(
                _error(
                    "max_borrowed exceeds lender max_connections",
                    f"{base_path}.max_borrowed",
                    line,
                    details={
                        "max_borrowed": max_borrowed,
                        "lender_max_connections": lender_config.max_connections,
                    },
                )
            )
            continue
        parsed_rules.append(
            BorrowRule(
                lender_pool=lender,
                borrower_pool=borrower,
                max_borrowed=max_borrowed,
                return_deadline_ms=deadline_ms,
                source_index=index,
            )
        )
        rule_index_by_pair[pair] = index

    cycle = _find_cycle(parsed_pools, parsed_rules)
    if cycle:
        closing_lender, closing_borrower = cycle[-2], cycle[-1]
        rule_index = rule_index_by_pair.get((closing_lender, closing_borrower), 0)
        errors.append(
            _error(
                "borrow relations contain a cycle",
                f"{borrows_label}.borrows[{rule_index}]",
                _line_for_text(borrows_text, closing_lender),
                business_code="E_BORROW",
                details={"cycle": cycle},
            )
        )

    if errors:
        raise SpecValidationError(errors)
    assert version is not None
    assert alert_threshold is not None
    assert rate_window_ms is not None
    assert percentage_scale is not None
    return ParameterSnapshot(
        version=version,
        alert_threshold=alert_threshold,
        rate_window_ms=rate_window_ms,
        percentage_scale=percentage_scale,
        pools=parsed_pools,
        borrow_rules=tuple(parsed_rules),
    )


def _find_cycle(pools: dict[str, PoolConfig], rules: Iterable[BorrowRule]) -> list[str] | None:
    adjacency: dict[str, list[str]] = {pool_id: [] for pool_id in sorted(pools)}
    for rule in sorted(rules, key=lambda item: (item.lender_pool, item.borrower_pool, item.source_index)):
        adjacency[rule.lender_pool].append(rule.borrower_pool)
    state: dict[str, int] = {pool_id: 0 for pool_id in pools}
    stack: list[str] = []
    stack_index: dict[str, int] = {}

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack_index[node] = len(stack)
        stack.append(node)
        for neighbour in adjacency[node]:
            if state[neighbour] == 0:
                found = visit(neighbour)
                if found:
                    return found
            elif state[neighbour] == 1:
                start = stack_index[neighbour]
                return stack[start:] + [neighbour]
        stack.pop()
        stack_index.pop(node)
        state[node] = 2
        return None

    for pool_id in sorted(pools):
        if state[pool_id] == 0:
            found = visit(pool_id)
            if found:
                return found
    return None


def load_parameter_snapshot(pools_path: str | Path, borrows_path: str | Path) -> ParameterSnapshot:
    pools_doc, pools_text, pools_errors = _read_json(pools_path, "pools")
    borrows_doc, borrows_text, borrow_errors = _read_json(borrows_path, "borrows")
    errors = [*pools_errors, *borrow_errors]
    if errors:
        raise SpecValidationError(errors)
    assert pools_doc is not None
    assert borrows_doc is not None
    snapshot = _validate_pool_block(
        pools_doc,
        pools_text,
        "pools",
        borrows_doc,
        borrows_text,
        "borrows",
        require_borrows=True,
    )
    assert snapshot is not None
    return snapshot


def load_events(path: str | Path, snapshot: ParameterSnapshot) -> list[TraceEvent]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SpecValidationError([_error(f"cannot read events: {exc}", "events", 1)]) from exc

    events: list[TraceEvent] = []
    errors: list[ErrorRecord] = []
    seen_sequences: set[int] = set()
    established: set[str] = set()
    pool_last_timestamp: dict[str, int] = {}
    expected_seq = 1
    event_index = 0

    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        base_path = f"events[{event_index}]"
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            errors.append(_error(f"invalid NDJSON: {exc.msg}", base_path, line_number))
            event_index += 1
            continue
        if not isinstance(raw, dict):
            errors.append(_error("event must be an object", base_path, line_number))
            event_index += 1
            continue

        seq = raw.get("seq")
        if not _is_int(seq) or seq <= 0:
            errors.append(_error("seq must be a positive integer", f"{base_path}.seq", line_number))
            event_index += 1
            continue
        if seq in seen_sequences:
            errors.append(
                _error(
                    "seq must be unique",
                    f"{base_path}.seq",
                    line_number,
                    business_code="E_GAP",
                    details={"seq": seq},
                )
            )
        if seq != expected_seq:
            errors.append(
                _error(
                    "event sequences must be continuous",
                    f"{base_path}.seq",
                    line_number,
                    business_code="E_GAP",
                    details={"expected": expected_seq, "actual": seq},
                )
            )
        expected_seq = seq + 1 if seq >= expected_seq else expected_seq
        seen_sequences.add(seq)

        pool = raw.get("pool")
        if not isinstance(pool, str) or not pool.strip():
            errors.append(_error("pool must be a non-empty string", f"{base_path}.pool", line_number))
            pool = None
        elif pool not in snapshot.pools:
            errors.append(_error("unknown pool", f"{base_path}.pool", line_number, details={"pool_id": pool}))

        connection = raw.get("connection")
        if not isinstance(connection, str) or not connection.strip():
            errors.append(_error("connection must be a non-empty string", f"{base_path}.connection", line_number))
            connection = None

        raw_type = raw.get("type")
        event_type = CANONICAL_EVENT_TYPES.get(raw_type) if isinstance(raw_type, str) else None
        if event_type is None:
            errors.append(
                _error(
                    "invalid event type",
                    f"{base_path}.type",
                    line_number,
                    details={"allowed": sorted(CANONICAL_EVENT_TYPES)},
                )
            )

        timestamp = raw.get("timestamp_ms")
        if not _is_int(timestamp) or timestamp < 0:
            errors.append(_error("timestamp_ms must be a non-negative integer", f"{base_path}.timestamp_ms", line_number))
            timestamp = None
        bytes_count = raw.get("bytes")
        if not _is_int(bytes_count) or bytes_count < 0:
            errors.append(_error("bytes must be a non-negative integer", f"{base_path}.bytes", line_number))
            bytes_count = None

        if pool and timestamp is not None:
            previous = pool_last_timestamp.get(pool)
            if previous is not None and timestamp < previous:
                errors.append(
                    _error(
                        "timestamp_ms must not move backwards within a pool",
                        f"{base_path}.timestamp_ms",
                        line_number,
                        details={"pool_id": pool, "previous_timestamp_ms": previous, "timestamp_ms": timestamp},
                    )
                )
            pool_last_timestamp[pool] = timestamp

        if connection and event_type:
            if event_type == "establish":
                if connection in established:
                    errors.append(
                        _error(
                            "connection may only have one establishment",
                            f"{base_path}.connection",
                            line_number,
                            details={"connection_id": connection},
                        )
                    )
                established.add(connection)
            else:
                if connection not in established:
                    errors.append(
                        _error(
                            "establish must precede request, idle_release and close",
                            f"{base_path}.connection",
                            line_number,
                            details={"connection_id": connection, "event_type": event_type},
                        )
                    )

        if (
            event_type is not None
            and pool is not None
            and connection is not None
            and timestamp is not None
            and bytes_count is not None
        ):
            events.append(
                TraceEvent(
                    seq=seq,
                    pool=pool,
                    connection=connection,
                    event_type=event_type,
                    timestamp_ms=timestamp,
                    bytes_count=bytes_count,
                    line=line_number,
                    raw=raw,
                )
            )
        event_index += 1

    if errors:
        raise SpecValidationError(errors)
    return events


def load_reconfigurations(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    schedules: list[dict[str, Any]] = []
    errors: list[ErrorRecord] = []
    for path in paths:
        label = str(path)
        doc, text, read_errors = _read_json(path, label)
        if read_errors:
            errors.extend(read_errors)
            continue
        assert doc is not None
        after_seq = doc.get("after_event_seq")
        if not _is_int(after_seq) or after_seq < 0:
            errors.append(
                _error(
                    "after_event_seq must be a non-negative integer",
                    f"{label}.after_event_seq",
                    _line_for_text(text, "after_event_seq"),
                )
            )
            continue
        pools_doc = {
            key: doc.get(key)
            for key in ("version", "alert_threshold", "rate_window_ms", "percentage_scale", "pools")
        }
        borrows_doc = {"version": doc.get("version"), "borrows": doc.get("borrows")}
        try:
            snapshot = _validate_pool_block(
                pools_doc,
                text,
                label,
                borrows_doc,
                text,
                label,
                require_borrows=True,
            )
        except SpecValidationError as exc:
            errors.extend(exc.records)
            continue
        assert snapshot is not None
        schedules.append({"after_event_seq": after_seq, "snapshot": snapshot})
    if errors:
        raise SpecValidationError(errors)
    schedules.sort(key=lambda item: item["after_event_seq"])
    seen_after: set[int] = set()
    for schedule in schedules:
        after_seq = schedule["after_event_seq"]
        if after_seq in seen_after:
            raise SpecValidationError(
                [
                    _error(
                        "after_event_seq must be unique across reconfigurations",
                        "reconfigurations.after_event_seq",
                        None,
                        details={"after_event_seq": after_seq},
                    )
                ]
            )
        seen_after.add(after_seq)
    versions = [schedule["snapshot"].version for schedule in schedules]
    if versions != sorted(versions) or len(set(versions)) != len(versions):
        raise SpecValidationError(
            [
                _error(
                    "reconfiguration versions must be strictly increasing",
                    "reconfigurations.version",
                    None,
                    details={"versions": versions},
                )
            ]
        )
    return schedules


def deserialize_parameter_snapshot(data: dict[str, Any], *, label: str = "stored_parameter_snapshot") -> ParameterSnapshot:
    """Rebuild and revalidate a parameter snapshot stored in the SQLite ledger."""

    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    pools_doc = {
        key: data.get(key)
        for key in ("version", "alert_threshold", "rate_window_ms", "percentage_scale", "pools")
    }
    borrows_doc = {"version": data.get("version"), "borrows": data.get("borrows")}
    snapshot = _validate_pool_block(
        pools_doc,
        text,
        label,
        borrows_doc,
        text,
        label,
        require_borrows=True,
    )
    assert snapshot is not None
    return snapshot

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import DiagnosticError
from .model import (
    Budget,
    Dataset,
    Manifest,
    Predicate,
    PredicateCondition,
    Query,
    ReadPoint,
    RecordVersion,
    Transaction,
    VisibilityPolicy,
)
from .value_ops import version_value_digest


SUPPORTED_SCHEMA_VERSION = 1
VALID_OUTCOMES = ("committed", "rolledback")
VALID_OPS = ("gt", ">", "gte", ">=", "lt", "<", "lte", "<=", "eq", "==", "ne", "!=")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _fail(code: str, location: dict[str, Any], message: str, **details: Any) -> None:
    raise DiagnosticError(code, location, message, details)


def _require_mapping(value: Any, code: str, location: dict[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(code, location, f"{label} must be a JSON object")
    return value


def _require_list(value: Any, code: str, location: dict[str, Any], label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(code, location, f"{label} must be a JSON array")
    return value


def _require_nonempty_string(value: Any, code: str, location: dict[str, Any], label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code, location, f"{label} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, code: str, location: dict[str, Any], label: str) -> int:
    if not _is_int(value) or value < 0:
        _fail(code, location, f"{label} must be a non-negative integer")
    return value


def load_dataset(path: str | Path) -> Dataset:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise DiagnosticError(
            "E_QUERY",
            {"file": str(source), "field": "path"},
            "input file does not exist",
        ) from exc
    except json.JSONDecodeError as exc:
        raise DiagnosticError(
            "E_QUERY",
            {"file": str(source), "field": "json", "line": exc.lineno, "column": exc.colno},
            "input file is not valid JSON",
        ) from exc
    return dataset_from_dict(payload)


def dataset_from_dict(payload: Any) -> Dataset:
    root = _require_mapping(payload, "E_QUERY", {"file": "input", "field": "root"}, "dataset")
    raw_manifest = root.get("manifest", {})
    manifest_obj = _require_mapping(
        raw_manifest,
        "E_QUERY",
        {"file": "input", "field": "manifest"},
        "manifest",
    )
    schema_version = manifest_obj.get("schema_version", root.get("schema_version", SUPPORTED_SCHEMA_VERSION))
    if not _is_int(schema_version) or schema_version != SUPPORTED_SCHEMA_VERSION:
        _fail(
            "E_QUERY",
            {"file": "input", "field": "manifest.schema_version"},
            f"schema_version must be integer {SUPPORTED_SCHEMA_VERSION}",
        )
    version_fields = ("txn_set_version", "chain_version", "budget_version", "read_point_set_version")
    versions: dict[str, int] = {}
    for field_name in version_fields:
        versions[field_name] = _require_nonnegative_int(
            manifest_obj.get(field_name, 1),
            "E_QUERY",
            {"file": "input", "field": f"manifest.{field_name}"},
            field_name,
        )
    dataset_id = _require_nonempty_string(
        manifest_obj.get("dataset_id", "local-mvcc-sample"),
        "E_QUERY",
        {"file": "input", "field": "manifest.dataset_id"},
        "dataset_id",
    )
    manifest = Manifest(
        schema_version=schema_version,
        dataset_id=dataset_id,
        **versions,
    )

    transactions = _load_transactions(root.get("transactions"))
    txns_by_id = {item.txn_id: item for item in transactions}
    records, chains = _load_records(root, txns_by_id)
    read_points = _load_read_points(root.get("read_points"), txns_by_id)
    read_points_by_id = {item.read_point_id: item for item in read_points}
    queries = _load_queries(root.get("queries"), read_points_by_id, chains)
    queries_by_id = {item.query_id: item for item in queries}
    budget = _load_budget(root.get("budget", {}))
    visibility_policy = _load_visibility_policy(root.get("visibility_policy", {}))

    return Dataset(
        manifest=manifest,
        transactions=transactions,
        records=records,
        read_points=read_points,
        queries=queries,
        budget=budget,
        visibility_policy=visibility_policy,
        chains=chains,
        txns_by_id=txns_by_id,
        read_points_by_id=read_points_by_id,
        queries_by_id=queries_by_id,
    )


def _load_transactions(raw: Any) -> tuple[Transaction, ...]:
    items = _require_list(raw, "E_TXN", {"txn_id": "<all>", "field": "transactions"}, "transactions")
    if not items:
        _fail("E_TXN", {"txn_id": "<all>", "field": "transactions"}, "transactions must not be empty")

    result: list[Transaction] = []
    seen: set[str] = set()
    for index, raw_item in enumerate(items):
        location = {"txn_id": f"<index:{index}>", "field": "transactions", "index": index}
        item = _require_mapping(raw_item, "E_TXN", location, "transaction")
        txn_id = _require_nonempty_string(item.get("txn_id"), "E_TXN", location, "txn_id")
        location = {"txn_id": txn_id, "field": "transactions", "index": index}
        if txn_id in seen:
            _fail("E_TXN", location, "txn_id must be unique", duplicate_id=txn_id)
        seen.add(txn_id)
        start_seq = _require_nonnegative_int(item.get("start_seq"), "E_TXN", {**location, "field": "start_seq"}, "start_seq")
        outcome = item.get("outcome")
        if outcome not in VALID_OUTCOMES:
            _fail(
                "E_TXN",
                {**location, "field": "outcome"},
                "outcome must be committed or rolledback",
            )
        commit_seq: int | None = None
        if outcome == "committed":
            if "commit_seq" not in item or item.get("commit_seq") is None:
                _fail(
                    "E_TXN",
                    {**location, "field": "commit_seq"},
                    "committed transaction must declare commit_seq",
                )
            commit_seq = _require_nonnegative_int(
                item.get("commit_seq"),
                "E_TXN",
                {**location, "field": "commit_seq"},
                "commit_seq",
            )
            if commit_seq < start_seq:
                _fail(
                    "E_TXN",
                    {**location, "field": "commit_seq"},
                    "committed transaction commit_seq must not be smaller than start_seq",
                    start_seq=start_seq,
                    commit_seq=commit_seq,
                )
        elif "commit_seq" in item:
            _fail(
                "E_TXN",
                {**location, "field": "commit_seq"},
                "rolledback transaction must not declare commit_seq",
            )
        result.append(
            Transaction(
                txn_id=txn_id,
                start_seq=start_seq,
                outcome=outcome,
                commit_seq=commit_seq,
            )
        )
    return tuple(sorted(result, key=lambda item: item.txn_id))


def _load_records(
    root: dict[str, Any],
    txns_by_id: dict[str, Transaction],
) -> tuple[tuple[RecordVersion, ...], dict[str, tuple[RecordVersion, ...]]]:
    if "records" in root and "version_chains" in root:
        _fail(
            "E_CHAIN",
            {"row_key": "<all>", "field": "records"},
            "declare either records or version_chains, not both",
        )

    raw_versions: list[tuple[int, dict[str, Any], str]] = []
    if "version_chains" in root:
        chains_raw = _require_list(
            root.get("version_chains"),
            "E_CHAIN",
            {"row_key": "<all>", "field": "version_chains"},
            "version_chains",
        )
        seen_row_keys: set[str] = set()
        for chain_index, chain_raw in enumerate(chains_raw):
            chain_location = {"row_key": f"<index:{chain_index}>", "field": "version_chains"}
            chain_obj = _require_mapping(chain_raw, "E_CHAIN", chain_location, "version chain")
            row_key = _require_nonempty_string(chain_obj.get("row_key"), "E_CHAIN", chain_location, "row_key")
            if row_key in seen_row_keys:
                _fail("E_CHAIN", {"row_key": row_key, "field": "row_key"}, "row_key must identify one chain")
            seen_row_keys.add(row_key)
            versions_raw = _require_list(
                chain_obj.get("versions"),
                "E_CHAIN",
                {"row_key": row_key, "field": "versions"},
                "versions",
            )
            if not versions_raw:
                _fail(
                    "E_CHAIN",
                    {"row_key": row_key, "field": "versions"},
                    "version chain must contain at least one version",
                )
            for version_index, version_raw in enumerate(versions_raw):
                raw = dict(_require_mapping(
                    version_raw,
                    "E_CHAIN",
                    {"row_key": row_key, "field": "versions", "index": version_index},
                    "record version",
                ))
                declared_row = raw.get("row_key", row_key)
                if declared_row != row_key:
                    _fail(
                        "E_CHAIN",
                        {"row_key": declared_row, "field": "row_key"},
                        "nested version row_key must match its chain row_key",
                        chain_row_key=row_key,
                    )
                raw["row_key"] = row_key
                raw_versions.append((len(raw_versions), raw, row_key))
    else:
        records_raw = _require_list(
            root.get("records"),
            "E_CHAIN",
            {"row_key": "<all>", "field": "records"},
            "records",
        )
        for index, record_raw in enumerate(records_raw):
            raw_versions.append(
                (
                    index,
                    _require_mapping(
                        record_raw,
                        "E_CHAIN",
                        {"row_key": f"<index:{index}>", "field": "records"},
                        "record version",
                    ),
                    f"<index:{index}>",
                )
            )

    grouped: dict[str, list[tuple[int, RecordVersion]]] = {}
    seen_keys: set[tuple[str, int]] = set()
    for output_index, raw, fallback_row in raw_versions:
        initial_location = {"row_key": fallback_row, "field": "records", "index": output_index}
        item = _require_mapping(raw, "E_CHAIN", initial_location, "record version")
        row_key = _require_nonempty_string(item.get("row_key"), "E_CHAIN", initial_location, "row_key")
        version_field = "version_no" if "version_no" in item else "version"
        version_no = _require_nonnegative_int(
            item.get(version_field),
            "E_CHAIN",
            {"row_key": row_key, "field": version_field},
            "version_no",
        )
        location = {"row_key": row_key, "version_no": version_no, "field": "record"}
        key = (row_key, version_no)
        if key in seen_keys:
            _fail(
                "E_CHAIN",
                location,
                "version_no must be strictly unique inside a row chain",
            )
        seen_keys.add(key)
        txn_id = _require_nonempty_string(item.get("txn_id"), "E_CHAIN", {**location, "field": "txn_id"}, "txn_id")
        if txn_id not in txns_by_id:
            _fail(
                "E_CHAIN",
                {**location, "field": "txn_id"},
                "record references an unknown transaction",
                txn_id=txn_id,
            )
        value_summary = _require_mapping(
            item.get("value_summary"),
            "E_CHAIN",
            {**location, "field": "value_summary"},
            "value_summary",
        )
        computed_digest = version_value_digest(value_summary)
        declared_digest = item.get("value_digest")
        if declared_digest is None:
            value_digest = computed_digest
        else:
            if not isinstance(declared_digest, str) or declared_digest != computed_digest:
                _fail(
                    "E_CHAIN",
                    {**location, "field": "value_digest"},
                    "declared value_digest does not match value_summary",
                    expected=computed_digest,
                    actual=declared_digest,
                )
            value_digest = declared_digest
        prev_version = item.get("prev_version")
        if prev_version is not None:
            prev_version = _require_nonnegative_int(
                prev_version,
                "E_CHAIN",
                {**location, "field": "prev_version"},
                "prev_version",
            )
        record = RecordVersion(
            row_key=row_key,
            version_no=version_no,
            txn_id=txn_id,
            value_summary=dict(value_summary),
            value_digest=value_digest,
            prev_version=prev_version,
        )
        grouped.setdefault(row_key, []).append((output_index, record))

    if not grouped:
        _fail("E_CHAIN", {"row_key": "<all>", "field": "records"}, "records must not be empty")

    chains: dict[str, tuple[RecordVersion, ...]] = {}
    all_records: list[RecordVersion] = []
    for row_key in sorted(grouped):
        entries = sorted(grouped[row_key], key=lambda pair: pair[1].version_no)
        ordered = tuple(item for _, item in entries)
        _detect_cycle(row_key, ordered)
        for index, record in enumerate(ordered):
            if index == 0:
                if record.prev_version is not None:
                    _fail(
                        "E_CHAIN",
                        {"row_key": row_key, "version_no": record.version_no, "field": "prev_version"},
                        "first version must have a null previous-version reference",
                    )
            else:
                expected_prev = ordered[index - 1].version_no
                if record.prev_version != expected_prev:
                    _fail(
                        "E_CHAIN",
                        {"row_key": row_key, "version_no": record.version_no, "field": "prev_version"},
                        "previous-version reference must point to the immediately previous version",
                        expected=expected_prev,
                        actual=record.prev_version,
                    )
        chains[row_key] = ordered
        all_records.extend(ordered)

    records = tuple(sorted(all_records, key=lambda item: (item.row_key, item.version_no)))
    return records, chains


def _detect_cycle(row_key: str, versions: tuple[RecordVersion, ...]) -> None:
    by_version = {item.version_no: item for item in versions}
    state: dict[int, int] = {}

    def visit(version_no: int, path: list[int]) -> None:
        marker = state.get(version_no, 0)
        if marker == 1:
            cycle_start = path.index(version_no)
            cycle = path[cycle_start:] + [version_no]
            _fail(
                "E_CHAIN",
                {"row_key": row_key, "version_no": version_no, "field": "prev_version"},
                "previous-version references form a cycle",
                cycle=cycle,
            )
        if marker == 2:
            return
        state[version_no] = 1
        previous = by_version[version_no].prev_version
        if previous is not None and previous in by_version:
            visit(previous, path + [version_no])
        state[version_no] = 2

    for record in versions:
        visit(record.version_no, [])


def _load_read_points(raw: Any, txns_by_id: dict[str, Transaction]) -> tuple[ReadPoint, ...]:
    items = _require_list(raw, "E_QUERY", {"read_point_id": "<all>", "field": "read_points"}, "read_points")
    if not items:
        _fail("E_QUERY", {"read_point_id": "<all>", "field": "read_points"}, "read_points must not be empty")
    seen: set[str] = set()
    result: list[ReadPoint] = []
    for index, raw_item in enumerate(items):
        location = {"read_point_id": f"<index:{index}>", "field": "read_points", "index": index}
        item = _require_mapping(raw_item, "E_QUERY", location, "read point")
        read_point_id = _require_nonempty_string(item.get("read_point_id"), "E_QUERY", location, "read_point_id")
        location = {"read_point_id": read_point_id, "field": "read_points"}
        if read_point_id in seen:
            _fail("E_QUERY", location, "read_point_id must be unique")
        seen.add(read_point_id)
        snapshot_seq = _require_nonnegative_int(
            item.get("snapshot_seq"),
            "E_QUERY",
            {**location, "field": "snapshot_seq"},
            "snapshot_seq",
        )
        reader_txn_id = _require_nonempty_string(
            item.get("reader_txn_id"),
            "E_QUERY",
            {**location, "field": "reader_txn_id"},
            "reader_txn_id",
        )
        if reader_txn_id not in txns_by_id:
            _fail(
                "E_QUERY",
                {**location, "field": "reader_txn_id"},
                "read point references an unknown reader transaction",
                reader_txn_id=reader_txn_id,
            )
        result.append(
            ReadPoint(
                read_point_id=read_point_id,
                snapshot_seq=snapshot_seq,
                reader_txn_id=reader_txn_id,
            )
        )
    return tuple(sorted(result, key=lambda item: item.read_point_id))


def _load_queries(
    raw: Any,
    read_points_by_id: dict[str, ReadPoint],
    chains: dict[str, tuple[RecordVersion, ...]],
) -> tuple[Query, ...]:
    items = _require_list(raw, "E_QUERY", {"query_id": "<all>", "field": "queries"}, "queries")
    if not items:
        _fail("E_QUERY", {"query_id": "<all>", "field": "queries"}, "queries must not be empty")
    seen: set[str] = set()
    result: list[Query] = []
    for index, raw_item in enumerate(items):
        location = {"query_id": f"<index:{index}>", "field": "queries", "index": index}
        item = _require_mapping(raw_item, "E_QUERY", location, "query")
        query_id = _require_nonempty_string(item.get("query_id"), "E_QUERY", location, "query_id")
        location = {"query_id": query_id, "field": "queries"}
        if query_id in seen:
            _fail("E_QUERY", location, "query_id must be unique")
        seen.add(query_id)
        read_point_id = _require_nonempty_string(
            item.get("read_point_id"),
            "E_QUERY",
            {**location, "field": "read_point_id"},
            "read_point_id",
        )
        if read_point_id not in read_points_by_id:
            _fail(
                "E_QUERY",
                {**location, "field": "read_point_id"},
                "query references an unknown read point",
            )
        row_key = _require_nonempty_string(item.get("row_key"), "E_QUERY", {**location, "field": "row_key"}, "row_key")
        if row_key not in chains:
            _fail("E_QUERY", {**location, "field": "row_key"}, "query references an unknown row chain")
        predicate = _parse_predicate(item.get("predicate"), query_id)
        priority = item.get("priority", 0)
        if not _is_int(priority):
            _fail("E_QUERY", {**location, "field": "priority"}, "priority must be an integer")
        result.append(
            Query(
                query_id=query_id,
                read_point_id=read_point_id,
                row_key=row_key,
                predicate=predicate,
                priority=priority,
            )
        )
    return tuple(sorted(result, key=lambda item: item.query_id))


def _parse_predicate(raw: Any, query_id: str) -> Predicate:
    location = {"query_id": query_id, "field": "predicate"}
    item = _require_mapping(raw, "E_QUERY", location, "predicate")
    if "conditions" in item:
        mode = item.get("mode", "all")
        conditions_raw = item.get("conditions")
    elif "column" in item:
        mode = "all"
        conditions_raw = [item]
    else:
        _fail("E_QUERY", location, "predicate must declare conditions")
    if mode not in ("all", "any"):
        _fail("E_QUERY", {**location, "field": "predicate.mode"}, "predicate mode must be all or any")
    conditions_items = _require_list(
        conditions_raw,
        "E_QUERY",
        {**location, "field": "predicate.conditions"},
        "predicate conditions",
    )
    if not conditions_items:
        _fail("E_QUERY", {**location, "field": "predicate.conditions"}, "predicate must contain at least one condition")
    conditions: list[PredicateCondition] = []
    for index, condition_raw in enumerate(conditions_items):
        condition_location = {**location, "field": f"predicate.conditions[{index}]"}
        condition = _require_mapping(condition_raw, "E_QUERY", condition_location, "predicate condition")
        column = _require_nonempty_string(condition.get("column"), "E_QUERY", condition_location, "column")
        op = condition.get("op")
        if op not in VALID_OPS:
            _fail("E_QUERY", {**condition_location, "field": "op"}, "unsupported predicate operator")
        value = condition.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail("E_QUERY", {**condition_location, "field": "value"}, "predicate value must be numeric")
        conditions.append(PredicateCondition(column=column, op=op, value=value))
    return Predicate(mode=mode, conditions=tuple(conditions))


def _load_budget(raw: Any) -> Budget:
    location = {"field": "budget"}
    item = _require_mapping(raw, "E_QUERY", location, "budget")
    global_cost_budget = _require_nonnegative_int(
        item.get("global_cost_budget", 100),
        "E_QUERY",
        {**location, "field": "global_cost_budget"},
        "global_cost_budget",
    )
    round_cost_budget = _require_nonnegative_int(
        item.get("round_cost_budget", global_cost_budget),
        "E_QUERY",
        {**location, "field": "round_cost_budget"},
        "round_cost_budget",
    )
    global_concurrency = _require_nonnegative_int(
        item.get("global_concurrency", 4),
        "E_QUERY",
        {**location, "field": "global_concurrency"},
        "global_concurrency",
    )
    per_read_point_concurrency = _require_nonnegative_int(
        item.get("per_read_point_concurrency", 1),
        "E_QUERY",
        {**location, "field": "per_read_point_concurrency"},
        "per_read_point_concurrency",
    )
    task_timeout_ms = _require_nonnegative_int(
        item.get("task_timeout_ms", 5000),
        "E_QUERY",
        {**location, "field": "task_timeout_ms"},
        "task_timeout_ms",
    )
    if global_cost_budget <= 0:
        _fail("E_QUERY", {**location, "field": "global_cost_budget"}, "global_cost_budget must be positive")
    if round_cost_budget <= 0:
        _fail("E_QUERY", {**location, "field": "round_cost_budget"}, "round_cost_budget must be positive")
    if global_concurrency <= 0:
        _fail("E_QUERY", {**location, "field": "global_concurrency"}, "global_concurrency must be positive")
    if per_read_point_concurrency <= 0:
        _fail(
            "E_QUERY",
            {**location, "field": "per_read_point_concurrency"},
            "per_read_point_concurrency must be positive",
        )
    if task_timeout_ms <= 0:
        _fail("E_QUERY", {**location, "field": "task_timeout_ms"}, "task_timeout_ms must be positive")
    return Budget(
        global_cost_budget=global_cost_budget,
        round_cost_budget=round_cost_budget,
        global_concurrency=global_concurrency,
        per_read_point_concurrency=per_read_point_concurrency,
        task_timeout_ms=task_timeout_ms,
    )


def _load_visibility_policy(raw: Any) -> VisibilityPolicy:
    location = {"field": "visibility_policy"}
    item = _require_mapping(raw, "E_QUERY", location, "visibility_policy")
    own_uncommitted_visible = item.get("own_uncommitted_visible", True)
    own_write_ignores_snapshot = item.get("own_write_ignores_snapshot", True)
    if not isinstance(own_uncommitted_visible, bool):
        _fail(
            "E_QUERY",
            {**location, "field": "own_uncommitted_visible"},
            "own_uncommitted_visible must be boolean",
        )
    if not isinstance(own_write_ignores_snapshot, bool):
        _fail(
            "E_QUERY",
            {**location, "field": "own_write_ignores_snapshot"},
            "own_write_ignores_snapshot must be boolean",
        )
    declaration = item.get(
        "declaration",
        "Read-your-own-writes is explicit: a committed self version may ignore snapshot_seq when own_write_ignores_snapshot is true.",
    )
    if not isinstance(declaration, str) or not declaration.strip():
        _fail("E_QUERY", {**location, "field": "declaration"}, "visibility policy declaration must be non-empty")
    return VisibilityPolicy(
        own_uncommitted_visible=own_uncommitted_visible,
        own_write_ignores_snapshot=own_write_ignores_snapshot,
        declaration=declaration,
    )

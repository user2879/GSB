"""Loading and strong validation of columnar tables and query sets.

Any unknown field or type mismatch rejects the whole input with E_FORMAT
and reports the exact JSON path of the offending field.
"""

import json

import numpy as np

from .errors import PipelineError

SUPPORTED_TYPES = ("int64", "float64", "string")
TABLE_FIELDS = {"name", "timestamp_column", "columns"}
COLUMN_FIELDS = {"name", "type", "data", "null_bitmap"}
QUERY_FIELDS = {"id", "kind", "table", "group_by", "partition_by",
                "value_column", "aggregates", "interval", "max_rows",
                "out_of_order"}
INTERVAL_FIELDS = {"preceding", "following"}
OOO_FIELDS = {"policy", "max_lateness"}
SESSION_FIELDS = {"scale", "truncation", "batch_size",
                  "max_out_of_order_ratio", "window_aligned_batches",
                  "arrival"}
ARRIVAL_FIELDS = {"shuffle_seed", "shuffle_window"}
AGG_FUNCTIONS = ("count", "sum", "min", "max", "mean")
OOO_POLICIES = ("recompute", "drop")
TRUNC_DIRECTIONS = ("floor", "ceil", "toward_zero")

DEFAULT_SESSION = {
    "scale": 1000,
    "truncation": "toward_zero",
    "batch_size": 8,
    "max_out_of_order_ratio": 0.5,
    "window_aligned_batches": False,
    "arrival": {"shuffle_seed": None, "shuffle_window": 0},
}


def _err(path, message):
    raise PipelineError("E_FORMAT", f"{path}: {message}", field_path=path)


def _check_unknown(obj, allowed, path):
    if not isinstance(obj, dict):
        _err(path, f"expected object, got {type(obj).__name__}")
    for key in obj:
        if key not in allowed:
            _err(f"{path}.{key}", "unknown field")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value, path, minimum=None):
    if not _is_int(value):
        _err(path, f"expected integer, got {value!r}")
    if minimum is not None and value < minimum:
        _err(path, f"value {value} below minimum {minimum}")
    return value


class Column:
    def __init__(self, name, ctype, values, null_bitmap):
        self.name = name
        self.type = ctype
        self.values = values          # numpy array
        self.null_bitmap = null_bitmap  # numpy bool array

    def is_null(self, idx):
        return bool(self.null_bitmap[idx])

    def get(self, idx):
        return None if self.null_bitmap[idx] else self.values[idx].item() \
            if hasattr(self.values[idx], "item") else self.values[idx]


class Table:
    def __init__(self, name, timestamp_column, columns):
        self.name = name
        self.timestamp_column = timestamp_column
        self.columns = columns  # dict name -> Column

    @property
    def row_count(self):
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())).values)

    def column(self, name):
        return self.columns[name]


def _validate_column(obj, path):
    _check_unknown(obj, COLUMN_FIELDS, path)
    for field in ("name", "type", "data"):
        if field not in obj:
            _err(f"{path}.{field}", "missing required field")
    name = obj["name"]
    if not isinstance(name, str) or not name:
        _err(f"{path}.name", "column name must be a non-empty string")
    ctype = obj["type"]
    if ctype not in SUPPORTED_TYPES:
        _err(f"{path}.type", f"unsupported column type {ctype!r}")
    data = obj["data"]
    if not isinstance(data, list):
        _err(f"{path}.data", "column data must be a list")
    if ctype == "int64":
        for pos, item in enumerate(data):
            if not _is_int(item):
                _err(f"{path}.data[{pos}]", f"expected int, got {item!r}")
        values = np.array(data, dtype=np.int64)
    elif ctype == "float64":
        for pos, item in enumerate(data):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                _err(f"{path}.data[{pos}]", f"expected number, got {item!r}")
        values = np.array(data, dtype=np.float64)
    else:
        for pos, item in enumerate(data):
            if not isinstance(item, str):
                _err(f"{path}.data[{pos}]", f"expected string, got {item!r}")
        values = np.array(data, dtype=object)
    bitmap = obj.get("null_bitmap")
    if bitmap is None:
        null_bitmap = np.zeros(len(data), dtype=bool)
    else:
        if not isinstance(bitmap, list):
            _err(f"{path}.null_bitmap", "null bitmap must be a list")
        if len(bitmap) != len(data):
            _err(f"{path}.null_bitmap",
                 f"null bitmap length {len(bitmap)} != row count {len(data)}")
        for pos, bit in enumerate(bitmap):
            if bit not in (0, 1):
                _err(f"{path}.null_bitmap[{pos}]", "bitmap entries must be 0/1")
        null_bitmap = np.array(bitmap, dtype=bool)
    return Column(name, ctype, values, null_bitmap)


def validate_tables(doc):
    """Validate a {"tables": [...]} document; return {name: Table}."""
    _check_unknown(doc, {"tables"}, "$")
    tables = doc.get("tables")
    if not isinstance(tables, list) or not tables:
        _err("$.tables", "expected a non-empty list of tables")
    result = {}
    for tpos, tobj in enumerate(tables):
        tpath = f"$.tables[{tpos}]"
        _check_unknown(tobj, TABLE_FIELDS, tpath)
        for field in ("name", "timestamp_column", "columns"):
            if field not in tobj:
                _err(f"{tpath}.{field}", "missing required field")
        name = tobj["name"]
        if not isinstance(name, str) or not name:
            _err(f"{tpath}.name", "table name must be a non-empty string")
        if name in result:
            _err(f"{tpath}.name", f"duplicate table name {name!r}")
        cols = tobj["columns"]
        if not isinstance(cols, list) or not cols:
            _err(f"{tpath}.columns", "expected a non-empty list of columns")
        columns = {}
        for cpos, cobj in enumerate(cols):
            column = _validate_column(cobj, f"{tpath}.columns[{cpos}]")
            if column.name in columns:
                _err(f"{tpath}.columns[{cpos}].name",
                     f"duplicate column name {column.name!r}")
            columns[column.name] = column
        ts_col = tobj["timestamp_column"]
        if not isinstance(ts_col, str) or ts_col not in columns:
            _err(f"{tpath}.timestamp_column",
                 f"timestamp column {ts_col!r} not found")
        if columns[ts_col].type != "int64":
            _err(f"{tpath}.timestamp_column",
                 "timestamp column must have type int64")
        result[name] = Table(name, ts_col, columns)
    return result


class Query:
    def __init__(self, qid, kind, table, group_by, partition_by,
                 value_column, aggregates, interval, max_rows, out_of_order):
        self.qid = qid
        self.kind = kind
        self.table = table
        self.group_by = group_by
        self.partition_by = partition_by
        self.value_column = value_column
        self.aggregates = aggregates
        self.interval = interval
        self.max_rows = max_rows
        self.out_of_order = out_of_order


def _validate_query(obj, path, tables):
    _check_unknown(obj, QUERY_FIELDS, path)
    for field in ("id", "kind", "table", "value_column"):
        if field not in obj:
            _err(f"{path}.{field}", "missing required field")
    qid = obj["id"]
    if not isinstance(qid, str) or not qid:
        _err(f"{path}.id", "query id must be a non-empty string")
    kind = obj["kind"]
    if kind not in ("aggregate", "window"):
        _err(f"{path}.kind", f"unsupported query kind {kind!r}")
    table_name = obj["table"]
    if not isinstance(table_name, str) or table_name not in tables:
        _err(f"{path}.table", f"unknown table {table_name!r}")
    table = tables[table_name]
    value_column = obj["value_column"]
    if not isinstance(value_column, str) or value_column not in table.columns:
        _err(f"{path}.value_column",
             f"value column {value_column!r} not found in table")
    if table.columns[value_column].type not in ("int64", "float64"):
        _err(f"{path}.value_column", "value column must be numeric")

    group_by, partition_by = [], []
    aggregates = list(AGG_FUNCTIONS)
    interval, max_rows, out_of_order = None, None, None

    if kind == "aggregate":
        gb = obj.get("group_by")
        if not isinstance(gb, list) or not gb:
            _err(f"{path}.group_by", "aggregate query requires group_by")
        for pos, col in enumerate(gb):
            if not isinstance(col, str) or col not in table.columns:
                _err(f"{path}.group_by[{pos}]",
                     f"group key column {col!r} not found")
        group_by = list(gb)
        aggs = obj.get("aggregates", AGG_FUNCTIONS)
        if not isinstance(aggs, list) or not aggs:
            _err(f"{path}.aggregates", "expected a non-empty list")
        for pos, fn in enumerate(aggs):
            if fn not in AGG_FUNCTIONS:
                _err(f"{path}.aggregates[{pos}]",
                     f"unsupported aggregate {fn!r}")
        aggregates = list(aggs)
    else:
        pb = obj.get("partition_by")
        if not isinstance(pb, list) or not pb:
            _err(f"{path}.partition_by", "window query requires partition_by")
        for pos, col in enumerate(pb):
            if not isinstance(col, str) or col not in table.columns:
                _err(f"{path}.partition_by[{pos}]",
                     f"partition column {col!r} not found")
        partition_by = list(pb)
        iv = obj.get("interval")
        if iv is None:
            _err(f"{path}.interval", "window query requires interval")
        _check_unknown(iv, INTERVAL_FIELDS, f"{path}.interval")
        for field in ("preceding", "following"):
            if field not in iv:
                _err(f"{path}.interval.{field}", "missing required field")
        preceding = _require_int(iv["preceding"],
                                 f"{path}.interval.preceding", 0)
        following = _require_int(iv["following"],
                                 f"{path}.interval.following", 0)
        interval = {"preceding": preceding, "following": following}
        max_rows = _require_int(obj.get("max_rows"), f"{path}.max_rows", 1)
        ooo = obj.get("out_of_order", {})
        _check_unknown(ooo, OOO_FIELDS, f"{path}.out_of_order")
        policy = ooo.get("policy", "recompute")
        if policy not in OOO_POLICIES:
            _err(f"{path}.out_of_order.policy",
                 f"unsupported policy {policy!r}")
        max_lateness = _require_int(ooo.get("max_lateness", 0),
                                    f"{path}.out_of_order.max_lateness", 0)
        out_of_order = {"policy": policy, "max_lateness": max_lateness}
    return Query(qid, kind, table_name, group_by, partition_by,
                 value_column, aggregates, interval, max_rows, out_of_order)


def _validate_session(obj, path):
    _check_unknown(obj, SESSION_FIELDS, path)
    session = json.loads(json.dumps(DEFAULT_SESSION))
    session.update(obj)
    _require_int(session["scale"], f"{path}.scale", 1)
    if session["truncation"] not in TRUNC_DIRECTIONS:
        _err(f"{path}.truncation",
             f"unsupported truncation {session['truncation']!r}")
    _require_int(session["batch_size"], f"{path}.batch_size", 1)
    ratio = session["max_out_of_order_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) \
            or not 0.0 <= ratio <= 1.0:
        _err(f"{path}.max_out_of_order_ratio",
             "expected a number in [0, 1]")
    if not isinstance(session["window_aligned_batches"], bool):
        _err(f"{path}.window_aligned_batches", "expected a boolean")
    arrival = session["arrival"]
    _check_unknown(arrival, ARRIVAL_FIELDS, f"{path}.arrival")
    seed = arrival["shuffle_seed"]
    if seed is not None and not _is_int(seed):
        _err(f"{path}.arrival.shuffle_seed", "expected integer or null")
    _require_int(arrival["shuffle_window"],
                 f"{path}.arrival.shuffle_window", 0)
    return session


class QuerySet:
    def __init__(self, queries, session):
        self.queries = queries
        self.session = session


def validate_queries(doc, tables):
    """Validate a {"queries": [...], "session": {...}} document."""
    _check_unknown(doc, {"queries", "session"}, "$")
    raw_queries = doc.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        _err("$.queries", "expected a non-empty list of queries")
    queries = []
    seen_ids = set()
    for pos, obj in enumerate(raw_queries):
        query = _validate_query(obj, f"$.queries[{pos}]", tables)
        if query.qid in seen_ids:
            _err(f"$.queries[{pos}].id", f"duplicate query id {query.qid!r}")
        seen_ids.add(query.qid)
        queries.append(query)
    session = _validate_session(doc.get("session", {}), "$.session")
    return QuerySet(queries, session)


def load_tables_file(path):
    with open(path, "r", encoding="utf-8") as fh:
        return validate_tables(json.load(fh))


def load_queries_file(path, tables):
    with open(path, "r", encoding="utf-8") as fh:
        return validate_queries(json.load(fh), tables)

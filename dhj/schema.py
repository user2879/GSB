"""Load and strongly validate table definitions, sample rows and join query."""

import re
from dataclasses import dataclass

from .errors import SchemaError
from .jsonio import digest_hex

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
DTYPES = ("int", "float")
OPS = ("=", "!=", "<", "<=", ">", ">=")


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str


@dataclass
class Table:
    name: str
    columns: list
    rows: list
    null_bitmap: list
    digest: str = ""

    @property
    def dtype_map(self):
        return {col.name: col.dtype for col in self.columns}

    def row_digests_input(self):
        return {
            "columns": [[c.name, c.dtype] for c in self.columns],
            "rows": self.rows,
            "null_bitmap": self.null_bitmap,
        }


@dataclass
class Predicate:
    table: str
    column: str
    op: str
    value: object

    @property
    def ref(self):
        return f"{self.table}.{self.column}"


@dataclass
class Query:
    query_id: str
    left_table: str
    right_table: str
    join_key: str
    predicates: dict
    projections: list
    left_key_dtype: str = ""
    right_key_dtype: str = ""

    def all_predicates(self):
        for side in ("left", "right"):
            for pred in self.predicates.get(side, []):
                yield side, pred


@dataclass
class Spec:
    tables: dict
    query: Query
    source_digest: str = ""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return (_is_int(value) or isinstance(value, float)) and not isinstance(value, bool)


def _check_identifier(name, path, errors):
    if not isinstance(name, str) or not IDENT_RE.match(name or ""):
        errors.append(f"{path}: illegal identifier {name!r}")
        return False
    return True


def _validate_table(name, raw, errors):
    base = f"tables.{name}"
    if not isinstance(raw, dict):
        errors.append(f"{base}: table definition must be an object")
        return None
    if not _check_identifier(name, f"{base}.name", errors):
        return None

    cols_raw = raw.get("columns")
    if not isinstance(cols_raw, list) or not cols_raw:
        errors.append(f"{base}.columns: non-empty column list required")
        return None

    columns = []
    seen = set()
    for idx, col_raw in enumerate(cols_raw):
        cpath = f"{base}.columns[{idx}]"
        if not isinstance(col_raw, dict):
            errors.append(f"{cpath}: column must be an object")
            continue
        cname = col_raw.get("name")
        dtype = col_raw.get("dtype")
        ok_name = _check_identifier(cname, f"{cpath}.name", errors)
        if ok_name and cname in seen:
            errors.append(f"{cpath}.name: duplicate column {cname!r}")
        elif ok_name:
            seen.add(cname)
        if dtype not in DTYPES:
            errors.append(f"{cpath}.dtype: unsupported dtype {dtype!r}")
        else:
            columns.append(Column(cname, dtype))

    if len(columns) != len(cols_raw):
        return None

    rows_raw = raw.get("rows")
    bitmap_raw = raw.get("null_bitmap")
    if not isinstance(rows_raw, list):
        errors.append(f"{base}.rows: row list required")
        return None
    if not isinstance(bitmap_raw, list) or len(bitmap_raw) != len(rows_raw):
        errors.append(
            f"{base}.null_bitmap: length must equal row count "
            f"({len(bitmap_raw) if isinstance(bitmap_raw, list) else 'n/a'} "
            f"vs {len(rows_raw)})"
        )
        return None

    dtype_map = {c.name: c.dtype for c in columns}
    rows = []
    for ridx, row_raw in enumerate(rows_raw):
        rpath = f"{base}.rows[{ridx}]"
        if not isinstance(row_raw, dict):
            errors.append(f"{rpath}: row must be an object")
            continue
        if set(row_raw.keys()) != set(dtype_map.keys()):
            errors.append(f"{rpath}: row keys must exactly match declared columns")
            continue
        row = {}
        for col in columns:
            vpath = f"{rpath}.{col.name}"
            value = row_raw[col.name]
            if value is None:
                row[col.name] = None
                continue
            if col.dtype == "int" and not _is_int(value):
                errors.append(f"{vpath}: expected int, got {type(value).__name__}")
            elif col.dtype == "float" and not _is_number(value):
                errors.append(f"{vpath}: expected number, got {type(value).__name__}")
            else:
                row[col.name] = value
        rows.append(row)

    bitmap = []
    for bidx, bit in enumerate(bitmap_raw):
        bpath = f"{base}.null_bitmap[{bidx}]"
        if not _is_int(bit) or bit not in (0, 1):
            errors.append(f"{bpath}: null bitmap entries must be 0 or 1")
            continue
        bitmap.append(bit)
        if bit == 0 and rows and bidx < len(rows):
            null_cols = [c.name for c in columns if rows[bidx][c.name] is None]
            if not null_cols:
                errors.append(
                    f"{bpath}: bitmap marks row null but no column value is null"
                )
        if bit == 1 and rows and bidx < len(rows) and any(
            rows[bidx][c.name] is None for c in columns
        ):
            errors.append(f"{bpath}: bitmap marks row valid but a value is null")

    if errors:
        return None
    table = Table(name=name, columns=columns, rows=rows, null_bitmap=bitmap)
    table.digest = digest_hex(table.row_digests_input())
    return table


def _validate_query(raw, tables, errors):
    if not isinstance(raw, dict):
        errors.append("query: query definition must be an object")
        return None
    qid = raw.get("query_id")
    _check_identifier(qid, "query.query_id", errors)

    left = raw.get("left_table")
    right = raw.get("right_table")
    for side_name, table_name in (("left_table", left), ("right_table", right)):
        if not isinstance(table_name, str) or table_name not in tables:
            errors.append(f"query.{side_name}: unknown table {table_name!r}")
    if left == right and isinstance(left, str):
        errors.append("query.left_table/right_table: self joins are not supported")

    join_key = raw.get("join_key")
    if not isinstance(join_key, str) or not IDENT_RE.match(join_key or ""):
        errors.append("query.join_key: illegal column name")
    else:
        for side_name, table_name in (("left_table", left), ("right_table", right)):
            if isinstance(table_name, str) and table_name in tables:
                if join_key not in tables[table_name].dtype_map:
                    errors.append(
                        f"query.join_key: missing on {side_name} "
                        f"table {table_name}"
                    )

    predicates = {"left": [], "right": []}
    preds_raw = raw.get("predicates", {})
    if not isinstance(preds_raw, dict):
        errors.append("query.predicates: must be an object with left/right lists")
    else:
        for side in ("left", "right"):
            side_table = left if side == "left" else right
            side_preds = preds_raw.get(side, [])
            if not isinstance(side_preds, list):
                errors.append(f"query.predicates.{side}: must be a list")
                continue
            for pidx, pred_raw in enumerate(side_preds):
                ppath = f"query.predicates.{side}[{pidx}]"
                if not isinstance(pred_raw, dict):
                    errors.append(f"{ppath}: predicate must be an object")
                    continue
                ref = pred_raw.get("ref")
                op = pred_raw.get("op")
                value = pred_raw.get("value")
                if not isinstance(ref, str) or ref.count(".") != 1:
                    errors.append(f"{ppath}.ref: expected table.column reference")
                    continue
                ptable, pcol = ref.split(".")
                if ptable != side_table:
                    errors.append(
                        f"{ppath}.ref: {ref!r} does not belong to {side} table"
                    )
                if op not in OPS:
                    errors.append(f"{ppath}.op: unsupported operator {op!r}")
                dtype = None
                if (
                    isinstance(side_table, str)
                    and side_table in tables
                    and pcol in tables[side_table].dtype_map
                ):
                    dtype = tables[side_table].dtype_map[pcol]
                else:
                    errors.append(f"{ppath}.ref: unknown column {ref!r}")
                if dtype == "int" and not _is_int(value):
                    errors.append(f"{ppath}.value: expected int for {ref}")
                elif dtype == "float" and not _is_number(value):
                    errors.append(f"{ppath}.value: expected number for {ref}")
                if dtype is not None and op in OPS:
                    predicates[side].append(
                        Predicate(ptable, pcol, op, value)
                    )

    projections_raw = raw.get("projections")
    projections = []
    if not isinstance(projections_raw, list) or not projections_raw:
        errors.append("query.projections: non-empty reference list required")
    else:
        for pidx, ref in enumerate(projections_raw):
            ppath = f"query.projections[{pidx}]"
            if not isinstance(ref, str) or ref.count(".") != 1:
                errors.append(f"{ppath}: expected table.column reference")
                continue
            ptable, pcol = ref.split(".")
            if ptable not in tables or pcol not in tables[ptable].dtype_map:
                errors.append(f"{ppath}: unknown column {ref!r}")
                continue
            if ref in projections:
                errors.append(f"{ppath}: duplicate projection {ref!r}")
                continue
            projections.append(ref)

    if errors:
        return None
    left_dtype = tables[left].dtype_map[join_key]
    right_dtype = tables[right].dtype_map[join_key]
    if left_dtype not in DTYPES or right_dtype not in DTYPES:
        errors.append("query.join_key: join key types are not comparable")
        return None
    return Query(
        query_id=qid,
        left_table=left,
        right_table=right,
        join_key=join_key,
        predicates=predicates,
        projections=projections,
        left_key_dtype=left_dtype,
        right_key_dtype=right_dtype,
    )


def load_spec(raw):
    """Validate the whole document and either return :class:`Spec` or raise
    a single :class:`SchemaError` listing every offending field path."""
    errors = []
    if not isinstance(raw, dict):
        raise SchemaError("input document must be a JSON object", ["$"])

    tables_raw = raw.get("tables")
    if not isinstance(tables_raw, dict) or len(tables_raw) != 2:
        raise SchemaError(
            "exactly two tables must be declared", ["tables"]
        )
    names = list(tables_raw.keys())
    for name in names:
        if not isinstance(name, str) or not IDENT_RE.match(name or ""):
            errors.append(f"tables.{name}: illegal table name")
    tables = {}
    for name in names:
        table = _validate_table(name, tables_raw[name], errors)
        if table is not None:
            tables[name] = table

    if len(tables) == 2:
        query = _validate_query(raw.get("query"), tables, errors)
    else:
        query = None

    if errors:
        raise SchemaError(
            f"schema validation failed with {len(errors)} error(s)",
            sorted(set(errors)),
        )

    spec = Spec(tables=tables, query=query)
    spec.source_digest = digest_hex(
        {
            "tables": {name: table.row_digests_input()
                       for name, table in tables.items()},
            "query": query_fingerprint_input(query),
        }
    )
    return spec


def load_spec_file(path):
    from .jsonio import read_json

    return load_spec(read_json(path))


def query_fingerprint_input(query):
    return {
        "query_id": query.query_id,
        "left_table": query.left_table,
        "right_table": query.right_table,
        "join_key": query.join_key,
        "predicates": [
            {"ref": pred.ref, "op": pred.op, "value": pred.value}
            for _, pred in query.all_predicates()
        ],
        "projections": list(query.projections),
    }


def query_fingerprint(query):
    return digest_hex(query_fingerprint_input(query))

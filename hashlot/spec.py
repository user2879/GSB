"""列式表、查询定义与槽位拓扑的加载及强校验。

校验失败统一抛出 :class:`SpecError`（错误码 E_SPEC），
``location`` 为首个违规的稳定 JSON 路径（如 ``tables[0].bitmap``）。
"""

import json
from dataclasses import dataclass, field

from .errors import SpecError
from .fixedpoint import validate_scale, INT64_MIN, INT64_MAX
from .hashing import validate_partition_count

SUPPORTED_TYPES = ("int64", "float64")
SUPPORTED_FUNCS = ("count", "sum", "avg")
SUPPORTED_OPS = ("eq", "ne", "lt", "le", "gt", "ge")


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    bitmap: tuple
    values: tuple


@dataclass(frozen=True)
class Table:
    name: str
    row_count: int
    columns: dict


@dataclass(frozen=True)
class Filter:
    table: str
    column: str
    op: str
    value: object


@dataclass(frozen=True)
class Query:
    source_table: str
    filters: tuple
    group_keys: tuple
    aggregates: tuple


@dataclass(frozen=True)
class Borrow:
    lender: int
    borrower: int
    borrowed_key_cap: int
    deadline_ms: int


@dataclass(frozen=True)
class PartitionSpec:
    partition_id: int
    max_connections: int
    key_cap_per_connection: int
    idle_recycle_ms: int


@dataclass(frozen=True)
class Topology:
    partition_count: int
    partitions: tuple
    borrows: tuple
    global_max_connections: int


@dataclass(frozen=True)
class Hotspot:
    enabled: bool
    row_threshold: int
    subpartition_count: int


@dataclass(frozen=True)
class SessionSpec:
    scale: int
    sample_path: str = ""
    batch_size: int = 64
    slot_params_version: int = 1
    topology: Topology = None
    hotspot: Hotspot = None
    tables: tuple = ()
    query: Query = None
    release_on_key_cap: bool = False
    raw: dict = field(default_factory=dict, compare=False)


def _fail(message, location):
    raise SpecError(message, location)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _require(condition, message, location):
    if not condition:
        _fail(message, location)


def _load_column(table_name, col_index, raw):
    loc = f"tables[{table_name}].columns[{col_index}]"
    _require(isinstance(raw, dict), "列定义必须为对象", loc)
    name = raw.get("name")
    dtype = raw.get("dtype")
    bitmap = raw.get("null_bitmap")
    values = raw.get("values")
    _require(isinstance(name, str) and name, "列名必须为非空字符串", f"{loc}.name")
    _require(dtype in SUPPORTED_TYPES, f"类型必须属于 {SUPPORTED_TYPES}", f"{loc}.dtype")
    _require(isinstance(bitmap, list), "空值位图必须为数组", f"{loc}.null_bitmap")
    _require(isinstance(values, list), "样本值必须为数组", f"{loc}.values")
    _require(
        all(b in (0, 1) for b in bitmap),
        "空值位图只能包含 0/1",
        f"{loc}.null_bitmap",
    )
    for i, v in enumerate(values):
        if v is None:
            continue
        if dtype == "int64":
            _require(_is_int(v), "int64 列样本值必须为整数", f"{loc}.values[{i}]")
            _require(INT64_MIN <= v <= INT64_MAX, "整数超出 int64 范围", f"{loc}.values[{i}]")
        else:
            _require(
                isinstance(v, (int, float)) and not isinstance(v, bool),
                "float64 列样本值必须为数值",
                f"{loc}.values[{i}]",
            )
    return Column(name=name, dtype=dtype, bitmap=tuple(bitmap), values=tuple(values))


def _load_table(index, raw):
    loc = f"tables[{index}]"
    _require(isinstance(raw, dict), "表定义必须为对象", loc)
    name = raw.get("name")
    _require(isinstance(name, str) and name, "表名必须为非空字符串", f"{loc}.name")
    row_count = raw.get("row_count")
    _require(_is_int(row_count) and row_count >= 0, "row_count 必须为非负整数", f"{loc}.row_count")
    cols_raw = raw.get("columns")
    _require(isinstance(cols_raw, list) and cols_raw, "columns 必须为非空数组", f"{loc}.columns")
    seen = set()
    columns = {}
    for ci, col_raw in enumerate(cols_raw):
        col = _load_column(name, ci, col_raw)
        _require(col.name not in seen, f"列名重复: {col.name}", f"tables[{name}].columns[{ci}].name")
        seen.add(col.name)
        _require(len(col.bitmap) == row_count, "空值位图长度与行数不一致", f"tables[{name}].columns[{col.name}].null_bitmap")
        _require(len(col.values) == row_count, "样本值长度与行数不一致", f"tables[{name}].columns[{col.name}].values")
        columns[col.name] = col
    return Table(name=name, row_count=row_count, columns=columns)


def _load_query(raw, tables_by_name):
    loc = "query"
    _require(isinstance(raw, dict), "query 必须为对象", loc)
    source = raw.get("source_table")
    _require(isinstance(source, str), "source_table 必须为字符串", f"{loc}.source_table")
    _require(source in tables_by_name, f"源表不存在: {source}", f"{loc}.source_table")
    filters_raw = raw.get("filters", [])
    _require(isinstance(filters_raw, list), "filters 必须为数组", f"{loc}.filters")
    filters = []
    for fi, fraw in enumerate(filters_raw):
        floc = f"{loc}.filters[{fi}]"
        _require(isinstance(fraw, dict), "过滤条件必须为对象", floc)
        tname, cname, op = fraw.get("table"), fraw.get("column"), fraw.get("op")
        _require(tname in tables_by_name, f"过滤表不存在: {tname}", f"{floc}.table")
        table = tables_by_name[tname]
        _require(isinstance(cname, str) and cname in table.columns, f"过滤列不存在: {cname}", f"{floc}.column")
        _require(op in SUPPORTED_OPS, f"操作符必须属于 {SUPPORTED_OPS}", f"{floc}.op")
        value = fraw.get("value")
        dtype = table.columns[cname].dtype
        if dtype == "int64":
            _require(_is_int(value), "int64 过滤值必须为整数", f"{floc}.value")
        else:
            _require(isinstance(value, (int, float)) and not isinstance(value, bool),
                     "float64 过滤值必须为数值", f"{floc}.value")
        filters.append(Filter(table=tname, column=cname, op=op, value=value))
    keys = raw.get("group_keys")
    _require(isinstance(keys, list) and keys, "group_keys 必须为非空数组", f"{loc}.group_keys")
    src = tables_by_name[source]
    for ki, key in enumerate(keys):
        _require(isinstance(key, str), "分组键必须为列名字符串", f"{loc}.group_keys[{ki}]")
        _require(key in src.columns, f"分组键列不存在: {key}", f"{loc}.group_keys[{ki}]")
        _require(src.columns[key].dtype in SUPPORTED_TYPES, "分组键类型不受支持", f"{loc}.group_keys[{ki}]")
    aggs_raw = raw.get("aggregates")
    _require(isinstance(aggs_raw, list) and aggs_raw, "aggregates 必须为非空数组", f"{loc}.aggregates")
    aggs = []
    seen_agg = set()
    for ai, araw in enumerate(aggs_raw):
        aloc = f"{loc}.aggregates[{ai}]"
        _require(isinstance(araw, dict), "聚合定义必须为对象", aloc)
        func, col = araw.get("func"), araw.get("column")
        alias = araw.get("alias")
        _require(func in SUPPORTED_FUNCS, f"聚合函数必须属于 {SUPPORTED_FUNCS}", f"{aloc}.func")
        _require(isinstance(alias, str) and alias, "聚合别名必须为非空字符串", f"{aloc}.alias")
        _require(alias not in seen_agg, f"聚合别名重复: {alias}", f"{aloc}.alias")
        seen_agg.add(alias)
        if func == "count":
            _require(col is None or col == "*", "count 聚合 column 必须为 * 或 null", f"{aloc}.column")
        else:
            _require(isinstance(col, str) and col in src.columns, f"聚合列不存在: {col}", f"{aloc}.column")
            _require(src.columns[col].dtype in SUPPORTED_TYPES,
                     "聚合列类型不受支持", f"{aloc}.column")
        aggs.append((func, col, alias))
    return Query(source_table=source, filters=tuple(filters),
                 group_keys=tuple(keys), aggregates=tuple(aggs))


def _detect_cycle(edges, node_count):
    """返回首个检测到的环（节点 id 列表）；无环返回 None。

    edges: dict 借入方 -> 出借方（每个借入方至多一条出借边）。
    采用 Kahn 拓扑排序，节点按整数顺序入队以保证结果稳定。
    """
    indeg = {i: 0 for i in range(node_count)}
    adj = {i: [] for i in range(node_count)}
    for borrower, lender in edges.items():
        adj[borrower].append(lender)
        indeg[lender] += 1
    queue = sorted(n for n, d in indeg.items() if d == 0)
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
        queue.sort()
    if visited == node_count:
        return None
    remaining = sorted(n for n, d in indeg.items() if d > 0)
    return remaining


def _load_topology(raw, validate_borrow_acyclic=True):
    loc = "topology"
    _require(isinstance(raw, dict), "topology 必须为对象", loc)
    pc = raw.get("partition_count")
    _require(_is_int(pc), "partition_count 必须为整数", f"{loc}.partition_count")
    try:
        validate_partition_count(pc)
    except ValueError as exc:
        _fail(str(exc), f"{loc}.partition_count")
    gmax = raw.get("global_max_connections")
    _require(_is_int(gmax) and gmax > 0, "global_max_connections 必须为正整数", f"{loc}.global_max_connections")
    parts_raw = raw.get("partitions")
    _require(isinstance(parts_raw, list) and len(parts_raw) == pc,
             "partitions 数量必须等于 partition_count", f"{loc}.partitions")
    partitions = []
    for pi, praw in enumerate(parts_raw):
        ploc = f"{loc}.partitions[{pi}]"
        _require(isinstance(praw, dict), "分区配置必须为对象", ploc)
        pid = praw.get("partition_id")
        _require(_is_int(pid) and 0 <= pid < pc, "partition_id 越界", f"{ploc}.partition_id")
        _require(pid == pi, "partition_id 必须与数组下标一致且连续", f"{ploc}.partition_id")
        mc = praw.get("max_connections")
        kc = praw.get("key_cap_per_connection")
        idle = praw.get("idle_recycle_ms")
        _require(_is_int(mc) and mc >= 0, "max_connections 必须为非负整数（0 表示完全依赖借用）", f"{ploc}.max_connections")
        _require(_is_int(kc) and kc > 0, "key_cap_per_connection 必须为正整数", f"{ploc}.key_cap_per_connection")
        _require(_is_int(idle) and idle >= 0, "idle_recycle_ms 必须为非负整数", f"{ploc}.idle_recycle_ms")
        partitions.append(PartitionSpec(pid, mc, kc, idle))
    borrows_raw = raw.get("borrows", [])
    _require(isinstance(borrows_raw, list), "borrows 必须为数组", f"{loc}.borrows")
    edge_map = {}
    borrows = []
    for bi, braw in enumerate(borrows_raw):
        bloc = f"{loc}.borrows[{bi}]"
        _require(isinstance(braw, dict), "借用关系必须为对象", bloc)
        lender, borrower = braw.get("lender"), braw.get("borrower")
        cap, deadline = braw.get("borrowed_key_cap"), braw.get("deadline_ms")
        _require(_is_int(lender) and 0 <= lender < pc, "lender 越界", f"{bloc}.lender")
        _require(_is_int(borrower) and 0 <= borrower < pc, "borrower 越界", f"{bloc}.borrower")
        _require(lender != borrower, "借用关系不得指向自身", f"{bloc}.borrower")
        _require(_is_int(cap) and cap > 0, "borrowed_key_cap 必须为正整数", f"{bloc}.borrowed_key_cap")
        _require(_is_int(deadline) and deadline > 0, "deadline_ms 必须为正整数", f"{bloc}.deadline_ms")
        _require(borrower not in edge_map, "每个借入方只能声明一条借用关系", f"{bloc}.borrower")
        edge_map[borrower] = lender
        borrows.append(Borrow(lender, borrower, cap, deadline))
    cycle = _detect_cycle(edge_map, pc)
    if cycle is not None and validate_borrow_acyclic:
        _fail(f"借用关系存在环，涉及分区 {cycle}", f"{loc}.borrows")
    total_owned = sum(p.max_connections for p in partitions)
    _require(gmax <= total_owned, "global_max_connections 不得超过各分区独占连接总和", f"{loc}.global_max_connections")
    return Topology(pc, tuple(partitions), tuple(borrows), gmax)


def _load_hotspot(raw):
    loc = "hotspot"
    _require(isinstance(raw, dict), "hotspot 必须为对象", loc)
    enabled = raw.get("enabled", False)
    _require(isinstance(enabled, bool), "enabled 必须为布尔值", f"{loc}.enabled")
    threshold = raw.get("row_threshold")
    sub = raw.get("subpartition_count", 2)
    _require(_is_int(threshold) and threshold > 0, "row_threshold 必须为正整数", f"{loc}.row_threshold")
    try:
        validate_partition_count(sub)
    except ValueError as exc:
        _fail(str(exc), f"{loc}.subpartition_count")
    _require(sub >= 2, "子分区数至少为 2", f"{loc}.subpartition_count")
    return Hotspot(enabled, threshold, sub)


def load_spec(path, validate_borrow_acyclic=True):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        _fail(f"样本文件不存在: {path}", path)
    except json.JSONDecodeError as exc:
        _fail(f"JSON 解析失败: {exc.msg}", f"{path}:{exc.lineno}:{exc.colno}")
    _require(isinstance(raw, dict), "顶层必须为 JSON 对象", "$")
    scale = raw.get("scale")
    try:
        validate_scale(scale)
    except ValueError as exc:
        _fail(str(exc), "scale")
    tables_raw = raw.get("tables")
    _require(isinstance(tables_raw, list) and tables_raw, "tables 必须为非空数组", "tables")
    names = set()
    tables = []
    for ti, traw in enumerate(tables_raw):
        table = _load_table(ti, traw)
        _require(table.name not in names, f"表名重复: {table.name}", f"tables[{ti}].name")
        names.add(table.name)
        tables.append(table)
    tables_by_name = {t.name: t for t in tables}
    query = _load_query(raw.get("query"), tables_by_name)
    topology = _load_topology(raw["topology"], validate_borrow_acyclic)
    hotspot = _load_hotspot(raw.get("hotspot", {"enabled": False, "row_threshold": 1 << 30}))
    batch = raw.get("batch_size", 64)
    _require(_is_int(batch) and batch > 0, "batch_size 必须为正整数", "batch_size")
    version = raw.get("slot_params_version", 1)
    _require(_is_int(version) and version > 0, "slot_params_version 必须为正整数", "slot_params_version")
    release_on_key_cap = bool(raw.get("release_on_key_cap", False))
    _require(isinstance(raw.get("release_on_key_cap", False), bool),
             "release_on_key_cap 必须为布尔值", "release_on_key_cap")
    return SessionSpec(
        scale=scale,
        sample_path=path,
        batch_size=batch,
        slot_params_version=version,
        topology=topology,
        hotspot=hotspot,
        tables=tuple(tables),
        query=query,
        release_on_key_cap=release_on_key_cap,
        raw=raw,
    )

"""本地 JSON 声明与变更集的加载及强校验（E_SPEC）。

校验规则（任一不合法整体拒绝，并稳定报告首个违规键与字段）：
- 键名合法且唯一；节点标识合法且唯一。
- 取值 / 派生系数可量化为定点整数（有限十进制数）。
- 声明内配置版本号按声明顺序严格单调递增；变更集版本号严格单调递增
  且不小于声明中的最大版本号。
- 过期毫秒为正整数。
- 派生引用目标必须存在；派生依赖图无环。
- 节点 host 必须为回环字面量 "127.0.0.1"，端口为合法整数。
- 订阅引用的配置键与节点标识必须存在。
"""

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal

from .errors import SpecError
from .quantize import check_scale_exp, quantize

KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")
LOOPBACK_LITERAL = "127.0.0.1"
MIN_PORT = 1024
MAX_PORT = 65535


@dataclass(frozen=True)
class Item:
    key: str
    value: Decimal
    version: int
    expire_ms: int
    derived_from: str  # "" 表示无派生引用
    derived_coeff: Decimal  # derived_from 为空时无意义（须为 None 声明）


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    subscribes: tuple
    seed: int
    sample_size: int


@dataclass(frozen=True)
class SessionParams:
    scale_exp: int
    groups: int
    global_concurrency: int
    node_concurrency: int
    dispatch_timeout_ms: int
    backoff_ms: tuple


@dataclass(frozen=True)
class Declaration:
    session: SessionParams
    items: tuple  # 声明顺序
    nodes: tuple

    def item_map(self):
        return {it.key: it for it in self.items}

    def node_map(self):
        return {nd.node_id: nd for nd in self.nodes}


@dataclass(frozen=True)
class Change:
    key: str
    value: object  # None 表示未变更
    expire_ms: object
    derived_coeff: object
    subscribers_add: tuple
    subscribers_remove: tuple


@dataclass(frozen=True)
class Changeset:
    version: int
    changes: tuple

    def changed_keys(self):
        return sorted(c.key for c in self.changes)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _decimal_or_spec(value, key, field_name):
    if isinstance(value, bool) or value is None:
        raise SpecError("value must be a finite number", key=key, field=field_name)
    try:
        dec = Decimal(str(value))
    except Exception as exc:  # noqa: BLE001
        raise SpecError("value must be a finite number", key=key, field=field_name) from exc
    if not dec.is_finite():
        raise SpecError("value must be finite", key=key, field=field_name)
    return dec


def _check_quantizable(dec, scale_exp, key, field_name):
    try:
        quantize(dec, scale_exp)
    except SpecError as exc:
        raise SpecError(
            "value is not quantizable to fixed-point", key=key, field=field_name
        ) from exc


def _require(cond, message, key=None, field_name=None):
    if not cond:
        raise SpecError(message, key=key, field=field_name)


def _parse_session(raw):
    _require(isinstance(raw, dict), "session must be an object", field_name="session")
    scale_exp = raw.get("scale_exp")
    check_scale_exp(scale_exp)  # raises E_SPEC with field=scale_exp
    groups = raw.get("groups", 8)
    _require(_is_int(groups) and groups > 0, "groups must be a positive integer",
             field_name="groups")
    gconc = raw.get("global_concurrency", 4)
    _require(_is_int(gconc) and gconc > 0,
             "global_concurrency must be a positive integer",
             field_name="global_concurrency")
    nconc = raw.get("node_concurrency", 1)
    _require(_is_int(nconc) and nconc > 0,
             "node_concurrency must be a positive integer",
             field_name="node_concurrency")
    timeout = raw.get("dispatch_timeout_ms", 800)
    _require(_is_int(timeout) and timeout > 0,
             "dispatch_timeout_ms must be a positive integer",
             field_name="dispatch_timeout_ms")
    backoff = raw.get("backoff_ms", [50, 100, 200])
    _require(isinstance(backoff, list) and all(_is_int(b) and b >= 0 for b in backoff),
             "backoff_ms must be a list of non-negative integers",
             field_name="backoff_ms")
    return SessionParams(
        scale_exp=scale_exp,
        groups=groups,
        global_concurrency=gconc,
        node_concurrency=nconc,
        dispatch_timeout_ms=timeout,
        backoff_ms=tuple(backoff),
    )


def _parse_items(raw_items, scale_exp):
    _require(isinstance(raw_items, list) and raw_items,
             "items must be a non-empty array", field_name="items")
    seen = set()
    items = []
    prev_version = 0
    for raw in raw_items:
        _require(isinstance(raw, dict), "item must be an object", field_name="items")
        key = raw.get("key")
        _require(isinstance(key, str) and KEY_RE.match(key),
                 "item key is missing or illegal", key=key, field_name="key")
        _require(key not in seen, "duplicate item key", key=key, field_name="key")
        seen.add(key)
        _require("value" in raw, "item value is required", key=key, field_name="value")
        value = _decimal_or_spec(raw.get("value"), key, "value")
        _check_quantizable(value, scale_exp, key, "value")
        version = raw.get("version")
        _require(_is_int(version) and version >= 1,
                 "item version must be an integer >= 1", key=key, field_name="version")
        _require(version > prev_version,
                 "item versions must be strictly increasing in declaration order",
                 key=key, field_name="version")
        prev_version = version
        expire = raw.get("expire_ms")
        _require(_is_int(expire) and expire > 0,
                 "expire_ms must be a positive integer", key=key, field_name="expire_ms")
        derived_from = raw.get("derived_from")
        _require(derived_from is None or (isinstance(derived_from, str) and derived_from),
                 "derived_from must be null or a key string",
                 key=key, field_name="derived_from")
        coeff = raw.get("derived_coeff")
        if derived_from is None:
            _require(coeff is None,
                     "derived_coeff requires derived_from", key=key,
                     field_name="derived_coeff")
            coeff_dec = None
        else:
            _require(coeff is not None,
                     "derived item requires derived_coeff", key=key,
                     field_name="derived_coeff")
            coeff_dec = _decimal_or_spec(coeff, key, "derived_coeff")
            _check_quantizable(coeff_dec, scale_exp, key, "derived_coeff")
        items.append(Item(
            key=key, value=value, version=version, expire_ms=expire,
            derived_from=derived_from or "",
            derived_coeff=coeff_dec if coeff_dec is not None else Decimal(0),
        ))
    # 派生引用目标存在
    for it in items:
        if it.derived_from and it.derived_from not in seen:
            raise SpecError("derived_from target does not exist",
                            key=it.key, field="derived_from")
    # 派生依赖图无环（确定性：按键名排序的着色 DFS）
    color = {}
    parent = {it.key: it.derived_from for it in items if it.derived_from}

    def visit(node):
        color[node] = 1
        nxt = parent.get(node, "")
        if nxt:
            state = color.get(nxt, 0)
            if state == 1:
                # 报告环上字典序最小的键，保证首个违规键稳定
                cycle = [nxt]
                cur = parent.get(nxt, "")
                while cur and cur != nxt:
                    cycle.append(cur)
                    cur = parent.get(cur, "")
                raise SpecError("derived dependency graph has a cycle",
                                key=min(cycle), field="derived_from")
            if state == 0:
                visit(nxt)
        color[node] = 2

    for key in sorted(parent):
        if color.get(key, 0) == 0:
            visit(key)
    return tuple(items)


def _parse_nodes(raw_nodes, known_keys):
    _require(isinstance(raw_nodes, list) and raw_nodes,
             "nodes must be a non-empty array", field_name="nodes")
    seen = set()
    nodes = []
    for raw in raw_nodes:
        _require(isinstance(raw, dict), "node must be an object", field_name="nodes")
        nid = raw.get("id")
        _require(isinstance(nid, str) and KEY_RE.match(nid),
                 "node id is missing or illegal", key=nid, field_name="id")
        _require(nid not in seen, "duplicate node id", key=nid, field_name="id")
        seen.add(nid)
        host = raw.get("host")
        _require(host == LOOPBACK_LITERAL,
                 "node host must be the loopback literal 127.0.0.1",
                 key=nid, field_name="host")
        port = raw.get("port")
        _require(_is_int(port) and MIN_PORT <= port <= MAX_PORT,
                 "node port must be an integer in %d..%d" % (MIN_PORT, MAX_PORT),
                 key=nid, field_name="port")
        subs = raw.get("subscribes")
        _require(isinstance(subs, list) and subs,
                 "subscribes must be a non-empty array", key=nid,
                 field_name="subscribes")
        _require(len(set(subs)) == len(subs),
                 "subscribes contains duplicates", key=nid, field_name="subscribes")
        for sub in subs:
            _require(sub in known_keys,
                     "subscribed key does not exist", key=nid,
                     field_name="subscribes")
        seed = raw.get("seed")
        _require(_is_int(seed) and seed >= 0,
                 "seed must be a non-negative integer", key=nid, field_name="seed")
        size = raw.get("sample_size")
        _require(_is_int(size) and size > 0,
                 "sample_size must be a positive integer", key=nid,
                 field_name="sample_size")
        nodes.append(NodeSpec(
            node_id=nid, host=host, port=port,
            subscribes=tuple(sorted(subs)), seed=seed, sample_size=size,
        ))
    return tuple(nodes)


def validate_declaration(raw):
    _require(isinstance(raw, dict), "declaration must be an object")
    session = _parse_session(raw.get("session"))
    items = _parse_items(raw.get("items"), session.scale_exp)
    nodes = _parse_nodes(raw.get("nodes"), {it.key for it in items})
    return Declaration(session=session, items=items, nodes=nodes)


def load_declaration(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return validate_declaration(raw)


def validate_changeset(raw, decl, prev_version):
    """校验变更集；prev_version 为上一变更集版本（无则为 None）。

    规则：变更集版本号不小于声明最大版本号，且彼此严格单调递增。
    """
    _require(isinstance(raw, dict), "changeset must be an object")
    version = raw.get("version")
    _require(_is_int(version), "changeset version must be an integer",
             field_name="version")
    declared_max = max(it.version for it in decl.items)
    _require(version >= declared_max,
             "changeset version must be >= declared max version",
             field_name="version")
    if prev_version is not None:
        _require(version > prev_version,
                 "changeset versions must be strictly increasing",
                 field_name="version")
    raw_changes = raw.get("changes")
    _require(isinstance(raw_changes, list) and raw_changes,
             "changes must be a non-empty array", field_name="changes")
    known_keys = {it.key for it in decl.items}
    known_nodes = {nd.node_id for nd in decl.nodes}
    changes = []
    seen_keys = set()
    for rc in raw_changes:
        _require(isinstance(rc, dict), "change must be an object",
                 field_name="changes")
        key = rc.get("key")
        _require(key in known_keys, "change targets an unknown key",
                 key=key, field_name="key")
        _require(key not in seen_keys, "duplicate change for key",
                 key=key, field_name="key")
        seen_keys.add(key)
        value = rc.get("value")
        if value is not None:
            value = _decimal_or_spec(value, key, "value")
            _check_quantizable(value, decl.session.scale_exp, key, "value")
        expire = rc.get("expire_ms")
        if expire is not None:
            _require(_is_int(expire) and expire > 0,
                     "expire_ms must be a positive integer",
                     key=key, field_name="expire_ms")
        coeff = rc.get("derived_coeff")
        if coeff is not None:
            item = decl.item_map()[key]
            _require(item.derived_from,
                     "derived_coeff change on a non-derived item",
                     key=key, field_name="derived_coeff")
            coeff = _decimal_or_spec(coeff, key, "derived_coeff")
            _check_quantizable(coeff, decl.session.scale_exp, key, "derived_coeff")
        add = rc.get("subscribers_add", [])
        remove = rc.get("subscribers_remove", [])
        _require(isinstance(add, list) and isinstance(remove, list),
                 "subscribers_add/remove must be arrays",
                 key=key, field_name="subscribers_add")
        for nid in list(add) + list(remove):
            _require(nid in known_nodes, "unknown subscriber node",
                     key=key, field_name="subscribers_add")
        _require(not set(add) & set(remove),
                 "subscribers_add and subscribers_remove overlap",
                 key=key, field_name="subscribers_add")
        _require(value is not None or expire is not None or coeff is not None
                 or add or remove,
                 "change must modify at least one field", key=key, field_name="changes")
        changes.append(Change(
            key=key, value=value, expire_ms=expire, derived_coeff=coeff,
            subscribers_add=tuple(sorted(add)),
            subscribers_remove=tuple(sorted(remove)),
        ))
    return Changeset(version=version, changes=tuple(changes))


def load_changeset(path, decl, prev_version):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return validate_changeset(raw, decl, prev_version)


# ---------------------------------------------------------------------------
# 世界状态：声明 + 已应用变更集后的有效配置视图
# ---------------------------------------------------------------------------

@dataclass
class ItemState:
    key: str
    value: Decimal
    version: int
    expire_ms: int
    derived_from: str
    derived_coeff: Decimal


@dataclass
class WorldState:
    items: dict  # key -> ItemState
    subs: dict   # key -> sorted list of node_id
    applied_versions: list = field(default_factory=list)

    def canonical(self):
        doc = {
            "items": [
                {
                    "key": k,
                    "value": str(self.items[k].value),
                    "version": self.items[k].version,
                    "expire_ms": self.items[k].expire_ms,
                    "derived_from": self.items[k].derived_from,
                    "derived_coeff": str(self.items[k].derived_coeff),
                }
                for k in sorted(self.items)
            ],
            "subs": {k: list(self.subs[k]) for k in sorted(self.subs)},
            "applied_versions": list(self.applied_versions),
        }
        return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def initial_state(decl):
    items = {
        it.key: ItemState(it.key, it.value, it.version, it.expire_ms,
                          it.derived_from, it.derived_coeff)
        for it in decl.items
    }
    subs = {it.key: [] for it in decl.items}
    for nd in decl.nodes:
        for key in nd.subscribes:
            subs[key].append(nd.node_id)
    subs = {k: sorted(v) for k, v in subs.items()}
    return WorldState(items=items, subs=subs, applied_versions=[])


def apply_changeset(state, changeset):
    """返回应用变更集后的新 WorldState（不修改入参）。"""
    items = {k: ItemState(v.key, v.value, v.version, v.expire_ms,
                          v.derived_from, v.derived_coeff)
             for k, v in state.items.items()}
    subs = {k: list(v) for k, v in state.subs.items()}
    for ch in changeset.changes:
        it = items[ch.key]
        content_changed = False
        if ch.value is not None:
            it.value = ch.value
            content_changed = True
        if ch.expire_ms is not None:
            it.expire_ms = ch.expire_ms
            content_changed = True
        if ch.derived_coeff is not None:
            it.derived_coeff = ch.derived_coeff
            content_changed = True
        # 仅内容变化提升配置版本号；订阅集合变化不改变项内容，
        # 既有订阅者的下发产物保持逐字节不变。
        if content_changed:
            it.version = changeset.version
        cur = set(subs[ch.key])
        cur |= set(ch.subscribers_add)
        cur -= set(ch.subscribers_remove)
        subs[ch.key] = sorted(cur)
    return WorldState(items=items, subs=subs,
                      applied_versions=state.applied_versions + [changeset.version])

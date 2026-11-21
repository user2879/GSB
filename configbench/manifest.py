"""配置版本清单、节点注册表与样本声明的加载与强校验。

校验顺序固定、不依赖字典/哈希遍历顺序；任一不合法立即抛出
``ManifestError(E_MANIFEST)``，错误体带首个违规字段 field 与判定依据 basis，
整个清单被整体拒绝（不产生部分加载状态）。
"""

import json
import re
from dataclasses import dataclass, field

from .errors import ManifestError

KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
SUPPORTED_VALUE_TYPES = ("int", "fixed")
SUPPORTED_STRATEGIES = ("retry_then_fail", "fail_fast")


@dataclass(frozen=True)
class DispatchPolicy:
    timeout_ms: int
    retries: int
    strategy: str


@dataclass(frozen=True)
class DeriveEdge:
    key: str
    coef_raw: int
    coef_scale: int


@dataclass(frozen=True)
class ItemSpec:
    key: str
    value_type: str
    value: int            # int 类型即本身；fixed 类型为 raw
    scale: int            # int 类型为 0；fixed 类型为定点小数位
    targets: tuple
    dispatch: DispatchPolicy
    derives_from: tuple


@dataclass(frozen=True)
class ManifestVersion:
    version: int
    items: dict


@dataclass(frozen=True)
class RegistryNode:
    node_id: str
    host: str
    port: int
    group_key: str
    scale: int
    fail_inject: str  # None | "port" | "handshake" | "field"


@dataclass(frozen=True)
class Registry:
    nodes: dict


@dataclass(frozen=True)
class SampleDecl:
    seed: int
    n_samples: int
    n_groups: int
    value_range: int
    declared_scale: int


@dataclass(frozen=True)
class Manifest:
    latency_budget_ms: int
    default_dispatch: DispatchPolicy
    versions: tuple

    def version(self, number):
        for v in self.versions:
            if v.version == number:
                return v
        raise ManifestError(
            f"版本 {number} 不存在",
            field=f"manifest.version[{number}]",
            basis="版本号必须存在于清单中",
        )

    @property
    def latest(self):
        return self.versions[-1]


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ManifestError(
            f"文件不存在: {path}", field="file", basis="清单文件必须可读"
        )
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"JSON 解析失败: {exc}",
            field="file",
            basis=f"必须是合法 JSON，行 {exc.lineno} 列 {exc.colno}",
        )


def _require(cond, field, basis, message):
    if not cond:
        raise ManifestError(message, field=field, basis=basis)


def _parse_dispatch(data, default, field_path):
    if data is None:
        return default
    _require(isinstance(data, dict), field_path, "必须是对象", "分发策略格式非法")
    timeout = data.get("timeout_ms", default.timeout_ms)
    retries = data.get("retries", default.retries)
    strategy = data.get("strategy", default.strategy)
    _require(
        isinstance(timeout, int) and timeout > 0,
        f"{field_path}.timeout_ms",
        "必须是正整数毫秒",
        "timeout_ms 非法",
    )
    _require(
        isinstance(retries, int) and retries >= 0,
        f"{field_path}.retries",
        "必须是非负整数（额外重试次数）",
        "retries 非法",
    )
    _require(
        strategy in SUPPORTED_STRATEGIES,
        f"{field_path}.strategy",
        f"取值必须属于 {SUPPORTED_STRATEGIES}",
        "strategy 非法",
    )
    return DispatchPolicy(timeout, retries, strategy)


def load_registry(path):
    data = _load_json(path)
    _require(isinstance(data, dict) and isinstance(data.get("nodes"), list),
             "registry.nodes", "必须是节点对象数组", "注册表结构非法")
    nodes = {}
    seen_ports = set()
    for idx, nd in enumerate(data["nodes"]):
        p = f"registry.nodes[{idx}]"
        _require(isinstance(nd, dict), p, "必须是对象", "节点条目非法")
        node_id = nd.get("node_id")
        _require(isinstance(node_id, str) and KEY_PATTERN.match(node_id or ""),
                 f"{p}.node_id", "须匹配 ^[a-z][a-z0-9_-]{0,63}$", "节点标识非法")
        _require(node_id not in nodes, f"{p}.node_id",
                 "节点标识全局唯一", "节点标识重复")
        host = nd.get("host", "127.0.0.1")
        _require(host == "127.0.0.1", f"{p}.host",
                 "实验台仅允许回环地址 127.0.0.1", "host 非法")
        port = nd.get("port")
        _require(isinstance(port, int) and 1024 <= port <= 65535,
                 f"{p}.port", "必须是 [1024,65535] 内整数", "端口非法")
        _require(port not in seen_ports, f"{p}.port",
                 "端口全局不重复", "端口重复")
        seen_ports.add(port)
        group_key = nd.get("group_key")
        _require(isinstance(group_key, str) and KEY_PATTERN.match(group_key or ""),
                 f"{p}.group_key", "须为合法键名", "分组键非法")
        scale = nd.get("scale", 0)
        _require(isinstance(scale, int) and 0 <= scale <= 12,
                 f"{p}.scale", "必须是 [0,12] 内整数", "缩放因子非法")
        fail = nd.get("fail_inject")
        _require(fail in (None, "port", "handshake", "field", "flaky"),
                 f"{p}.fail_inject",
                 "取值为 null/port/handshake/field/flaky", "故障注入配置非法")
        nodes[node_id] = RegistryNode(node_id, host, port, group_key, scale, fail)
    _require(len(nodes) > 0, "registry.nodes", "至少声明一个节点", "注册表为空")
    return Registry(nodes)


def _parse_item(key, raw, default, registry, known_keys, field_prefix):
    p = f"{field_prefix}.items[{key}]"
    _require(KEY_PATTERN.match(key or ""), f"{p}.<key>",
             "键名须匹配 ^[a-z][a-z0-9_-]{0,63}$", "配置项键非法")
    _require(isinstance(raw, dict), p, "必须是对象", "配置项格式非法")
    vtype = raw.get("value_type")
    _require(vtype in SUPPORTED_VALUE_TYPES, f"{p}.value_type",
             f"受支持数值类型为 {SUPPORTED_VALUE_TYPES}", "数值类型不受支持")
    if vtype == "int":
        value = raw.get("value")
        _require(isinstance(value, int) and not isinstance(value, bool),
                 f"{p}.value", "int 类型必须是 JSON 整数", "数值类型不匹配")
        scale = 0
    else:
        fixed = raw.get("value")
        _require(isinstance(fixed, dict)
                 and isinstance(fixed.get("raw"), int)
                 and isinstance(fixed.get("scale"), int)
                 and fixed["scale"] >= 0,
                 f"{p}.value", "fixed 类型必须是 {raw:int, scale:int>=0}",
                 "定点数值格式非法")
        value, scale = fixed["raw"], fixed["scale"]
    targets = raw.get("targets")
    _require(isinstance(targets, list) and len(targets) > 0,
             f"{p}.targets", "必须是非空数组", "目标节点非法")
    for t in targets:
        _require(t in registry.nodes, f"{p}.targets[{t}]",
                 "目标节点必须存在于注册表", "目标节点未注册")
    edges_raw = raw.get("derives_from", [])
    edges = []
    _require(isinstance(edges_raw, list), f"{p}.derives_from",
             "必须是数组", "派生关系格式非法")
    for e in edges_raw:
        ep = f"{p}.derives_from"
        _require(isinstance(e, dict), ep, "派生边必须是对象", "派生边非法")
        ref = e.get("key")
        _require(ref in known_keys, f"{ep}.key",
                 "派生引用必须指向同版本已声明的配置项", "派生引用不存在")
        coef = e.get("coef")
        _require(isinstance(coef, dict)
                 and isinstance(coef.get("raw"), int)
                 and isinstance(coef.get("scale"), int)
                 and coef["scale"] >= 0,
                 f"{ep}.coef", "系数必须是 {raw:int, scale:int>=0}",
                 "派生系数非法")
        edges.append(DeriveEdge(ref, coef["raw"], coef["scale"]))
    dispatch = _parse_dispatch(raw.get("dispatch"), default, f"{p}.dispatch")
    return ItemSpec(key, vtype, value, scale, tuple(sorted(targets)), dispatch,
                    tuple(edges))


def _assert_acyclic(items, field_prefix):
    # 稳定拓扑 Kahn：候选集合始终按键名排序，遍历顺序确定。
    indegree = {k: 0 for k in items}
    dependents = {k: [] for k in items}
    for key, spec in items.items():
        for edge in spec.derives_from:
            indegree[key] += 1
            dependents[edge.key].append(key)
    ready = sorted(k for k, d in indegree.items() if d == 0)
    ordered = []
    while ready:
        cur = ready.pop(0)
        ordered.append(cur)
        for nxt in sorted(dependents[cur]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort()
    if len(ordered) != len(items):
        cycle_keys = sorted(k for k, d in indegree.items() if d > 0)
        raise ManifestError(
            "派生依赖存在环",
            field=f"{field_prefix}.derives_from",
            basis=f"仍有入度的节点: {cycle_keys}",
        )
    return ordered


def load_manifest(path, registry):
    data = _load_json(path)
    _require(isinstance(data, dict), "manifest", "必须是对象", "清单结构非法")
    budget = data.get("latency_budget_ms")
    _require(isinstance(budget, int) and budget > 0,
             "manifest.latency_budget_ms", "必须是正整数毫秒", "延迟预算非法")
    default = _parse_dispatch(data.get("default_dispatch"),
                              DispatchPolicy(2000, 1, "retry_then_fail"),
                              "manifest.default_dispatch")
    versions_raw = data.get("versions")
    _require(isinstance(versions_raw, list) and len(versions_raw) > 0,
             "manifest.versions", "必须是非空数组", "版本清单非法")
    versions = []
    prev_number = None
    seen_numbers = set()
    for idx, vraw in enumerate(versions_raw):
        vp = f"manifest.versions[{idx}]"
        _require(isinstance(vraw, dict), vp, "必须是对象", "版本条目非法")
        number = vraw.get("version")
        _require(isinstance(number, int) and number > 0,
                 f"{vp}.version", "必须是正整数", "版本号非法")
        _require(number not in seen_numbers, f"{vp}.version",
                 "版本号全局唯一", "版本号重复")
        if prev_number is not None:
            _require(number > prev_number, f"{vp}.version",
                     f"版本号须严格单调递增，上一版本为 {prev_number}",
                     "版本号未单调递增")
        seen_numbers.add(number)
        prev_number = number
        items_raw = vraw.get("items")
        _require(isinstance(items_raw, dict) and len(items_raw) > 0,
                 f"{vp}.items", "必须是非空对象", "配置项集合非法")
        # 先按键名稳定排序，保证“同版本已声明”的引用判定确定。
        items = {}
        for key in sorted(items_raw):
            spec = _parse_item(key, items_raw[key], default, registry,
                               set(sorted(items_raw)), vp)
            items[key] = spec
        _assert_acyclic(items, vp)
        versions.append(ManifestVersion(number, items))
    return Manifest(budget, default, tuple(versions))


def load_sample_decl(path):
    data = _load_json(path)
    p = "samples"
    _require(isinstance(data, dict), p, "必须是对象", "样本声明非法")
    seed = data.get("seed")
    _require(isinstance(seed, int) and 0 <= seed <= 2**32 - 1,
             f"{p}.seed", "必须是 [0,2^32) 整数", "种子非法")
    n_samples = data.get("n_samples")
    _require(isinstance(n_samples, int) and n_samples > 0,
             f"{p}.n_samples", "必须是正整数", "样本数非法")
    n_groups = data.get("n_groups")
    _require(isinstance(n_groups, int) and n_groups > 0,
             f"{p}.n_groups", "必须是正整数", "分组数非法")
    value_range = data.get("value_range")
    _require(isinstance(value_range, int) and value_range > 0,
             f"{p}.value_range", "必须是正整数", "样本值域非法")
    declared_scale = data.get("declared_scale", 0)
    _require(isinstance(declared_scale, int) and declared_scale >= 0,
             f"{p}.declared_scale", "必须是非负整数", "声明缩放因子非法")
    return SampleDecl(seed, n_samples, n_groups, value_range, declared_scale)


def load_all(registry_path, manifest_path, samples_path):
    registry = load_registry(registry_path)
    manifest = load_manifest(manifest_path, registry)
    samples = load_sample_decl(samples_path)
    return registry, manifest, samples

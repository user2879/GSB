"""配置清单与节点注册表的数据模型、加载与强校验（需求 1）。

清单 JSON 结构::

    {
      "version": 2,
      "strategy": "broadcast" | "canary" | "rolling",
      "items": {
        "item_key": {
          "value": 1.25,                 # 叶子项：受支持数值
          "derived": false,
          "targets": ["node-a", "node-b"],
          "work_repeat": 1
        },
        "derived_key": {
          "derived": true,
          "coeffs": {"base_a": 2.0, "base_b": -1.0},
          "const": 0.0,
          "targets": ["node-a"],
          "work_repeat": 1
        }
      }
    }

任何字段违规都会汇总为违规条目并整体以 ``E_MANIFEST`` 拒绝。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canon import topo_sort
from .errors import ManifestError
from .canon import canonical_digest

KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
SUPPORTED_STRATEGIES = ("broadcast", "canary", "rolling")
_REAL_TYPES = (int, float)


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    fail_rate: float = 0.0


@dataclass(frozen=True)
class ItemSpec:
    key: str
    derived: bool
    targets: tuple[str, ...]
    value: float | None = None
    coeffs: dict[str, float] = field(default_factory=dict)
    const: float = 0.0
    work_repeat: int = 1
    extra_sleep_ms: int = 0


@dataclass(frozen=True)
class Manifest:
    version: int
    strategy: str
    items: dict[str, ItemSpec]
    source_digest: str

    def keys(self) -> list[str]:
        return sorted(self.items)

    def leaf_keys(self) -> list[str]:
        return sorted(key for key, item in self.items.items() if not item.derived)

    def derived_keys(self) -> list[str]:
        return sorted(key for key, item in self.items.items() if item.derived)

    def edges(self) -> dict[str, set[str]]:
        """派生依赖边：源项 -> 派生项。"""
        graph = {key: set() for key in self.items}
        for key, item in self.items.items():
            if item.derived:
                for source in item.coeffs:
                    graph[source].add(key)
        return graph

    def topological_order(self) -> list[str]:
        return list(topo_sort(sorted(self.items), self.edges()))


@dataclass(frozen=True)
class Registry:
    nodes: dict[str, NodeSpec]

    def node_ids(self) -> list[str]:
        return sorted(self.nodes)


def _read_json(path: str | Path) -> Any:
    file_path = Path(path)
    if not file_path.is_file():
        raise ManifestError("JSON 文件不存在", [{"file": str(file_path), "field": "$", "problem": "missing"}])
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(
            "JSON 语法错误",
            [{"file": str(file_path), "field": "$", "problem": f"json decode: {exc.msg}"}],
        ) from exc


def load_registry(path: str | Path) -> Registry:
    raw = _read_json(path)
    violations: list[dict[str, Any]] = []
    if not isinstance(raw, dict) or not isinstance(raw.get("nodes"), list):
        raise ManifestError("节点注册表结构非法", [{"field": "$", "problem": "expect {nodes: [...]}"}])
    nodes: dict[str, NodeSpec] = {}
    seen_ports: dict[int, str] = {}
    for index, entry in enumerate(raw["nodes"]):
        field_path = f"nodes[{index}]"
        if not isinstance(entry, dict):
            violations.append({"field": field_path, "problem": "节点条目必须是对象"})
            continue
        node_id = entry.get("node_id")
        if not isinstance(node_id, str) or not KEY_PATTERN.fullmatch(node_id):
            violations.append({"field": f"{field_path}.node_id", "value": node_id, "problem": "非法节点标识"})
            continue
        if node_id in nodes:
            violations.append({"field": f"{field_path}.node_id", "value": node_id, "problem": "节点重复"})
            continue
        host = entry.get("host", "127.0.0.1")
        if host not in ("127.0.0.1", "localhost", "::1"):
            violations.append({"field": f"{field_path}.host", "value": host, "problem": "仅允许回环地址"})
        port = entry.get("port")
        if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
            violations.append({"field": f"{field_path}.port", "value": port, "problem": "端口须为 1024..65535 的整数"})
        elif port in seen_ports:
            violations.append(
                {
                    "field": f"{field_path}.port",
                    "value": port,
                    "problem": f"端口与节点 {seen_ports[port]} 重复",
                }
            )
        else:
            seen_ports[port] = node_id
        fail_rate = entry.get("fail_rate", 0.0)
        if not isinstance(fail_rate, (int, float)) or isinstance(fail_rate, bool) or not 0.0 <= float(fail_rate) <= 1.0:
            violations.append({"field": f"{field_path}.fail_rate", "value": fail_rate, "problem": "须为 [0,1] 数值"})
            fail_rate = 0.0
        if not violations or all(v["field"] != f"{field_path}.node_id" for v in violations):
            nodes[node_id] = NodeSpec(node_id, host, port if isinstance(port, int) else 0, float(fail_rate))
    if violations:
        raise ManifestError("节点注册表校验失败", violations)
    return Registry(nodes)


def _is_real_number(value: Any) -> bool:
    return isinstance(value, _REAL_TYPES) and not isinstance(value, bool) and float(value) == float(value)


def _violation(violations: list[dict[str, Any]], field: str, value: Any, problem: str) -> None:
    violations.append({"field": field, "value": _safe_value(value), "problem": problem})


def _safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def load_manifest(path: str | Path, registry: Registry, expected_version: int | None = None) -> Manifest:
    raw = _read_json(path)
    violations: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        raise ManifestError("配置清单根节点必须是对象", [{"field": "$", "problem": "type"}])

    version = raw.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        _violation(violations, "version", version, "版本号须为正整数")
    elif expected_version is not None and version <= expected_version:
        _violation(
            violations,
            "version",
            version,
            f"候选版本号须严格大于基线版本 {expected_version}，期望至少 {expected_version + 1}",
        )

    strategy = raw.get("strategy")
    if strategy not in SUPPORTED_STRATEGIES:
        _violation(violations, "strategy", strategy, f"分发策略须属于 {SUPPORTED_STRATEGIES}")

    raw_items = raw.get("items")
    if not isinstance(raw_items, dict) or not raw_items:
        _violation(violations, "items", raw_items, "配置项集合须为非空对象")
        raw_items = {}

    items: dict[str, ItemSpec] = {}
    for key, entry in raw_items.items():
        field_path = f"items.{key}"
        if not KEY_PATTERN.fullmatch(key):
            _violation(violations, field_path, key, "配置项键须匹配 [A-Za-z_][A-Za-z0-9_]{0,63}")
        if not isinstance(entry, dict):
            _violation(violations, field_path, entry, "配置项条目须是对象")
            continue
        derived = bool(entry.get("derived", False))
        targets = _validate_targets(entry.get("targets", []), registry, field_path, violations)
        work_repeat = entry.get("work_repeat", 1)
        if not isinstance(work_repeat, int) or isinstance(work_repeat, bool) or work_repeat < 1 or work_repeat > 1_000_000:
            _violation(violations, f"{field_path}.work_repeat", work_repeat, "须为 [1, 1000000] 的整数")
            work_repeat = 1
        extra_sleep_ms = entry.get("extra_sleep_ms", 0)
        if (
            not isinstance(extra_sleep_ms, int)
            or isinstance(extra_sleep_ms, bool)
            or not 0 <= extra_sleep_ms <= 60_000
        ):
            _violation(violations, f"{field_path}.extra_sleep_ms", extra_sleep_ms, "须为 [0, 60000] 的整数毫秒")
            extra_sleep_ms = 0
        if derived:
            coeffs_raw = entry.get("coeffs")
            coeffs: dict[str, float] = {}
            if not isinstance(coeffs_raw, dict) or not coeffs_raw:
                _violation(violations, f"{field_path}.coeffs", coeffs_raw, "派生项须含非空 coeffs 映射")
            else:
                for source, coefficient in coeffs_raw.items():
                    if not KEY_PATTERN.fullmatch(source):
                        _violation(violations, f"{field_path}.coeffs.{source}", coefficient, "派生引用键非法")
                    elif not _is_real_number(coefficient) or float(coefficient) == 0.0:
                        _violation(violations, f"{field_path}.coeffs.{source}", coefficient, "派生系数须为非零有限数值")
                    else:
                        coeffs[source] = float(coefficient)
            const = entry.get("const", 0.0)
            if not _is_real_number(const):
                _violation(violations, f"{field_path}.const", const, "常量项须为有限数值")
                const = 0.0
            items[key] = ItemSpec(key, True, tuple(targets), None, coeffs, float(const), work_repeat, extra_sleep_ms)
        else:
            value = entry.get("value")
            if not _is_real_number(value):
                _violation(violations, f"{field_path}.value", value, "叶子项数值须为有限 int/float")
                value = 0.0
            items[key] = ItemSpec(key, False, tuple(targets), float(value), {}, 0.0, work_repeat, extra_sleep_ms)

    _validate_references_and_cycles(items, violations)
    if violations:
        raise ManifestError(f"配置清单 v{version} 校验失败", violations)
    assert isinstance(version, int) and isinstance(strategy, str)
    return Manifest(version, strategy, items, canonical_digest(raw))


def _validate_targets(
    raw_targets: Any,
    registry: Registry,
    field_path: str,
    violations: list[dict[str, Any]],
) -> list[str]:
    if not isinstance(raw_targets, list) or not raw_targets:
        _violation(violations, f"{field_path}.targets", raw_targets, "目标节点集合须为非空数组")
        return []
    targets: list[str] = []
    seen: set[str] = set()
    for node_id in raw_targets:
        if not isinstance(node_id, str) or node_id not in registry.nodes:
            _violation(violations, f"{field_path}.targets", node_id, "目标节点未在注册表登记")
            continue
        if node_id in seen:
            _violation(violations, f"{field_path}.targets", node_id, "目标节点在同一配置项内重复")
            continue
        seen.add(node_id)
        targets.append(node_id)
    return targets


def _validate_references_and_cycles(items: dict[str, ItemSpec], violations: list[dict[str, Any]]) -> None:
    for key, item in items.items():
        if not item.derived:
            continue
        for source in item.coeffs:
            if source not in items:
                _violation(violations, f"items.{key}.coeffs.{source}", source, "派生引用的配置项不存在")
            if source == key:
                _violation(violations, f"items.{key}.coeffs.{key}", source, "派生项不得引用自身")
    graph = {key: set() for key in items}
    for key, item in items.items():
        if item.derived:
            for source in item.coeffs:
                if source in items and source != key:
                    graph[source].add(key)
    visiting: set[str] = set()
    visited: set[str] = set()

    def detect(node: str, trail: tuple[str, ...]) -> None:
        if node in visiting:
            cycle = list(trail[trail.index(node):]) + [node]
            _violation(violations, "items", node, f"派生依赖存在环: {' -> '.join(cycle)}")
            return
        if node in visited:
            return
        visiting.add(node)
        for nxt in sorted(graph[node]):
            detect(nxt, trail + (node,))
        visiting.discard(node)
        visited.add(node)

    for node in sorted(graph):
        detect(node, ())


def load_manifest_pair(
    baseline_path: str | Path,
    candidate_path: str | Path,
    registry_path: str | Path,
) -> tuple[Manifest, Manifest, Registry]:
    """加载并联合校验两版清单，保证候选版本号严格大于基线版本号。"""
    registry = load_registry(registry_path)
    baseline = load_manifest(baseline_path, registry)
    candidate = load_manifest(candidate_path, registry, expected_version=baseline.version)
    if candidate.version <= baseline.version:
        raise ManifestError(
            "候选版本号未严格递增",
            [
                {"field": "version", "value": baseline.version, "problem": "baseline"},
                {"field": "version", "value": candidate.version, "problem": "candidate"},
            ],
        )
    return baseline, candidate, registry


def key_set_delta(baseline: Manifest, candidate: Manifest) -> dict[str, list[str]]:
    """枚举两版配置项键集合差异。"""
    base_keys = set(baseline.items)
    cand_keys = set(candidate.items)
    return {
        "added": sorted(cand_keys - base_keys),
        "removed": sorted(base_keys - cand_keys),
        "shared": sorted(base_keys & cand_keys),
    }

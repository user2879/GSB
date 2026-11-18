"""清单与节点注册表的加载、强校验。

非法输入一律抛 :class:`ManifestError`（E_MANIFEST），details 中带
``version`` 与 ``path``（出错字段的 JSON 路径），调用方整体拒绝该清单。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .errors import ManifestError
from .model import (
    ConfigItem,
    DispatchStrategy,
    ManifestVersion,
    NodeSpec,
    SUPPORTED_NUM_TYPES,
)

KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
REQUIRED_STRATEGY_FIELDS = ("batch_size", "timeout_ms", "max_retries")


def _load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"文件不存在: {p}", {"path": str(p)})
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"JSON 解析失败: {exc.msg}",
            {"path": str(p), "line": exc.lineno, "column": exc.colno},
        ) from exc


def load_nodes(path: str | Path) -> Dict[str, NodeSpec]:
    data = _load_json(path)
    if not isinstance(data, dict) or "nodes" not in data:
        raise ManifestError("节点注册表缺少顶层 nodes 数组", {"path": "nodes"})
    nodes: Dict[str, NodeSpec] = {}
    for i, raw in enumerate(data["nodes"]):
        base = f"nodes[{i}]"
        if not isinstance(raw, dict):
            raise ManifestError("节点条目必须是对象", {"path": base})
        for field_name in ("node_id", "port"):
            if field_name not in raw:
                raise ManifestError(
                    f"节点缺少字段 {field_name}",
                    {"path": f"{base}.{field_name}"},
                )
        node_id = raw["node_id"]
        if not isinstance(node_id, str) or not KEY_PATTERN.match(node_id):
            raise ManifestError(
                "node_id 非法（字母/下划线开头，仅含字母数字下划线点）",
                {"path": f"{base}.node_id", "value": node_id},
            )
        port = raw["port"]
        if not isinstance(port, int) or isinstance(port, bool) or not (
            1024 <= port <= 65535
        ):
            raise ManifestError(
                "port 必须是 1024..65535 的整数",
                {"path": f"{base}.port", "value": port},
            )
        host = raw.get("host", "127.0.0.1")
        if not isinstance(host, str):
            raise ManifestError("host 必须是字符串", {"path": f"{base}.host"})
        if node_id in nodes:
            raise ManifestError(
                "node_id 重复", {"path": f"{base}.node_id", "value": node_id}
            )
        nodes[node_id] = NodeSpec(node_id=node_id, host=host, port=port)
    if not nodes:
        raise ManifestError("节点注册表为空", {"path": "nodes"})
    ports = [n.port for n in nodes.values()]
    if len(set(ports)) != len(ports):
        raise ManifestError("节点端口冲突", {"path": "nodes[*].port"})
    return nodes


def _validate_item(raw: Any, index: int, version: int, node_ids: set[str]) -> ConfigItem:
    base = f"versions[@version={version}].items[{index}]"
    if not isinstance(raw, dict):
        raise ManifestError("配置项必须是对象", {"version": version, "path": base})
    if "key" not in raw:
        raise ManifestError("配置项缺少 key", {"version": version, "path": f"{base}.key"})
    key = raw["key"]
    if not isinstance(key, str) or not KEY_PATTERN.match(key):
        raise ManifestError(
            "配置项 key 非法",
            {"version": version, "path": f"{base}.key", "value": key},
        )
    if "value" not in raw:
        raise ManifestError(
            "配置项缺少 value", {"version": version, "path": f"{base}.value"}
        )
    value = raw["value"]
    if isinstance(value, bool) or not isinstance(value, SUPPORTED_NUM_TYPES):
        raise ManifestError(
            "value 仅支持整数或浮点",
            {"version": version, "path": f"{base}.value", "type": type(value).__name__},
        )
    if not (value == value):  # NaN 拒绝，保证确定性
        raise ManifestError(
            "value 不能为 NaN", {"version": version, "path": f"{base}.value"}
        )
    targets = raw.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ManifestError(
            "targets 必须是非空数组",
            {"version": version, "path": f"{base}.targets"},
        )
    for j, target in enumerate(targets):
        if not isinstance(target, str) or target not in node_ids:
            raise ManifestError(
                f"目标节点 {target!r} 不在注册表中",
                {
                    "version": version,
                    "path": f"{base}.targets[{j}]",
                    "value": target,
                },
            )
    derivation = None
    if raw.get("derivation") is not None:
        d_raw = raw["derivation"]
        d_path = f"{base}.derivation"
        if not isinstance(d_raw, dict) or not isinstance(d_raw.get("refs"), list):
            raise ManifestError(
                "derivation 必须包含 refs 数组",
                {"version": version, "path": d_path},
            )
        if not d_raw["refs"]:
            raise ManifestError(
                "derivation.refs 不能为空数组",
                {"version": version, "path": f"{d_path}.refs"},
            )
        refs: List[Tuple[str, float]] = []
        for j, pair in enumerate(d_raw["refs"]):
            pair_path = f"{d_path}.refs[{j}]"
            if not isinstance(pair, list) or len(pair) != 2:
                raise ManifestError(
                    "ref 必须是 [key, coef] 二元数组",
                    {"version": version, "path": pair_path},
                )
            ref_key, coef = pair
            if not isinstance(ref_key, str) or not KEY_PATTERN.match(ref_key):
                raise ManifestError(
                    "派生引用 key 非法",
                    {"version": version, "path": f"{pair_path}[0]", "value": ref_key},
                )
            if isinstance(coef, bool) or not isinstance(coef, SUPPORTED_NUM_TYPES):
                raise ManifestError(
                    "派生系数仅支持整数或浮点",
                    {"version": version, "path": f"{pair_path}[1]"},
                )
            refs.append((ref_key, float(coef)))
        from .model import Derivation

        derivation = Derivation(refs=tuple(refs))
    return ConfigItem(
        key=key,
        value=float(value),
        targets=tuple(targets),
        derivation=derivation,
    )


def _validate_strategy(raw: Any, version: int) -> DispatchStrategy:
    base = f"versions[@version={version}].strategy"
    if not isinstance(raw, dict):
        raise ManifestError("分发策略必须是对象", {"version": version, "path": base})
    for field_name in REQUIRED_STRATEGY_FIELDS:
        if field_name not in raw:
            raise ManifestError(
                f"分发策略缺少字段 {field_name}",
                {"version": version, "path": f"{base}.{field_name}"},
            )
        val = raw[field_name]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ManifestError(
                f"分发策略字段 {field_name} 必须是整数",
                {"version": version, "path": f"{base}.{field_name}", "value": val},
            )
    if raw["batch_size"] <= 0:
        raise ManifestError(
            "batch_size 必须为正整数",
            {"version": version, "path": f"{base}.batch_size"},
        )
    if raw["timeout_ms"] <= 0:
        raise ManifestError(
            "timeout_ms 必须为正整数",
            {"version": version, "path": f"{base}.timeout_ms"},
        )
    if raw["max_retries"] < 0:
        raise ManifestError(
            "max_retries 不能为负",
            {"version": version, "path": f"{base}.max_retries"},
        )
    return DispatchStrategy(
        batch_size=raw["batch_size"],
        timeout_ms=raw["timeout_ms"],
        max_retries=raw["max_retries"],
    )


def load_manifest(
    path: str | Path, node_ids: set[str] | None = None
) -> List[ManifestVersion]:
    """加载并强校验整个版本清单，返回按版本号排序的版本列表。"""
    data = _load_json(path)
    node_ids = node_ids or set()
    if not isinstance(data, dict) or not isinstance(data.get("versions"), list):
        raise ManifestError(
            "清单缺少顶层 versions 数组", {"path": "versions"}
        )
    if not data["versions"]:
        raise ManifestError("清单至少包含一个版本", {"path": "versions"})

    versions: List[ManifestVersion] = []
    seen_versions: set[int] = set()
    seen_keys: set[str] = set()
    prev_version = None
    for vi, raw_version in enumerate(data["versions"]):
        vbase = f"versions[{vi}]"
        if not isinstance(raw_version, dict) or "version" not in raw_version:
            raise ManifestError(
                "版本条目缺少 version", {"path": f"{vbase}.version"}
            )
        version = raw_version["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise ManifestError(
                "版本号必须是整数",
                {"path": f"{vbase}.version", "value": version},
            )
        if version in seen_versions:
            raise ManifestError(
                f"版本号 {version} 重复",
                {"version": version, "path": f"{vbase}.version"},
            )
        if prev_version is not None and version <= prev_version:
            raise ManifestError(
                f"版本号必须单调递增: 前一版本 {prev_version}, 当前 {version}",
                {"version": version, "path": f"{vbase}.version"},
            )
        seen_versions.add(version)
        prev_version = version

        raw_items = raw_version.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise ManifestError(
                "items 必须是非空数组",
                {"version": version, "path": f"{vbase}.items"},
            )
        items: List[ConfigItem] = []
        local_keys: set[str] = set()
        for i, raw_item in enumerate(raw_items):
            item = _validate_item(raw_item, i, version, node_ids)
            if item.key in local_keys:
                raise ManifestError(
                    f"版本内配置项 key 重复: {item.key}",
                    {"version": version,
                     "path": f"versions[@version={version}].items[*].key"},
                )
            local_keys.add(item.key)
            seen_keys.add(item.key)
            items.append(item)

        # 派生引用的闭环校验（引用键须存在于同一版本）
        item_map = {it.key: it for it in items}
        for item in items:
            if item.derivation:
                for j, (ref_key, _coef) in enumerate(item.derivation.refs):
                    if ref_key not in item_map:
                        raise ManifestError(
                            f"配置项 {item.key} 派生引用 {ref_key} 在本版本不存在",
                            {
                                "version": version,
                                "path": (
                                    f"versions[@version={version}]"
                                    f".items[key={item.key}].derivation.refs[{j}][0]"
                                ),
                            },
                        )
        strategy = _validate_strategy(raw_version.get("strategy"), version)
        versions.append(
            ManifestVersion(version=version, items=tuple(items), strategy=strategy)
        )
    return versions

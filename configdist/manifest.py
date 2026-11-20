"""配置清单 v1/v2/v3 编解码与强校验。

格式演进：
  v1: {"format":"v1","entries":[{"key","value"}]}
  v2: + "manifest_version"，条目增加 "targets"
  v3: + "strategy"，条目增加 "priority"，新增 "derived" 派生关系
所有校验失败均以 E_FORMAT 整体拒绝，并给出出错字段路径与来源版本。
"""

import json
from dataclasses import dataclass, field

from .digest import DERIVED_OPS
from .errors import FormatError

FORMAT_V1 = "v1"
FORMAT_V2 = "v2"
FORMAT_V3 = "v3"
CURRENT_FORMAT = FORMAT_V3
KNOWN_FORMATS = (FORMAT_V1, FORMAT_V2, FORMAT_V3)

_ENTRY_KEYS = {
    FORMAT_V1: {"key", "value"},
    FORMAT_V2: {"key", "value", "targets"},
    FORMAT_V3: {"key", "value", "targets", "priority"},
}
_TOP_KEYS = {
    FORMAT_V1: {"format", "entries"},
    FORMAT_V2: {"format", "entries", "manifest_version"},
    FORMAT_V3: {"format", "entries", "manifest_version", "strategy", "derived"},
}
_STRATEGY_KEYS = {"timeout_ms", "max_retries", "global_concurrency", "node_quota"}
_DERIVED_KEYS = {"key", "op", "inputs", "targets", "priority"}


@dataclass
class Strategy:
    timeout_ms: int = 800
    max_retries: int = 2
    global_concurrency: int = 4
    node_quota: int = 2


@dataclass
class Entry:
    key: str
    value: object
    targets: list = field(default_factory=lambda: ["*"])
    priority: int = 0


@dataclass
class Derived:
    key: str
    op: str
    inputs: list
    targets: list = field(default_factory=lambda: ["*"])
    priority: int = 0


@dataclass
class Manifest:
    format: str
    manifest_version: int
    entries: list
    derived: list = field(default_factory=list)
    strategy: Strategy = field(default_factory=Strategy)

    def all_keys(self):
        return [e.key for e in self.entries] + [d.key for d in self.derived]


class _DuplicateKey(Exception):
    def __init__(self, key):
        self.key = key


def _no_dupes(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise _DuplicateKey(k)
        out[k] = v
    return out


def _parse_json(text):
    try:
        return json.loads(text, object_pairs_hook=_no_dupes)
    except _DuplicateKey as exc:
        raise FormatError(f"JSON 对象键重复: {exc.key!r}", path=f"$.{exc.key}",
                          source_version="unknown")
    except json.JSONDecodeError as exc:
        raise FormatError(f"JSON 解析失败: {exc}", path="$", source_version="unknown")


def _check_value(value, path, ver):
    if isinstance(value, bool) or value is None:
        if value is None:
            raise FormatError("键值类型不受支持: null", path=path, source_version=ver)
        return  # bool 允许
    if isinstance(value, (int, str)):
        return
    raise FormatError(f"键值类型不受支持: {type(value).__name__}",
                      path=path, source_version=ver)


def _check_targets(targets, path, ver):
    if not isinstance(targets, list) or not targets:
        raise FormatError("targets 必须为非空字符串数组", path=path, source_version=ver)
    for i, t in enumerate(targets):
        if not isinstance(t, str) or not t:
            raise FormatError("targets 元素必须为非空字符串",
                              path=f"{path}[{i}]", source_version=ver)


def _check_unknown(obj, allowed, path, ver):
    for k in obj:
        if k not in allowed:
            raise FormatError(f"未知字段: {k!r}", path=f"{path}.{k}", source_version=ver)


def decode_manifest(text):
    """解码并强校验清单文本，返回 Manifest；失败抛 FormatError(E_FORMAT)。"""
    raw = _parse_json(text)
    if not isinstance(raw, dict):
        raise FormatError("清单顶层必须为 JSON 对象", path="$", source_version="unknown")
    ver = raw.get("format")
    if ver not in KNOWN_FORMATS:
        raise FormatError(f"未知格式版本: {ver!r}", path="format",
                          source_version=repr(ver))
    _check_unknown(raw, _TOP_KEYS[ver], "$", ver)

    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise FormatError("entries 缺失或不是非空数组", path="entries", source_version=ver)

    if ver == FORMAT_V1:
        manifest_version = 1
    else:
        manifest_version = raw.get("manifest_version")
        if not isinstance(manifest_version, int) or isinstance(manifest_version, bool) \
                or manifest_version < 1:
            raise FormatError("manifest_version 必须为 >=1 的整数",
                              path="manifest_version", source_version=ver)

    entries = []
    seen = set()
    for i, rec in enumerate(entries_raw):
        path = f"entries[{i}]"
        if not isinstance(rec, dict):
            raise FormatError("条目必须为对象", path=path, source_version=ver)
        _check_unknown(rec, _ENTRY_KEYS[ver], path, ver)
        key = rec.get("key")
        if not isinstance(key, str) or not key:
            raise FormatError("key 缺失或不是非空字符串", path=f"{path}.key",
                              source_version=ver)
        if key in seen:
            raise FormatError(f"重复键: {key!r}", path=f"{path}.key", source_version=ver)
        seen.add(key)
        if "value" not in rec:
            raise FormatError("value 缺失", path=f"{path}.value", source_version=ver)
        _check_value(rec["value"], f"{path}.value", ver)
        targets = rec.get("targets", ["*"])
        if ver != FORMAT_V1:
            _check_targets(targets, f"{path}.targets", ver)
        priority = rec.get("priority", 0)
        if ver == FORMAT_V3 and (not isinstance(priority, int)
                                 or isinstance(priority, bool)):
            raise FormatError("priority 必须为整数", path=f"{path}.priority",
                              source_version=ver)
        entries.append(Entry(key=key, value=rec["value"],
                             targets=list(targets), priority=priority))

    strategy = Strategy()
    if ver == FORMAT_V3 and "strategy" in raw:
        s = raw["strategy"]
        if not isinstance(s, dict):
            raise FormatError("strategy 必须为对象", path="strategy", source_version=ver)
        _check_unknown(s, _STRATEGY_KEYS, "strategy", ver)
        for name in _STRATEGY_KEYS:
            if name in s:
                v = s[name]
                if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                    raise FormatError(f"strategy.{name} 必须为非负整数",
                                      path=f"strategy.{name}", source_version=ver)
                setattr(strategy, name, v)

    derived = []
    if ver == FORMAT_V3:
        derived_raw = raw.get("derived", [])
        if not isinstance(derived_raw, list):
            raise FormatError("derived 必须为数组", path="derived", source_version=ver)
        available = set(seen)
        numeric = {e.key for e in entries
                   if isinstance(e.value, int) and not isinstance(e.value, bool)}
        for i, rec in enumerate(derived_raw):
            path = f"derived[{i}]"
            if not isinstance(rec, dict):
                raise FormatError("派生条目必须为对象", path=path, source_version=ver)
            _check_unknown(rec, _DERIVED_KEYS, path, ver)
            key = rec.get("key")
            if not isinstance(key, str) or not key:
                raise FormatError("派生 key 缺失或不是非空字符串",
                                  path=f"{path}.key", source_version=ver)
            if key in seen:
                raise FormatError(f"重复键: {key!r}", path=f"{path}.key",
                                  source_version=ver)
            seen.add(key)
            op = rec.get("op")
            if op not in DERIVED_OPS:
                raise FormatError(f"未知派生运算: {op!r}", path=f"{path}.op",
                                  source_version=ver)
            inputs = rec.get("inputs")
            if not isinstance(inputs, list) or len(inputs) != 2 \
                    or not all(isinstance(x, str) for x in inputs):
                raise FormatError("inputs 必须为两个键名的数组",
                                  path=f"{path}.inputs", source_version=ver)
            for j, ref in enumerate(inputs):
                if ref not in available:
                    raise FormatError(f"派生输入引用了不存在的键: {ref!r}",
                                      path=f"{path}.inputs[{j}]", source_version=ver)
                if ref not in numeric:
                    raise FormatError(f"派生输入不是数值键: {ref!r}",
                                      path=f"{path}.inputs[{j}]", source_version=ver)
            available.add(key)
            numeric.add(key)
            targets = rec.get("targets", ["*"])
            _check_targets(targets, f"{path}.targets", ver)
            priority = rec.get("priority", 0)
            if not isinstance(priority, int) or isinstance(priority, bool):
                raise FormatError("priority 必须为整数", path=f"{path}.priority",
                                  source_version=ver)
            derived.append(Derived(key=key, op=op, inputs=list(inputs),
                                   targets=list(targets), priority=priority))

    return Manifest(format=ver, manifest_version=manifest_version,
                    entries=entries, derived=derived, strategy=strategy)


def detect_format(text):
    """仅检测格式版本，用于归一前的分流。"""
    raw = _parse_json(text)
    if not isinstance(raw, dict):
        raise FormatError("清单顶层必须为 JSON 对象", path="$", source_version="unknown")
    ver = raw.get("format")
    if ver not in KNOWN_FORMATS:
        raise FormatError(f"未知格式版本: {ver!r}", path="format",
                          source_version=repr(ver))
    return ver, raw


def upgrade_entry(record, source_version):
    """把单条 v1/v2 条目升级为 v3 条目（记录级升级的基本单元）。"""
    out = {"key": record["key"], "value": record["value"]}
    if source_version == FORMAT_V1:
        out["targets"] = ["*"]
    else:
        out["targets"] = list(record.get("targets", ["*"]))
    out["priority"] = 0
    return out


def encode_manifest(manifest):
    """把 Manifest 序列化为当前版本（v3）的格式化 JSON 文本。"""
    obj = {
        "format": FORMAT_V3,
        "manifest_version": manifest.manifest_version,
        "strategy": {
            "timeout_ms": manifest.strategy.timeout_ms,
            "max_retries": manifest.strategy.max_retries,
            "global_concurrency": manifest.strategy.global_concurrency,
            "node_quota": manifest.strategy.node_quota,
        },
        "entries": [
            {"key": e.key, "value": e.value, "targets": e.targets,
             "priority": e.priority}
            for e in manifest.entries
        ],
        "derived": [
            {"key": d.key, "op": d.op, "inputs": d.inputs,
             "targets": d.targets, "priority": d.priority}
            for d in manifest.derived
        ],
    }
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

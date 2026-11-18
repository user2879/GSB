"""规则数据模型与强校验。

校验分为两层：
1. 单条/集合的字段级校验，失败统一以 E_RULE 报告（含规则标识与字段路径）；
2. 依赖图成环由 graph 模块以 E_DEPEND 报告。
任何字段级错误都会整体拒绝加载，绝不部分加载。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import FieldViolation

LOOPBACK_ADDRESSES = frozenset({"127.0.0.1", "::1"})
PORT_MIN, PORT_MAX = 1, 65535


@dataclass(frozen=True)
class Rule:
    rule_id: str
    dst_addr: str
    port_prefixes: tuple[int, ...]
    rate_kbps: int
    burst_bytes: int
    queue_limit_packets: int
    upstream: tuple[str, ...] = field(default_factory=tuple)
    actions: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "match": {
                "dst_addr": self.dst_addr,
                "port_prefixes": list(self.port_prefixes),
            },
            "shaping": {
                "rate_kbps": self.rate_kbps,
                "burst_bytes": self.burst_bytes,
                "queue_limit_packets": self.queue_limit_packets,
            },
            "upstream": list(self.upstream),
            "actions": list(self.actions),
        }


def _is_bool(value: Any) -> bool:
    # bool 是 int 的子类，须显式排除。
    return isinstance(value, bool)


def _validate_one(raw: Any, seen_ids: set[str],
                  violations: list[FieldViolation]) -> None:
    if not isinstance(raw, dict):
        violations.append(FieldViolation("<unknown>", "$", "规则必须是 JSON 对象"))
        return

    rid = raw.get("rule_id")
    if not isinstance(rid, str) or not rid:
        violations.append(FieldViolation(str(rid) if rid is not None else "<missing>",
                                         "rule_id", "规则标识必须是非空字符串"))
        rid = rid if isinstance(rid, str) and rid else "<invalid>"
    elif rid in seen_ids:
        violations.append(FieldViolation(rid, "rule_id", "规则标识重复"))
    else:
        seen_ids.add(rid)

    match = raw.get("match")
    if not isinstance(match, dict):
        violations.append(FieldViolation(rid, "match", "缺少匹配条件对象"))
    else:
        addr = match.get("dst_addr")
        if not isinstance(addr, str) or addr not in LOOPBACK_ADDRESSES:
            violations.append(FieldViolation(
                rid, "match.dst_addr",
                f"目的地址必须是回环字面量 {sorted(LOOPBACK_ADDRESSES)}"))
        prefixes = match.get("port_prefixes")
        if not isinstance(prefixes, list) or not prefixes:
            violations.append(FieldViolation(
                rid, "match.port_prefixes", "端口前缀必须是非空整数数组"))
        else:
            bad = False
            for idx, prefix in enumerate(prefixes):
                if _is_bool(prefix) or not isinstance(prefix, int) \
                        or not (PORT_MIN <= prefix <= PORT_MAX):
                    violations.append(FieldViolation(
                        rid, f"match.port_prefixes[{idx}]",
                        f"端口前缀必须是 {PORT_MIN}..{PORT_MAX} 的整数"))
                    bad = True
            if not bad and len(set(prefixes)) != len(prefixes):
                violations.append(FieldViolation(
                    rid, "match.port_prefixes", "端口前缀不得重复"))

    shaping = raw.get("shaping")
    if not isinstance(shaping, dict):
        violations.append(FieldViolation(rid, "shaping", "缺少整形参数对象"))
    else:
        rate = shaping.get("rate_kbps")
        if _is_bool(rate) or not isinstance(rate, int) or rate <= 0:
            violations.append(FieldViolation(
                rid, "shaping.rate_kbps", "承诺速率必须是正整数（kbps）"))
        burst = shaping.get("burst_bytes")
        if _is_bool(burst) or not isinstance(burst, int) or burst <= 0:
            violations.append(FieldViolation(
                rid, "shaping.burst_bytes", "突发字节必须是正整数"))
        queue_limit = shaping.get("queue_limit_packets")
        if _is_bool(queue_limit) or not isinstance(queue_limit, int) \
                or queue_limit < 0:
            violations.append(FieldViolation(
                rid, "shaping.queue_limit_packets",
                "队列上限包数必须是非负整数"))

    upstream = raw.get("upstream", [])
    if not isinstance(upstream, list) or not all(
            isinstance(item, str) and item for item in upstream):
        violations.append(FieldViolation(
            rid, "upstream", "上游引用必须是规则标识字符串数组"))
    elif len(set(upstream)) != len(upstream):
        violations.append(FieldViolation(rid, "upstream", "上游引用不得重复"))

    actions = raw.get("actions", [])
    if not isinstance(actions, list) or not all(
            isinstance(item, str) and item for item in actions):
        violations.append(FieldViolation(
            rid, "actions", "整形动作组合必须是字符串数组"))


def validate_rule_dicts(raw_rules: Any) -> list[FieldViolation]:
    """对原始 JSON 结构做字段级强校验，返回全部违规（不抛异常）。"""
    violations: list[FieldViolation] = []
    if not isinstance(raw_rules, list) or not raw_rules:
        violations.append(FieldViolation("<root>", "$",
                                         "规则集必须是非空 JSON 数组"))
        return violations
    seen: set[str] = set()
    for raw in raw_rules:
        _validate_one(raw, seen, violations)
    return violations


def parse_rules(raw_rules: list[dict[str, Any]]) -> list[Rule]:
    """把已通过字段校验的 dict 列表转换为不可变 Rule 列表（保持文件顺序）。"""
    rules: list[Rule] = []
    for raw in raw_rules:
        rules.append(Rule(
            rule_id=raw["rule_id"],
            dst_addr=raw["match"]["dst_addr"],
            port_prefixes=tuple(raw["match"]["port_prefixes"]),
            rate_kbps=raw["shaping"]["rate_kbps"],
            burst_bytes=raw["shaping"]["burst_bytes"],
            queue_limit_packets=raw["shaping"]["queue_limit_packets"],
            upstream=tuple(raw.get("upstream", [])),
            actions=tuple(raw.get("actions", [])),
        ))
    return rules


def rule_as_dict(rule: Rule) -> dict[str, Any]:
    return asdict(rule)

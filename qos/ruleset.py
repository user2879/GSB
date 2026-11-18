"""规则集加载：JSON 读取 + 字段强校验 + 依赖校验的整体拒绝语义。"""
from __future__ import annotations

import json
from pathlib import Path

from .errors import RuleError
from .graph import DependencyGraph
from .models import Rule, parse_rules, validate_rule_dicts


def load_rules(path: str | Path) -> list[Rule]:
    """从本地 JSON 加载并强校验规则集。

    任何字段级错误 -> E_RULE（整体拒绝，附规则标识与字段路径）；
    依赖成环 -> E_DEPEND；
    上游缺失等引用问题 -> E_RULE（字段路径 upstream）。
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuleError(f"无法读取规则文件 {path}: {exc}",
                        details={"path": str(path)}) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuleError(
            "规则文件 JSON 损坏，整体拒绝加载",
            details={"path": str(path), "field_path": "$",
                     "reason": f"JSONDecodeError: {exc.msg}",
                     "line": exc.lineno, "column": exc.colno},
        ) from exc

    violations = validate_rule_dicts(raw)
    if violations:
        raise RuleError(
            "规则集字段校验失败，整体拒绝加载",
            details={"violations": [v.to_dict() for v in violations]},
        )

    rules = parse_rules(raw)
    graph = DependencyGraph(rules)
    ref_violations = graph.validate_references()
    if ref_violations:
        raise RuleError(
            "规则集上游引用校验失败，整体拒绝加载",
            details={"violations": [v.to_dict() for v in ref_violations]},
        )
    graph.ensure_acyclic()  # 成环抛 E_DEPEND
    return rules


def build_graph(rules: list[Rule]) -> DependencyGraph:
    graph = DependencyGraph(rules)
    violations = graph.validate_all()
    if violations:
        raise RuleError(
            "规则集引用校验失败",
            details={"violations": [v.to_dict() for v in violations]},
        )
    return graph


def ruleset_version(rules: list[Rule]) -> str:
    """规则集内容版本：对按 rule_id 排序后的规范表示取哈希。"""
    from .canonical import short_hash
    ordered = sorted((r.to_dict() for r in rules),
                     key=lambda item: item["rule_id"])
    return short_hash(ordered)

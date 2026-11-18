"""样本、规则资源与打分参数加载层。

加载阶段产生的任何问题都转为带稳定错误码的失败项，
不抛出到主流程之外（配置文件错误除外，配置错误必须中止启动）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .errors import (
    ProofreadError,
    E_EMPTY_TEXT,
    E_PARSE_INVALID,
    E_RULE_UNKNOWN,
    E_RESOURCE_INVALID,
    E_DUPLICATE_ID,
)
from .fingerprint import content_fingerprint
from .models import (
    Sample,
    Rule,
    RuleResource,
    ScoringParams,
    ERROR_TYPES,
    EDIT_REPLACE,
    EDIT_INSERT,
    WRONG_CHAR,
    REPEAT,
    TERM,
    OMISSION,
)


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def load_samples(path: str | Path) -> Tuple[List[Sample], List[Dict[str, Any]]]:
    """逐行解析 JSONL，返回 (有效样本, 失败清单)。

    失败项按文件行号自然累积，后续统一按样本 ID 排序，保证顺序确定性。
    """
    samples: List[Sample] = []
    failures: List[Dict[str, Any]] = []
    seen_ids = set()
    source = str(path)

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(ProofreadError(
                    E_PARSE_INVALID, f"第 {line_no} 行 JSON 非法: {exc.msg}",
                    source=source).to_item())
                continue
            if not isinstance(obj, dict):
                failures.append(ProofreadError(
                    E_PARSE_INVALID, f"第 {line_no} 行不是 JSON 对象",
                    source=source).to_item())
                continue

            sample_id = obj.get("sample_id")
            text = obj.get("text")
            if not isinstance(sample_id, str) or not sample_id:
                failures.append(ProofreadError(
                    E_PARSE_INVALID, f"第 {line_no} 行缺少有效 sample_id",
                    source=source).to_item())
                continue
            if not isinstance(text, str) or not text.strip():
                failures.append(ProofreadError(
                    E_EMPTY_TEXT, f"样本 {sample_id} 原文为空",
                    sample_id=sample_id, source=source).to_item())
                continue
            if sample_id in seen_ids:
                failures.append(ProofreadError(
                    E_DUPLICATE_ID, f"样本 ID 重复: {sample_id}",
                    sample_id=sample_id, source=source).to_item())
                continue
            seen_ids.add(sample_id)

            for hint in obj.get("rule_hints", []) or []:
                # 引用校验在规则加载后做二次核验，这里仅保留字段
                if not isinstance(hint, str):
                    failures.append(ProofreadError(
                        E_PARSE_INVALID, f"样本 {sample_id} rule_hints 含非字符串项",
                        sample_id=sample_id, source=source).to_item())

            hints = obj.get("rule_hints", []) or []
            if not isinstance(hints, list) or not all(isinstance(h, str) for h in hints):
                failures.append(ProofreadError(
                    E_PARSE_INVALID, f"样本 {sample_id} rule_hints 必须是字符串数组",
                    sample_id=sample_id, source=source).to_item())
                hints = []
            samples.append(Sample(
                sample_id=sample_id,
                text=text,
                context=obj.get("context"),
                truth=obj.get("truth"),
                line_no=line_no,
                rule_hints=list(hints),
            ))
    return samples, failures


def _check_chars_in_table(label: str, text: str, table: frozenset, source: str) -> None:
    outside = sorted({ch for ch in text if ch not in table})
    if outside:
        raise ProofreadError(
            E_RESOURCE_INVALID,
            f"{label} 含受控字表外字符: {''.join(outside)}",
            source=source)


def load_rules(path: str | Path) -> RuleResource:
    source = str(path)
    with Path(path).open("r", encoding="utf-8") as handle:
        try:
            obj = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ProofreadError(E_PARSE_INVALID, f"规则资源 JSON 非法: {exc.msg}",
                                 source=source) from exc

    table_list = obj.get("char_table")
    if not isinstance(table_list, list) or not table_list:
        raise ProofreadError(E_RESOURCE_INVALID, "char_table 必须是非空数组", source=source)
    table = frozenset(table_list)
    if any(not isinstance(ch, str) or len(ch) != 1 for ch in table):
        raise ProofreadError(E_RESOURCE_INVALID, "char_table 只能含单字符", source=source)

    rule_weights = obj.get("rule_weights", {})
    if not isinstance(rule_weights, dict):
        raise ProofreadError(E_RESOURCE_INVALID, "rule_weights 必须是映射", source=source)
    for rule_id, weight in rule_weights.items():
        if not _is_finite(weight):
            raise ProofreadError(E_RESOURCE_INVALID,
                                 f"规则权重 {rule_id} 不是有限数", source=source)

    rules: List[Rule] = []
    known_ids = set()

    def _consume(rule_id: str, etype: str, pattern: str, replacement: str,
                 edit_type: str) -> None:
        if not rule_id or rule_id in known_ids:
            raise ProofreadError(E_RESOURCE_INVALID,
                                 f"规则 ID 为空或重复: {rule_id!r}", source=source)
        if not pattern:
            raise ProofreadError(E_RESOURCE_INVALID,
                                 f"规则 {rule_id} pattern 为空", source=source)
        _check_chars_in_table(f"规则 {rule_id} pattern", pattern, table, source)
        if replacement:
            check_text = replacement.replace("|", "") if edit_type == EDIT_INSERT                 else replacement
            _check_chars_in_table(f"规则 {rule_id} replacement", check_text,
                                  table, source)
        weight = float(rule_weights.get(rule_id, 1.0))
        if not math.isfinite(weight):
            raise ProofreadError(E_RESOURCE_INVALID,
                                 f"规则 {rule_id} 权重非有限数", source=source)
        known_ids.add(rule_id)
        rules.append(Rule(rule_id, etype, pattern, replacement, edit_type, weight))

    for pair in obj.get("confusion_pairs", []):
        wrong = pair.get("wrong")
        correct = pair.get("correct")
        rule_id = pair.get("rule_id")
        if not isinstance(wrong, str) or len(wrong) != 1 or not isinstance(correct, str) \
                or len(correct) != 1:
            raise ProofreadError(E_RESOURCE_INVALID,
                                 f"混淆字对 {rule_id} 两侧必须为单字符", source=source)
        _check_chars_in_table(f"混淆字对 {rule_id} wrong 侧", wrong, table, source)
        _check_chars_in_table(f"混淆字对 {rule_id} correct 侧", correct, table, source)
        _consume(rule_id, WRONG_CHAR, wrong, correct, EDIT_REPLACE)

    for item in obj.get("repeat_rules", []):
        _consume(item["rule_id"], REPEAT, item["pattern"],
                 item.get("replacement", ""), item.get("edit_type", EDIT_REPLACE))

    for item in obj.get("term_rules", []):
        _consume(item["rule_id"], TERM, item["pattern"],
                 item["replacement"], EDIT_REPLACE)

    for item in obj.get("omission_rules", []):
        rule_id = item["rule_id"]
        pattern = item["pattern"]
        replacement = item["replacement"]
        _consume(rule_id, OMISSION, pattern, replacement, EDIT_INSERT)
        # insert_offset 单独挂到最后一条规则上
        rules[-1] = Rule(
            rule_id=rules[-1].rule_id, etype=rules[-1].etype,
            pattern=rules[-1].pattern, replacement=rules[-1].replacement,
            edit_type=rules[-1].edit_type, weight=rules[-1].weight,
            insert_offset=int(item.get("insert_offset", 0)))

    for rule_id in rule_weights:
        if rule_id not in known_ids:
            raise ProofreadError(E_RULE_UNKNOWN,
                                 f"rule_weights 引用未知规则: {rule_id}", source=source)

    rules.sort(key=lambda rule: rule.rule_id)
    fingerprint = content_fingerprint({
        "char_table": sorted(table),
        "rules": [r.to_fingerprint_obj() for r in rules],
        "rule_weights": rule_weights,
    })
    return RuleResource(frozenset(table), rules, dict(rule_weights), fingerprint)


def validate_rule_hints(samples: List[Sample], resource: RuleResource,
                        source: str) -> List[Dict[str, Any]]:
    """样本 rule_hints 引用未知规则时进入失败清单（E_RULE_UNKNOWN）。"""
    failures: List[Dict[str, Any]] = []
    known = resource.rule_map()
    for sample in samples:
        for hint in sample.rule_hints:
            if hint not in known:
                failures.append(ProofreadError(
                    E_RULE_UNKNOWN, f"样本 {sample.sample_id} 引用未知规则: {hint}",
                    sample_id=sample.sample_id, source=source).to_item())
    return failures


def load_scoring_params(path: str | Path) -> ScoringParams:
    source = str(path)
    with Path(path).open("r", encoding="utf-8") as handle:
        try:
            obj = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ProofreadError(E_PARSE_INVALID, f"打分参数 JSON 非法: {exc.msg}",
                                 source=source) from exc

    detect_weights = obj.get("detect_weights")
    base_scores = obj.get("base_scores")
    if not isinstance(detect_weights, dict) or not isinstance(base_scores, dict):
        raise ProofreadError(E_RESOURCE_INVALID,
                             "detect_weights / base_scores 必须是映射", source=source)
    for etype in ERROR_TYPES:
        if etype not in detect_weights:
            raise ProofreadError(E_RULE_UNKNOWN,
                                 f"detect_weights 缺少错误类型: {etype}", source=source)
        if etype not in base_scores:
            raise ProofreadError(E_RULE_UNKNOWN,
                                 f"base_scores 缺少错误类型: {etype}", source=source)
    for label, table in (("detect_weights", detect_weights), ("base_scores", base_scores)):
        for etype, value in table.items():
            if etype not in ERROR_TYPES:
                raise ProofreadError(E_RULE_UNKNOWN,
                                     f"{label} 引用未知错误类型: {etype}", source=source)
            if not _is_finite(value):
                raise ProofreadError(E_RESOURCE_INVALID,
                                     f"{label}.{etype} 不是有限数", source=source)
    threshold = obj.get("decision_threshold")
    if not _is_finite(threshold):
        raise ProofreadError(E_RESOURCE_INVALID,
                             "decision_threshold 不是有限数", source=source)

    fingerprint = content_fingerprint({
        "detect_weights": detect_weights,
        "base_scores": base_scores,
        "decision_threshold": threshold,
    })
    return ScoringParams(
        detect_weights={k: float(v) for k, v in detect_weights.items()},
        base_scores={k: float(v) for k, v in base_scores.items()},
        decision_threshold=float(threshold),
        source_fingerprint=fingerprint,
    )

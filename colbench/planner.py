"""列裁剪与谓词下推的扫描计划。

支持的查询语言（确定性、无外部依赖）：

* 比较：``col = v``、``col <> v``、``col > v``、``col >= v``、
  ``col < v``、``col <= v``；
* 集合：``col IN (v1, v2, ...)``；
* 逻辑：``AND``（也支持小写 ``and``）、括号可任意嵌套、空白可任意；
* 投影：``SELECT a, b FROM ... WHERE ...``，也接受仅 WHERE 表达式。

规范化规则：AND 合取项按稳定键排序、字面量按类型规范化，因此空白与括号
差异不会改变计划摘要。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import SchemaError
from .schema import SegmentSpec

_TOKEN_RE = re.compile(
    r"\s*(\(|\)|,|>=|<=|<>|!=|=|>|<|\bIN\b|\bAND\b|\band\b|"
    r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|[+-]?\d+|[A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class Predicate:
    """叶子谓词；kind 为 eq/ne/gt/ge/lt/le/in。"""

    kind: str
    column: str
    values: tuple  # eq/ne/范围为单元素；in 为多元素

    def key(self) -> tuple:
        return (self.column, self.kind,
                tuple(repr(value) for value in self.values))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "column": self.column,
                "values": list(self.values)}


@dataclass
class ScanPlan:
    query_text: str
    projected_columns: list[str]
    pruned_columns: list[str]
    pushed: list[Predicate]
    retained: list[Predicate]
    reasons: list[str]
    candidate_blocks: list[dict] = field(default_factory=list)
    skipped_blocks: list[dict] = field(default_factory=list)
    plan_digest: str = ""

    def summary(self) -> dict:
        return {
            "plan_digest": self.plan_digest,
            "projected_columns": self.projected_columns,
            "pruned_columns": self.pruned_columns,
            "pushed": [predicate.to_dict() for predicate in self.pushed],
            "retained": [predicate.to_dict()
                         for predicate in self.retained],
            "candidate_block_count": len(self.candidate_blocks),
            "skipped_block_count": len(self.skipped_blocks),
            "reasons": self.reasons,
        }


# ---------------------------------------------------------------------------
# 分词与递归下降解析
# ---------------------------------------------------------------------------
class _Parser:
    def __init__(self, text: str):
        self.tokens = self._tokenize(text)
        self.pos = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = []
        pos = 0
        while pos < len(text):
            match = _TOKEN_RE.match(text, pos)
            if not match:
                raise SchemaError(f"查询在位置 {pos} 无法解析：{text[pos:]!r}")
            token = match.group(1)
            pos = match.end()
            if token:
                tokens.append(token)
        return tokens

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expect(self, token: str) -> None:
        if self.peek() != token:
            raise SchemaError(f"查询语法错误：期望 {token!r}，"
                              f"实际 {self.peek()!r}")
        self.pos += 1

    def parse(self) -> list[Predicate]:
        predicates = self._parse_and()
        if self.peek() is not None:
            raise SchemaError(f"查询存在未解析尾部：{self.peek()!r}")
        return predicates

    def _parse_and(self) -> list[Predicate]:
        items = self._parse_factor()
        while self.peek() in ("AND", "and"):
            self.take()
            items.extend(self._parse_factor())
        return items

    def _parse_factor(self) -> list[Predicate]:
        if self.peek() == "(":
            self.take()
            inner = self._parse_and()
            self.expect(")")
            return inner
        return [self._parse_leaf()]

    def _parse_leaf(self) -> Predicate:
        column = self.take()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column or ""):
            raise SchemaError(f"非法列名：{column!r}")
        operator = self.take()
        if operator in ("IN", "in"):
            self.expect("(")
            values = [self._parse_literal(self.take())]
            while self.peek() == ",":
                self.take()
                values.append(self._parse_literal(self.take()))
            self.expect(")")
            return Predicate("in", column, tuple(values))
        mapping = {"=": "eq", "==": "eq", "<>": "ne", "!=": "ne",
                   ">": "gt", ">=": "ge", "<": "lt", "<=": "le"}
        if operator not in mapping:
            raise SchemaError(f"不支持的运算符：{operator!r}")
        value = self._parse_literal(self.take())
        return Predicate(mapping[operator], column, (value,))

    @staticmethod
    def _parse_literal(token: str):
        if token is None:
            raise SchemaError("查询意外结束，缺少字面量")
        if (token.startswith("'") and token.endswith("'")) or \
                (token.startswith('"') and token.endswith('"')):
            return token[1:-1].replace("\\'", "'").replace('\\"', '"')
        try:
            return int(token)
        except ValueError as exc:
            raise SchemaError(f"仅支持整数字面量与字符串字面量：{token!r}") \
                from exc


def parse_query(text: str) -> tuple[list[str], list[Predicate]]:
    """解析查询，返回投影列与合取谓词列表（未排序）。"""
    cleaned = text.strip().rstrip(";").strip()
    if cleaned[:6].upper() == "WHERE ":
        cleaned = cleaned[6:].strip()
    projected: list[str] = []
    where_text = cleaned
    match = re.match(r"(?is)^SELECT\s+(.+?)\s+FROM\s+[A-Za-z0-9_.]+"
                     r"(?:\s+WHERE\s+(.+))?$", cleaned)
    if match:
        projected = [part.strip() for part in match.group(1).split(",")]
        where_text = match.group(2) or ""
    elif re.match(r"(?is)^SELECT\b", cleaned):
        raise SchemaError("SELECT 查询必须包含 FROM 子句")
    predicates = _Parser(where_text).parse() if where_text else []
    if "*" in projected:
        projected = []
    return projected, predicates


def _normalize_in(predicate: Predicate) -> Predicate:
    if predicate.kind != "in":
        return predicate
    values = sorted(set(predicate.values),
                    key=lambda value: (type(value).__name__, repr(value)))
    return Predicate("in", predicate.column, tuple(values))


def build_plan(query_text: str, spec: SegmentSpec) -> ScanPlan:
    projected, raw_predicates = parse_query(query_text)
    known = set(spec.column_names)

    unknown = sorted({column for column in projected
                      if column not in known}
                     | {predicate.column for predicate in raw_predicates
                        if predicate.column not in known})
    if unknown:
        raise SchemaError(f"查询引用了未知列：{unknown}",
                          {"unknown": unknown})

    referenced = set(projected)
    predicates = []
    for predicate in raw_predicates:
        normalized = _normalize_in(predicate)
        predicates.append(normalized)
        referenced.add(normalized.column)
    # 稳定排序：合取项顺序不影响语义，保证括号/空白差异下摘要一致
    predicates.sort(key=Predicate.key)

    pruned = [column.name for column in spec.columns
              if column.name not in referenced]
    pushed: list[Predicate] = []
    retained: list[Predicate] = []
    reasons: list[str] = []
    for predicate in predicates:
        if predicate.kind in {"eq", "ne", "gt", "ge", "lt", "le", "in"}:
            pushed.append(predicate)
            reasons.append(
                f"谓词 {predicate.column}.{predicate.kind} 可由块区域图/"
                "字典独立判定，下推到块级")
        else:
            retained.append(predicate)
            reasons.append(f"谓词 {predicate.kind} 保留在行级过滤")

    candidate_blocks, skipped_blocks = _select_blocks(
        spec, pushed, frozenset(referenced))
    plan = ScanPlan(
        query_text=query_text,
        projected_columns=[name for name in spec.column_names
                           if name in referenced and (not projected
                                                      or name in projected)]
        if projected else [name for name in spec.column_names
                           if name in referenced],
        pruned_columns=pruned,
        pushed=pushed, retained=retained, reasons=reasons,
        candidate_blocks=candidate_blocks, skipped_blocks=skipped_blocks)
    plan.plan_digest = stable_digest(plan, spec)
    return plan


def _select_blocks(spec: SegmentSpec,
                   predicates: list[Predicate],
                   referenced: frozenset) -> tuple[list[dict], list[dict]]:
    by_column: dict[str, list[Predicate]] = {}
    for predicate in predicates:
        by_column.setdefault(predicate.column, []).append(predicate)

    candidates: list[dict] = []
    skipped: list[dict] = []
    for group in spec.row_groups:
        for column in spec.columns:
            if column.name not in referenced:
                continue
            key = (group.index, column.name, 0)
            block_spec = next((block for block in spec.blocks
                               if (block.row_group, block.column,
                                   block.block_index) == key), None)
            if block_spec is None:
                continue
            values = block_spec.values
            minimum, maximum = min(values), max(values)
            distinct = set(values)
            matched, block_reasons = True, []
            for predicate in by_column.get(column.name, []):
                ok, reason = _evaluate_block(predicate, minimum, maximum,
                                             distinct)
                block_reasons.append(reason)
                if not ok:
                    matched = False
            record = {"segment_id": spec.segment_id,
                      "row_group": group.index, "column": column.name,
                      "block_index": 0,
                      "row_range": [group.start, group.end],
                      "reasons": block_reasons}
            if matched:
                candidates.append(record)
            else:
                record["skip_reason"] = "; ".join(block_reasons)
                skipped.append(record)
    candidates.sort(key=lambda item: (item["row_group"], item["column"]))
    skipped.sort(key=lambda item: (item["row_group"], item["column"]))
    return candidates, skipped


def _evaluate_block(predicate: Predicate, minimum, maximum,
                    distinct: set) -> tuple[bool, str]:
    kind = predicate.kind
    if kind == "eq":
        value = predicate.values[0]
        ok = minimum <= value <= maximum and value in distinct
        return ok, f"eq({value}) 区域[{minimum},{maximum}] 且字典命中={ok}"
    if kind == "ne":
        value = predicate.values[0]
        # 块内所有值都等于该值时才可能整体跳过；否则保留块、行级过滤
        ok = not (minimum == maximum == value)
        return ok, f"ne({value}) 块内非常量或常量不同，保留={ok}"
    if kind == "gt":
        value = predicate.values[0]
        ok = maximum > value
        return ok, f"gt({value}) 最大值 {maximum} > {value} = {ok}"
    if kind == "ge":
        value = predicate.values[0]
        ok = maximum >= value
        return ok, f"ge({value}) 最大值 {maximum} >= {value} = {ok}"
    if kind == "lt":
        value = predicate.values[0]
        ok = minimum < value
        return ok, f"lt({value}) 最小值 {minimum} < {value} = {ok}"
    if kind == "le":
        value = predicate.values[0]
        ok = minimum <= value
        return ok, f"le({value}) 最小值 {minimum} <= {value} = {ok}"
    if kind == "in":
        hits = [value for value in predicate.values if value in distinct]
        ok = bool(hits)
        return ok, f"in 字典命中 {hits}"
    return True, "未识别谓词保留块"


def stable_digest(plan: ScanPlan, spec: SegmentSpec) -> str:
    import hashlib
    payload = {
        "segment_id": spec.segment_id,
        "projected": plan.projected_columns,
        "pruned": plan.pruned_columns,
        "pushed": [predicate.to_dict() for predicate in plan.pushed],
        "retained": [predicate.to_dict()
                     for predicate in plan.retained],
        "candidates": [(item["row_group"], item["column"])
                       for item in plan.candidate_blocks],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load_query(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()

"""Query model: parsing, canonical normalization, column pruning and
predicate push-down planning against row-group/block min-max metadata."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from .errors import QueryError
from .model import BlockId, SegmentDefinition
from .segment import BlockMeta, LoadedSegment

SUPPORTED_OPS = {"eq", "ne", "gt", "ge", "lt", "le", "in", "and"}


@dataclass(frozen=True)
class Predicate:
    op: str
    column: str = ""
    value: Any = None
    children: Tuple["Predicate", ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        if self.op == "and":
            return {"op": "and", "args": [c.to_dict() for c in self.children]}
        return {"op": self.op, "column": self.column, "value": self.value}


def parse_query(raw: Dict[str, Any]) -> Tuple[List[str], Predicate]:
    select = list(raw.get("select", []))
    if not select or any(not isinstance(name, str) for name in select):
        raise QueryError("query.select must be a non-empty list of column names", {"raw": raw})
    if "filter" not in raw:
        raise QueryError("query.filter is required", {"raw": raw})
    predicate = _parse_predicate(raw["filter"])
    referenced = set(select) | set(_referenced_columns(predicate))
    return sorted(referenced), predicate


def _parse_predicate(node: Dict[str, Any]) -> Predicate:
    if not isinstance(node, dict) or "op" not in node:
        raise QueryError("predicate must be an object with an op field", {"node": node})
    op = node["op"].lower().strip()
    if op not in SUPPORTED_OPS:
        raise QueryError(
            f"unsupported operator: {node['op']}",
            {"operator": node["op"], "supported": sorted(SUPPORTED_OPS)},
        )
    if op == "and":
        args = node.get("args")
        if not isinstance(args, list) or len(args) < 2:
            raise QueryError("AND predicate requires at least two args", {"node": node})
        children = tuple(_parse_predicate(arg) for arg in args)
        return Predicate(op="and", children=tuple(sorted(children, key=_predicate_key)))
    column = node.get("column")
    if not isinstance(column, str) or not column:
        raise QueryError("leaf predicate requires a string column", {"node": node})
    if op == "in":
        values = node.get("value")
        if not isinstance(values, list) or not values:
            raise QueryError("IN predicate requires a non-empty value list", {"node": node})
        value = sorted(int(v) for v in values)
    else:
        value = int(node["value"])
    return Predicate(op=op, column=column, value=value)


def _referenced_columns(predicate: Predicate) -> List[str]:
    if predicate.op == "and":
        out: List[str] = []
        for child in predicate.children:
            out.extend(_referenced_columns(child))
        return out
    return [predicate.column]


def _predicate_key(predicate: Predicate) -> str:
    return json.dumps(predicate.to_dict(), sort_keys=True, separators=(",", ":"))


def canonical_query(raw: Dict[str, Any]) -> str:
    """Whitespace/parentheses-insensitive canonical text.

    Queries with identical semantics but different spacing or AND grouping
    normalize to the same string (flat sorted AND list), hence the same id.
    """
    referenced, predicate = parse_query(raw)
    canonical = {
        "select": sorted(raw["select"]),
        "filter": _canonical_predicate(predicate),
        "referenced": referenced,
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _canonical_predicate(predicate: Predicate) -> Dict[str, Any]:
    if predicate.op == "and":
        flat = _flatten_and(predicate)
        flat = sorted(flat, key=_predicate_key)
        return {"op": "and", "args": [_canonical_predicate(p) for p in flat]}
    return predicate.to_dict()


def _flatten_and(predicate: Predicate) -> List[Predicate]:
    if predicate.op != "and":
        return [predicate]
    out: List[Predicate] = []
    for child in predicate.children:
        out.extend(_flatten_and(child))
    return out


def query_id(raw: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_query(raw).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(predicate: Predicate, row: Dict[str, int]) -> bool:
    if predicate.op == "and":
        return all(evaluate(child, row) for child in predicate.children)
    actual = row[predicate.column]
    expected = predicate.value
    if predicate.op == "eq":
        return actual == expected
    if predicate.op == "ne":
        return actual != expected
    if predicate.op == "gt":
        return actual > expected
    if predicate.op == "ge":
        return actual >= expected
    if predicate.op == "lt":
        return actual < expected
    if predicate.op == "le":
        return actual <= expected
    if predicate.op == "in":
        return actual in expected
    raise QueryError(f"unsupported operator: {predicate.op}", {"operator": predicate.op})


def block_can_match(predicate: Predicate, meta: BlockMeta) -> bool:
    """Min/max push-down: cheap proof that a block cannot match."""
    if predicate.op == "and":
        return all(block_can_match(child, meta) for child in predicate.children)
    if meta.block_id.column != predicate.column:
        return True
    low, high = meta.min_value, meta.max_value
    op = predicate.op
    if op == "eq":
        return low <= predicate.value <= high
    if op == "ne":
        return True  # cannot eliminate on min/max alone
    if op == "gt":
        return high > predicate.value
    if op == "ge":
        return high >= predicate.value
    if op == "lt":
        return low < predicate.value
    if op == "le":
        return low <= predicate.value
    if op == "in":
        return any(low <= v <= high for v in predicate.value)
    raise QueryError(f"unsupported operator: {op}", {"operator": op})


# ---------------------------------------------------------------------------
# Scan plan
# ---------------------------------------------------------------------------

@dataclass
class ScanPlan:
    query_canonical: str
    query_hash: str
    selected_columns: List[str]
    pruned_columns: List[str]
    candidate_blocks: List[BlockId]
    skipped_blocks: List[BlockId]
    reasoning: List[str]
    predicate: Predicate

    def summary(self) -> Dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "query_canonical": self.query_canonical,
            "selected_columns": list(self.selected_columns),
            "pruned_columns": list(self.pruned_columns),
            "candidate_blocks": [b.to_dict() for b in self.candidate_blocks],
            "skipped_blocks": [b.to_dict() for b in self.skipped_blocks],
            "reasoning": list(self.reasoning),
            "predicate": _canonical_predicate(self.predicate),
        }


def build_scan_plan(
    query_raw: Dict[str, Any], segments: Sequence[LoadedSegment]
) -> ScanPlan:
    referenced, predicate = parse_query(query_raw)
    canonical = canonical_query(query_raw)

    all_columns: List[str] = []
    segment_columns: Dict[str, set] = {}
    for segment in segments:
        names = {col.name for col in segment.definition.columns}
        segment_columns[segment.definition.name] = names
        for col in segment.definition.columns:
            if col.name not in all_columns:
                all_columns.append(col.name)
    # A column is queryable if at least one segment provides it; blocks are
    # planned per segment so heterogeneous segment schemas remain valid.
    unknown = [name for name in referenced if name not in all_columns]
    if unknown:
        raise QueryError("query references unknown columns", {"unknown": unknown})

    selected = [name for name in all_columns if name in referenced]
    pruned = [name for name in all_columns if name not in referenced]
    reasoning = [
        f"column pruning keeps {len(selected)} of {len(all_columns)} columns: {selected}",
        f"pruned columns never decoded: {pruned}",
    ]

    candidates: List[BlockId] = []
    skipped: List[BlockId] = []
    filter_columns = sorted(set(_referenced_columns(predicate)))
    reasoning.append(f"predicate columns pushed to block min/max metadata: {filter_columns}")

    for segment in segments:
        segment_names = {col.name for col in segment.definition.columns}
        for meta in segment.blocks:
            if meta.block_id.column not in selected:
                # Pruned column blocks are neither candidates nor "skipped by
                # predicate"; they are excluded by column pruning.
                continue
            if meta.block_id.column not in segment_names:
                continue
            if block_can_match(predicate, meta):
                candidates.append(meta.block_id)
            else:
                skipped.append(meta.block_id)
                reasoning.append(
                    "push-down eliminates block "
                    f"{meta.block_id.column}/rg{meta.block_id.row_group} "
                    f"of segment {meta.block_id.segment}: range "
                    f"[{meta.min_value},{meta.max_value}] cannot satisfy predicate"
                )

    # Stable ordering: segment, row group, then schema column order.
    order = {segment.definition.name: idx for idx, segment in enumerate(segments)}
    column_order = {name: idx for idx, name in enumerate(all_columns)}
    candidates.sort(
        key=lambda b: (order[b.segment], b.row_group, column_order[b.column])
    )
    skipped.sort(key=lambda b: (order[b.segment], b.row_group, column_order[b.column]))
    return ScanPlan(
        query_canonical=canonical,
        query_hash=query_id(query_raw),
        selected_columns=selected,
        pruned_columns=pruned,
        candidate_blocks=candidates,
        skipped_blocks=skipped,
        reasoning=reasoning,
        predicate=predicate,
    )

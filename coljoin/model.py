"""块组、查询与代价权重的内存模型（纯数据容器）。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class HistogramBucket:
    low: float
    high: float
    count: int


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    encoding: str
    histogram: List[HistogramBucket]
    null_count: int
    ndv: int


@dataclass
class BlockGroupSpec:
    bg_id: str
    rows: int
    seed: int
    null_bitmap: List[int]
    columns: List[ColumnSpec]
    value_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def column(self, name: str) -> Optional[ColumnSpec]:
        for col in self.columns:
            if col.name == name:
                return col
        return None


@dataclass(frozen=True)
class JoinPredicate:
    left_bg: str
    left_col: str
    right_bg: str
    right_col: str

    def endpoints(self):
        return {frozenset((self.left_bg, self.right_bg))}


@dataclass(frozen=True)
class FilterPredicate:
    bg_id: str
    column: str
    op: str
    value: Any


@dataclass(frozen=True)
class Aggregate:
    func: str
    bg_id: str
    column: Optional[str]
    alias: str


@dataclass
class QuerySpec:
    query_id: str
    joins: List[JoinPredicate]
    filters: List[FilterPredicate]
    projections: List[Dict[str, str]]
    group_by: List[Dict[str, str]]
    aggregates: List[Aggregate]

    def bg_ids(self):
        ids = set()
        for pred in self.joins:
            ids.add(pred.left_bg)
            ids.add(pred.right_bg)
        for pred in self.filters:
            ids.add(pred.bg_id)
        for ref in self.projections + self.group_by:
            ids.add(ref["bg"])
        for agg in self.aggregates:
            ids.add(agg.bg_id)
        return ids


@dataclass(frozen=True)
class CostWeights:
    scan: int
    compare: int
    build: int
    probe: int
    memory: int
    weight_sum_min: int
    weight_sum_max: int


@dataclass(frozen=True)
class SessionSpec:
    scale: int
    block_granularity: int
    stats_version: str

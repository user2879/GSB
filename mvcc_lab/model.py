"""声明对象的数据模型。

仅承载经过强校验后的对象，不包含任何判定逻辑，保证模型可被主进程与
重放执行器子进程独立加载与复核。
"""

from dataclasses import dataclass, field
from typing import Optional

COMMITTED = "committed"
ROLLED_BACK = "rolledback"
PENDING = "pending"
OUTCOMES = (COMMITTED, ROLLED_BACK, PENDING)


@dataclass(frozen=True)
class Transaction:
    txn_id: str
    begin_seq: int
    outcome: str
    commit_seq: Optional[int]


@dataclass(frozen=True)
class RecordVersion:
    row_key: str
    writer_txn_id: str
    version: int
    value_digest: str
    prev_version: Optional[int]


@dataclass(frozen=True)
class ReadPoint:
    read_point_id: str
    snapshot_seq: int
    reader_txn_id: str


@dataclass(frozen=True)
class VisibilityQuery:
    query_id: str
    read_point_id: str
    row_key: str


@dataclass(frozen=True)
class Spec:
    spec_version: int
    epsilon: float
    global_concurrency: int
    per_readpoint_quota: int
    transactions: tuple[Transaction, ...]
    records: tuple[RecordVersion, ...]
    read_points: tuple[ReadPoint, ...]
    queries: tuple[VisibilityQuery, ...]
    row_chains: dict[str, tuple[RecordVersion, ...]] = field(default_factory=dict)
    max_seq: int = 0
    _txn_index: dict[str, Transaction] = field(default_factory=dict, repr=False)
    _rp_index: dict[str, ReadPoint] = field(default_factory=dict, repr=False)
    _rp_queries: dict[str, tuple[VisibilityQuery, ...]] = field(
        default_factory=dict, repr=False
    )

    def txn(self, txn_id: str) -> Transaction:
        return self._txn_index[txn_id]

    def read_point(self, read_point_id: str) -> ReadPoint:
        return self._rp_index[read_point_id]

    def queries_of(self, read_point_id: str) -> tuple[VisibilityQuery, ...]:
        return self._rp_queries.get(read_point_id, ())

    def __post_init__(self):
        self._txn_index.update({t.txn_id: t for t in self.transactions})
        self._rp_index.update({r.read_point_id: r for r in self.read_points})
        rp_queries: dict[str, list[VisibilityQuery]] = {}
        for query in self.queries:
            rp_queries.setdefault(query.read_point_id, []).append(query)
        self._rp_queries.update({k: tuple(v) for k, v in rp_queries.items()})

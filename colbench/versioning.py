"""参数版本演进、陈旧结论防护与整体替换视图。

三类版本独立单调递增：

* ``budget``：全局并行度、每列解码内存预算、每段并发配额；
* ``encoding``：列/块编码选择或压缩标记；
* ``format``：段文件格式版本。

任何参数修改都会产生新的 :class:`ParameterVersion`；旧版本产生的任务轨迹、
统计与结果摘要会被标记为陈旧（stale），不参与当前展示与汇总。当前视图通过
:meth:`ViewStore.replace` 整体替换，替换结果稳定列出受影响块与清空的轨迹数。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

BUDGET_VERSION = "budget"
ENCODING_VERSION = "encoding"
FORMAT_VERSION = "format"
VERSION_KINDS = (BUDGET_VERSION, ENCODING_VERSION, FORMAT_VERSION)


@dataclass(frozen=True)
class BudgetParams:
    global_parallelism: int
    per_column_memory: int   # 字节；为 0 时调度直接 E_BUDGET
    per_segment_concurrency: int

    def validate(self) -> None:
        for label, value in (("global_parallelism", self.global_parallelism),
                             ("per_column_memory", self.per_column_memory),
                             ("per_segment_concurrency",
                              self.per_segment_concurrency)):
            if not isinstance(value, int) or value < 0:
                from .errors import BudgetError
                raise BudgetError(f"{label} 必须为非负整数",
                                  {"value": value})
        if self.global_parallelism <= 0:
            from .errors import BudgetError
            raise BudgetError("global_parallelism 必须大于 0",
                              {"value": self.global_parallelism})
        if self.per_segment_concurrency <= 0:
            from .errors import BudgetError
            raise BudgetError("per_segment_concurrency 必须大于 0",
                              {"value": self.per_segment_concurrency})
        if self.per_column_memory <= 0:
            from .errors import BudgetError
            raise BudgetError(
                "解码内存预算为零，任何块任务都无法获得内存配额",
                details={"per_column_memory": 0})

    def key(self) -> tuple:
        return (self.global_parallelism, self.per_column_memory,
                self.per_segment_concurrency)


@dataclass(frozen=True)
class ParameterVersion:
    budget: int
    encoding: int
    format: int

    def key(self) -> tuple:
        return (self.budget, self.encoding, self.format)

    def to_dict(self) -> dict:
        return asdict(self)


class VersionRegistry:
    """持久化三类版本号，参数变化时对应版本单调递增。"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.counters = {kind: 1 for kind in VERSION_KINDS}
        self._budget: BudgetParams | None = None
        self._encoding_signature: tuple | None = None
        self._format_signature: int | None = None
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.counters = {kind: int(raw["counters"][kind])
                         for kind in VERSION_KINDS}
        self._budget = BudgetParams(**raw["budget"]) if raw.get("budget") \
            else None
        self._encoding_signature = tuple(raw["encoding_signature"]) \
            if raw.get("encoding_signature") else None
        self._format_signature = raw.get("format_signature")

    def save(self) -> None:
        if not self.path:
            return
        payload = {"counters": self.counters,
                   "budget": asdict(self._budget) if self._budget else None,
                   "encoding_signature": list(self._encoding_signature)
                   if self._encoding_signature else None,
                   "format_signature": self._format_signature}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8")

    def current(self) -> ParameterVersion:
        return ParameterVersion(**self.counters)

    def observe_budget(self, params: BudgetParams) -> ParameterVersion:
        params.validate()
        if self._budget is None or params.key() != self._budget.key():
            self.counters[BUDGET_VERSION] += 1 if self._budget else 0
            self._budget = params
        if self._budget is None:
            self._budget = params
        self.save()
        return self.current()

    def observe_encoding(self, signature: tuple) -> ParameterVersion:
        if self._encoding_signature is not None \
                and signature != self._encoding_signature:
            self.counters[ENCODING_VERSION] += 1
        self._encoding_signature = signature
        self.save()
        return self.current()

    def observe_format(self, format_version: int) -> ParameterVersion:
        if self._format_signature is not None \
                and format_version != self._format_signature:
            self.counters[FORMAT_VERSION] += 1
        self._format_signature = format_version
        self.save()
        return self.current()


@dataclass
class ViewState:
    """当前展示视图：整体替换，绝不增量合并。"""

    version: ParameterVersion
    trajectories: list[dict] = field(default_factory=list)
    result_digests: list[dict] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)
    stale_trajectories: list[dict] = field(default_factory=list)

    def active_block_keys(self) -> list[tuple]:
        return sorted({(item["row_group"], item["column"],
                        item["block_index"]) for item in self.trajectories})


@dataclass(frozen=True)
class ReplacementReport:
    previous_version: ParameterVersion | None
    current_version: ParameterVersion
    cleared_trajectory_count: int
    affected_block_keys: list[tuple]
    stale_count: int

    def to_dict(self) -> dict:
        return {
            "previous_version": self.previous_version.to_dict()
            if self.previous_version else None,
            "current_version": self.current_version.to_dict(),
            "cleared_trajectory_count": self.cleared_trajectory_count,
            "affected_block_keys": [list(key) for key in
                                    self.affected_block_keys],
            "stale_count": self.stale_count,
        }


class ViewStore:
    def __init__(self):
        self._state: ViewState | None = None

    @property
    def state(self) -> ViewState | None:
        return self._state

    def replace(self, new_state: ViewState) -> ReplacementReport:
        previous = self._state
        if previous is None:
            self._state = new_state
            return ReplacementReport(None, new_state.version, 0, [], 0)
        affected = sorted(
            set(previous.active_block_keys())
            | set(new_state.active_block_keys()))
        cleared = len(previous.trajectories)
        stale = list(previous.trajectories)
        report = ReplacementReport(
            previous_version=previous.version,
            current_version=new_state.version,
            cleared_trajectory_count=cleared,
            affected_block_keys=affected,
            stale_count=len(stale))
        self._state = new_state
        self._state.stale_trajectories = stale
        return report

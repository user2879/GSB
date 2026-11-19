"""带预算的诊断视图与不可变快照。

Snapshot 为不可变对象：创建时计算整体内容哈希与版本号，之后只读；
视图只渲染当前快照，整体替换通过版本戳+内容哈希识别，杜绝半更新。

刷新预算：每个瓦片渲染有声明预算（毫秒），超预算则跳帧降级，
记录降级次数并返回 E_BUDGET_EXCEEDED；降级只影响展示密度，
不改变任何测量/对齐状态。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import E_BUDGET_EXCEEDED, E_SNAPSHOT_STALE, BusinessError, make_record
from .graph import node_id
from .hashing import hash_json


@dataclass(frozen=True)
class Tile:
    channel: int
    kind: str                 # pulse | drift | residual | gap | status
    start_sec: float
    end_sec: float
    values: tuple[float, ...]
    mark: str                 # recomputed | reused

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "kind": self.kind,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "values": list(self.values),
            "mark": self.mark,
        }


@dataclass(frozen=True)
class Snapshot:
    version: int
    param_version: str
    tiles: tuple[Tile, ...]
    timeline_hashes: dict[str, str]
    degraded: int
    content_hash: str
    errors: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        body = {
            "version": self.version,
            "param_version": self.param_version,
            "degraded": self.degraded,
            "content_hash": self.content_hash,
            "timeline_hashes": dict(sorted(self.timeline_hashes.items())),
            "tiles": [t.to_dict() for t in self.tiles],
            "errors": [e.to_dict() for e in self.errors],
        }
        return body


class SnapshotStore:
    """只持有不可变快照；替换必须带匹配的版本戳与哈希。"""

    def __init__(self) -> None:
        self._current: Snapshot | None = None
        self._history: list[Snapshot] = []
        self.degrade_count_total = 0

    @property
    def current(self) -> Snapshot | None:
        return self._current

    def replace(self, snap: Snapshot) -> None:
        expect_hash = hash_json({
            "version": snap.version,
            "param_version": snap.param_version,
            "degraded": snap.degraded,
            "tiles": [t.to_dict() for t in snap.tiles],
            "timeline_hashes": dict(sorted(snap.timeline_hashes.items())),
        })
        if expect_hash != snap.content_hash:
            raise BusinessError(E_SNAPSHOT_STALE,
                                "快照内容哈希与载荷不一致，拒绝半更新")
        if self._current is not None and snap.version <= self._current.version:
            raise BusinessError(E_SNAPSHOT_STALE,
                                "新版本快照必须单调递增")
        self._current = snap
        self._history.append(snap)

    def render_guard(self, expected_version: int) -> Snapshot:
        """视图渲染前校验：只允许渲染当前最新版本。"""
        if self._current is None:
            raise BusinessError(E_SNAPSHOT_STALE, "尚无可用快照")
        if expected_version != self._current.version:
            raise BusinessError(
                E_SNAPSHOT_STALE,
                f"视图请求 v{expected_version}，当前为 v{self._current.version}")
        return self._current


def build_tiles(channel: int, graph_states: dict[str, Any],
                estimation: Any, gap_result: Any, dual: Any,
                mark: str) -> tuple[Tile, ...]:
    """从节点产物投影出诊断瓦片（确定性、无随机）。"""
    tiles: list[Tile] = []
    ppm_series = tuple(round(s.ppm, 6) for s in estimation.segments)
    tiles.append(Tile(channel, "drift", 0.0, 8.0, ppm_series, mark))
    rms_series = tuple(round(s.rms_residual, 6) for s in estimation.segments)
    tiles.append(Tile(channel, "residual", 0.0, 8.0, rms_series, mark))
    # 脉冲位置以偏差置信区间瓦片表达
    ci_series = (round(estimation.audio_bias_ci[0], 6),
                 round(estimation.audio_bias_ci[1], 6))
    tiles.append(Tile(channel, "pulse", 0.0, 8.0, ci_series, mark))
    gap_series = tuple(float(r.length) for r in gap_result.records)
    tiles.append(Tile(channel, "gap", 0.0, 8.0, gap_series, mark))
    tiles.append(Tile(channel, "status", 0.0, 8.0,
                      (float(dual.max_ref_diff_sec),), mark))
    return tuple(tiles)


def create_snapshot(version: int,
                    param_version: str,
                    per_channel: dict[int, dict[str, Any]],
                    graph,
                    budget_ms: float,
                    prior_degraded: int,
                    clock: Callable[[], float] = time.perf_counter) -> Snapshot:
    """构造不可变快照；模拟逐瓦片预算，超限跳帧降级。"""
    tiles: list[Tile] = []
    timeline_hashes: dict[str, str] = {}
    degraded = prior_degraded
    errors: list = []
    budget_exceeded = False
    for ch, bundle in per_channel.items():
        mark = "reused" if bundle.get("reused") else "recomputed"
        for tile in build_tiles(ch, graph, bundle["estimation"],
                                bundle["gap"], bundle["dual"], mark):
            t0 = clock()
            # 预算测量：序列化瓦片即代表一次渲染工作量
            _ = hash_json(tile.to_dict())
            elapsed_ms = (clock() - t0) * 1000.0
            if elapsed_ms > budget_ms:
                # 超预算：跳帧（跳过该瓦片展示），仅记录降级，不改状态
                degraded += 1
                budget_exceeded = True
                continue
            tiles.append(tile)
        nid = node_id("map", ch)
        timeline_hashes[nid] = graph.nodes[nid].content_hash or ""
    if budget_exceeded:
        errors.append(make_record(
            E_BUDGET_EXCEEDED,
            f"瓦片渲染超出预算 {budget_ms}ms，已跳帧降级 "
            f"（累计降级 {degraded} 次，测量状态不变）",
            degraded=degraded, budget_ms=budget_ms))
    body = {
        "version": version,
        "param_version": param_version,
        "degraded": degraded,
        "tiles": [t.to_dict() for t in tiles],
        "timeline_hashes": dict(sorted(timeline_hashes.items())),
    }
    content_hash = hash_json(body)
    snap = Snapshot(
        version=version,
        param_version=param_version,
        tiles=tuple(tiles),
        timeline_hashes=timeline_hashes,
        degraded=degraded,
        content_hash=content_hash,
        errors=tuple(errors),
    )
    return snap

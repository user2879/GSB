"""失效传播引擎：并发互斥、逻辑时钟、账本与快照恢复。"""

from __future__ import annotations
import threading
from dataclasses import dataclass
from pathlib import Path

from . import STATE_VERSION
from .errors import EClock, EReplay
from .model import Entry, EntryID, Graph, RoundRecord
from .propagation import (
    expired_entries,
    namespace_seed_entries,
    reachable_closure,
)
from .replay import replay_rounds
from .snapshot import SnapshotData, decode_snapshot_file, write_snapshot
from .store import RestoreReport, Store


@dataclass(frozen=True)
class PropagationResult:
    round_record: RoundRecord | None
    affected: tuple[str, ...]


class Engine:
    """所有公开操作都在同一把 RLock 下串行化，保证条目级每轮互斥。"""

    def __init__(self, graph: Graph, store: Store | None = None,
                 format_version: int = 3) -> None:
        self.graph = graph
        self.store = store or Store()
        self.format_version = format_version
        self._lock = threading.RLock()
        self._clock_ms = 0
        self._round_no = 0
        self.records: list[RoundRecord] = []
        self.alive: set[EntryID] = set(graph.entries)
        self.versions: dict[EntryID, int] = {
            entry_id: 0 for entry_id in graph.entries
        }
        self.store.bootstrap(graph)

    def close(self) -> None:
        with self._lock:
            self.store.close()

    def _advance_clock(self, now_ms: int) -> None:
        if not isinstance(now_ms, int) or isinstance(now_ms, bool):
            raise EClock("逻辑时钟必须是整数毫秒",
                         details={"given": repr(now_ms)})
        if now_ms < self._clock_ms:
            raise EClock(
                f"逻辑时钟倒退: current={self._clock_ms}, given={now_ms}",
                details={
                    "current_clock_ms": self._clock_ms,
                    "given_clock_ms": now_ms,
                },
            )
        self._clock_ms = now_ms

    def explicit_invalidate(self, entry_id: EntryID,
                            now_ms: int) -> PropagationResult:
        with self._lock:
            self._advance_clock(now_ms)
            return self._propagate(
                explicit={entry_id}, expired=set(), refresh_namespaces=[],
                reason="explicit", now_ms=now_ms)

    def reap_expired(self, now_ms: int) -> PropagationResult:
        with self._lock:
            self._advance_clock(now_ms)
            expired = set(expired_entries(self.graph, now_ms, self.alive))
            return self._propagate(
                explicit=set(), expired=expired, refresh_namespaces=[],
                reason="expired", now_ms=now_ms)

    def refresh_namespace(self, namespace: str,
                          now_ms: int) -> PropagationResult:
        with self._lock:
            self._advance_clock(now_ms)
            return self._propagate(
                explicit=set(), expired=set(),
                refresh_namespaces=[namespace],
                reason="namespace_refresh", now_ms=now_ms)

    def submit_batch(self, *, explicit: list[EntryID] | None = None,
                     reap_now_ms: int | None = None,
                     refresh_namespaces: list[str] | None = None,
                     now_ms: int) -> PropagationResult:
        """同一轮提交多类来源；优先级 explicit > expired > refresh。

        同一条目即使同时满足多个来源，也只会在一个触发集合中出现一次，
        因此每轮至多失效一次、注释版本只递增一次。
        """
        with self._lock:
            self._advance_clock(now_ms)
            explicit_set = {entry for entry in (explicit or [])
                            if entry in self.alive}
            expired_set: set[EntryID] = set()
            if reap_now_ms is not None:
                if reap_now_ms > now_ms:
                    raise EClock(
                        "回收时钟不能晚于本轮真值时钟",
                        details={"reap_clock_ms": reap_now_ms,
                                 "now_ms": now_ms})
                expired_set = {
                    entry for entry in expired_entries(
                        self.graph, reap_now_ms, self.alive)
                    if entry not in explicit_set
                }
            refresh_seeds = set(
                namespace_seed_entries(
                    self.graph, list(refresh_namespaces or [])))
            refresh_set = {
                entry for entry in refresh_seeds
                if entry in self.alive
                and entry not in explicit_set and entry not in expired_set
            }
            reason = ("explicit" if explicit_set else
                      "expired" if expired_set else
                      "namespace_refresh" if refresh_set else "noop")
            return self._propagate(
                explicit=explicit_set, expired=expired_set,
                refresh_namespaces=list(refresh_namespaces or []),
                reason=reason, now_ms=now_ms,
                override_refresh_seeds=refresh_set)

    def _propagate(self, *, explicit: set[EntryID],
                   expired: set[EntryID],
                   refresh_namespaces: list[str], reason: str,
                   now_ms: int,
                   override_refresh_seeds: set[EntryID] | None = None
                   ) -> PropagationResult:
        explicit = {entry for entry in explicit if entry in self.alive}
        expired = {entry for entry in expired if entry in self.alive
                   and entry not in explicit}
        if override_refresh_seeds is None:
            refresh_seeds = set(
                namespace_seed_entries(self.graph, refresh_namespaces))
        else:
            refresh_seeds = set(override_refresh_seeds)
        refresh = {
            entry for entry in refresh_seeds
            if entry in self.alive
            and entry not in explicit and entry not in expired
        }
        seeds = explicit | expired | refresh
        ordered = (
            reachable_closure(self.graph, seeds, alive=self.alive)
            if seeds else [])
        affected_ids = [entry_id for entry_id in ordered
                        if entry_id in self.alive]
        if not affected_ids:
            return PropagationResult(None, ())

        self._round_no += 1
        for entry_id in affected_ids:
            self.versions[entry_id] += 1
            self.alive.discard(entry_id)
            self.store.update_entry_state(
                entry_id, self.versions[entry_id], alive=False)

        affected_text = tuple(entry_id.to_str()
                              for entry_id in affected_ids)
        record = RoundRecord(
            round_no=self._round_no,
            clock_ms=now_ms,
            trigger_reason=reason,
            explicit_triggers=tuple(sorted(e.to_str() for e in explicit)),
            expired_triggers=tuple(sorted(e.to_str() for e in expired)),
            refresh_triggers=tuple(sorted(e.to_str() for e in refresh)),
            affected=affected_text,
            annotation_versions={
                entry_id.to_str(): self.versions[entry_id]
                for entry_id in affected_ids
            },
            format_version=self.format_version,
        )
        self.records.append(record)
        self.store.append_round(record)
        return PropagationResult(record, affected_text)

    def save_snapshot(self, path: str | Path,
                      format_version: int | None = None) -> None:
        with self._lock:
            graph = self._snapshot_graph()
            write_snapshot(
                path, graph, self.records,
                format_version or self.format_version,
                state_version=STATE_VERSION)

    def _snapshot_graph(self) -> Graph:
        entries = {
            entry_id: Entry(
                id=entry_id,
                expire_ms=entry.expire_ms,
                digest=entry.digest,
                digest_algorithm=entry.digest_algorithm,
                annotation_version=self.versions[entry_id],
                alive=entry_id in self.alive,
            )
            for entry_id, entry in self.graph.entries.items()
        }
        return Graph(
            namespaces=dict(self.graph.namespaces),
            entries=entries,
            annotation_edges={
                source: set(targets)
                for source, targets in self.graph.annotation_edges.items()
            },
            namespace_parents={
                name: tuple(parents)
                for name, parents in self.graph.namespace_parents.items()
            },
        )

    @classmethod
    def restore(cls, snapshot_path: str | Path,
                store: Store | None = None) -> tuple[
            "Engine", SnapshotData, RestoreReport, EReplay | None]:
        snapshot = decode_snapshot_file(snapshot_path)
        engine = cls(snapshot.graph, store=store,
                     format_version=snapshot.format_version)
        # restore 构造会把初始图 bootstrap 到 SQLite；replace_all 随后整表替换。
        report = engine.store.replace_all(
            snapshot.graph, snapshot.rounds, snapshot.format_version)

        # 重放必须从全量存活、注释版本为 0 的初始状态独立推导，
        # 不能预先套用快照终态，否则会掩盖轨迹与账本的不一致。
        replay_error: EReplay | None = None
        try:
            replay_rounds(engine.graph, snapshot.rounds,
                          snapshot.format_version)
        except EReplay as exc:
            replay_error = exc

        engine.records = list(snapshot.rounds)
        engine._round_no = len(snapshot.rounds)
        engine._clock_ms = max(
            (record.clock_ms for record in snapshot.rounds), default=0)
        for record in snapshot.rounds:
            for entry_text, version in record.annotation_versions.items():
                entry_id = EntryID.parse(entry_text)
                engine.versions[entry_id] = version
                engine.alive.discard(entry_id)
        return engine, snapshot, report, replay_error

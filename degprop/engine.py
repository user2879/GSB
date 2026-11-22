"""诊断组件编排引擎：加载 -> 模拟服务 -> 真实回环采集 -> 判定 -> 传播 -> 账本。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .availability import AvailabilityEvaluator
from .collector import CallScheduler
from .errors import SnapshotError
from .ledger import Ledger, verify_replay
from .loader import load_events, load_topology, topology_digest
from .mock_service import MockCluster
from .propagation import PropagationPlanner

logger = logging.getLogger("degprop.engine")


@dataclass
class BatchReport:
    batch_id: int
    record_count: int
    records: list
    states: dict
    triggers: list
    propagation: object
    digest: str
    replacement: dict = field(default_factory=dict)


class DiagnosticsEngine:
    def __init__(self, topology_path: str | Path, events_path: str | Path,
                 ledger_path: str | Path, *, slow_ms: int = 400) -> None:
        self.topology = load_topology(topology_path)
        self.events = load_events(events_path, self.topology)
        self.digest = topology_digest(self.topology)
        self.ledger_path = Path(ledger_path)
        self.slow_ms = slow_ms
        self.snapshot_error: dict | None = None
        try:
            self.ledger, self.replacement = Ledger.open_or_replace(
                self.ledger_path, self.topology.topology_version, self.digest,
                raise_on_mismatch=True)
        except SnapshotError as exc:
            # 先以 E_SNAPSHOT 报明依据并写结构化日志，再整体替换重开。
            self.snapshot_error = exc.as_dict()
            logger.error("snapshot_replaced",
                         extra={f"biz_{k}": v for k, v in exc.as_dict().items()})
            self.ledger, self.replacement = Ledger.open_or_replace(
                self.ledger_path, self.topology.topology_version, self.digest,
                raise_on_mismatch=False)
        self.cluster: MockCluster | None = None
        self.port_map: dict = {}

    def __enter__(self) -> "DiagnosticsEngine":
        self.cluster = MockCluster(self.topology, slow_ms=self.slow_ms)
        bindings = self.cluster.start()
        self.port_map = {
            sid: {"port": b.port, "refused_port": b.refused_port}
            for sid, b in bindings.items()
        }
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.cluster:
            self.cluster.stop()
        self.ledger.close()

    def run_batch(self, batch_events: list, batch_id: int) -> BatchReport:
        scheduler = CallScheduler(self.topology, self.port_map)
        records = scheduler.run(batch_events)
        logger.info("batch_collected",
                    extra={"batch_id": batch_id, "records": len(records),
                           "max_global_in_flight": scheduler.stats.max_global_in_flight,
                           "max_domain_in_flight": scheduler.stats.max_domain_in_flight,
                           "queued_calls": scheduler.stats.queued_calls})

        evaluator = AvailabilityEvaluator(self.topology)
        evaluator.ingest(records)
        planner = PropagationPlanner(self.topology, evaluator.states,
                                     evaluator.triggers)
        propagation = planner.run()

        # 强制传播命中的上游补记 forced 触发（forced 边无条件传播）；
        # 阈值传播命中的上游本身已有观测触发，不重复补记。
        forced_upstreams = {
            hop.upstream_service for hop in propagation.hops
            if hop.propagated and hop.edge_mode == "forced"
        }
        seeded = {t.service_id for t in evaluator.triggers}
        present = {r.downstream for r in records}
        for service_id in sorted(forced_upstreams - seeded):
            anchor = min((r.seq for r in records if r.downstream in present),
                         default=1)
            evaluator.add_forced_trigger(anchor, service_id)

        last_seq = max(event.seq for event in batch_events)
        digest = self.ledger.append_batch(
            batch_id, records, evaluator.states, evaluator.triggers,
            propagation.hops, propagation.domain_plans, last_seq)
        logger.info("batch_committed",
                    extra={"batch_id": batch_id, "digest": digest,
                           "last_seq": last_seq,
                           "domains": list(propagation.domain_plans)})
        return BatchReport(batch_id, len(records), records, evaluator.states,
                           evaluator.triggers, propagation, digest,
                           dict(self.replacement))

    def resume_and_replay(self) -> dict:
        """恢复后按账本拓扑版本重放：先复核已持久化事件，再处理剩余事件。"""
        cursor_row = self.ledger.cursor()
        last_seq = cursor_row["last_seq"]
        stored = self.ledger.stored_outcomes()
        stored_seqs = [row["seq"] for row in stored]
        replay_events = [event for event in self.events
                         if event.seq in set(stored_seqs)]
        scheduler = CallScheduler(self.topology, self.port_map)
        replay_records = scheduler.run(replay_events)
        # 首个不一致定位到事件序号、服务标识与字段，差异以 E_REPLAY 报明依据。
        verify_replay(stored, replay_records)

        remaining = [event for event in self.events if event.seq > last_seq]
        new_batch_id = cursor_row["last_batch"] + 1
        appended = 0
        if remaining:
            self.run_batch(remaining, new_batch_id)
            appended = len(remaining)
        result = {"last_seq": last_seq, "replayed": len(replay_records),
                  "appended": appended, "remaining": len(remaining),
                  "verified": True}
        logger.info("replay_verified", extra=result)
        return result

"""端到端运行器：确定性引擎裁决 + 真实回环子进程 + SQLite 账本。"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .bucket import BucketState, TokenBucket
from .engine import SchedulerEngine, ShardOutcome
from .errors import DiffError
from .ledger import UNKNOWN, Ledger
from .sample import generate_batch
from .spec import Spec
from .supervisor import Supervisor


class Runner:
    def __init__(self, spec: Spec, ledger_path: str | Path,
                 with_workers: bool = True) -> None:
        self.spec = spec
        self.ledger = Ledger(ledger_path, spec)
        self.engine = SchedulerEngine(spec)
        self.with_workers = with_workers
        self.supervisor: Supervisor | None = None
        self._summaries: dict[str, dict[str, Any]] = {}

    def run(self) -> dict[str, Any]:
        result = self.engine.run()
        if self.with_workers:
            self._run_dispatched_shards(result)
        self._commit_outcomes(result)
        return self._build_report(result)

    def _run_dispatched_shards(self, result) -> None:
        self.supervisor = Supervisor(list(self.spec.nodes),
                                     self.engine.current_scale)
        clients = self.supervisor.start_all()
        self.renew_results: list[dict[str, Any]] = []
        self._dispatch_versions: dict[str, int] = {}
        try:
            for shard_id, outcome in sorted(result.outcomes.items()):
                if outcome.status != "DISPATCHED":
                    continue
                node = self.spec.node(outcome.node_id)
                shard = self.spec.shard(shard_id)
                batch = generate_batch(
                    seed=node.seed, shard_id=shard_id,
                    sample_rows=shard.sample_rows,
                    param_version=outcome.param_version)
                dispatch_version = 1
                receipt = clients[node.node_id].dispatch(
                    shard_id=shard_id,
                    param_version=outcome.param_version, batch=batch,
                    token_id=outcome.token_id,
                    token_version=dispatch_version)
                self._summaries[shard_id] = receipt["summary"]
                outcome.summary = receipt["summary"]
                self._dispatch_versions[shard_id] = dispatch_version
            self._run_renew_rpcs(clients)
        finally:
            self.supervisor.stop_all()
            self.supervisor = None

    def _run_renew_rpcs(self, clients) -> None:
        """对事件流中的 renew 事件做真实节点乐观校验。"""
        token_by_shard = {
            token.shard_id: token for token in self.engine.registry.all_tokens()
        }
        for event in self.spec.events:
            if event.kind != "renew":
                continue
            token = token_by_shard.get(event.shard_id)
            if token is None:
                continue
            client = clients[token.node_id]
            # RPC 携带续约事件声明的乐观版本，节点按下发时已知版本判定。
            node_known = self._dispatch_versions.get(event.shard_id, 1)
            accepted = client.lease_check(token.token_id,
                                          event.expected_version)
            self.renew_results.append({
                "event_seq": event.seq,
                "shard_id": event.shard_id,
                "token_id": token.token_id,
                "expected_version": event.expected_version,
                "node_known_version": node_known,
                "current_token_version": token.token_version,
                "node_accepted": accepted,
                "lease_code": None if accepted else "E_LEASE",
            })

    def _commit_outcomes(self, result) -> None:
        decisions_by_seq: dict[int, list[dict[str, Any]]] = {}
        for decision in result.decisions:
            decisions_by_seq.setdefault(decision["event_seq"], []).append(
                decision)
        refills_by_seq: dict[int, list[dict[str, Any]]] = {}
        for record in result.refill_audit:
            refills_by_seq.setdefault(record.get("event_seq", -1), [])
        for shard_id, outcome in sorted(result.outcomes.items()):
            node_records = [
                record for record in result.refill_audit
                if record["node_id"] == outcome.node_id
            ]
            self.ledger.commit_outcome(
                shard_id=shard_id, node_id=outcome.node_id,
                status=outcome.status,
                param_version=outcome.param_version,
                token_id=outcome.token_id,
                token_version=outcome.token_version,
                summary=outcome.summary, ownership=outcome.ownership,
                bucket=outcome.bucket, cursor_seq=outcome.cursor_seq,
                updated_ms=outcome.updated_ms,
                refill_records=node_records,
                decision_records=decisions_by_seq.get(outcome.cursor_seq, []),
            )

    def _build_report(self, result) -> dict[str, Any]:
        return {
            "outcomes": {
                shard_id: asdict(outcome)
                for shard_id, outcome in sorted(result.outcomes.items())
            },
            "refill_audit": result.refill_audit,
            "decisions": result.decisions,
            "invalidations": result.invalidations,
            "water_levels": result.water_levels,
            "revision_carryovers": result.revision_carryovers,
            "error_bounds": result.error_bounds,
            "dispatch_plans": result.dispatch_plans,
            "tokens": result.tokens,
            "renew_results": getattr(self, "renew_results", []),
            "shard_nodes": result.shard_nodes,
            "current_versions": result.current_versions,
        }


def rebuild_from_scratch(spec: Spec) -> dict[str, dict[str, Any]]:
    """从空白全量重跑引擎（不起子进程），用于增量/全量逐项比对。"""
    engine = SchedulerEngine(spec)
    result = engine.run()
    return {
        shard_id: asdict(outcome)
        for shard_id, outcome in result.outcomes.items()
    }


def verify_incremental_vs_rebuild(spec: Spec,
                                  incremental: dict[str, Any]) -> None:
    rebuilt = rebuild_from_scratch(spec)
    incremental_outcomes = incremental["outcomes"]
    for shard_id in sorted(set(rebuilt) | set(incremental_outcomes)):
        left = rebuilt.get(shard_id)
        right = incremental_outcomes.get(shard_id)
        if left is None or right is None:
            raise DiffError(
                f"shard {shard_id!r} presence differs",
                location=f"shards[{shard_id}]", field="<presence>")
        # 网络聚合摘要由真实子进程产生，纯引擎重建时不参与该字段比对。
        for field_name in ("node_id", "status", "param_version", "token_id",
                           "token_version", "ownership", "bucket",
                           "cursor_seq", "updated_ms"):
            if left[field_name] != right[field_name]:
                raise DiffError(
                    f"incremental vs rebuild mismatch on shard {shard_id!r} "
                    f"field {field_name!r}: {left[field_name]!r} != "
                    f"{right[field_name]!r}",
                    location=f"shards[{shard_id}]", field=field_name)


def recover_after_kill(spec: Spec, ledger_path: str | Path,
                       in_flight_shard: str, in_flight_node: str,
                       last_cursor: int, at_ms: int) -> dict[str, Any]:
    """强制终止恢复：未提交分片标 UNKNOWN 并先回收额度，再重放比对。"""
    ledger = Ledger(ledger_path, spec)
    node = spec.node(in_flight_node)
    bucket_state = BucketState(
        node_id=node.node_id, balance=0, remainder=0, rate=node.rate,
        burst=node.burst, scale=SchedulerEngine(spec).current_scale,
        last_ms=at_ms, param_version=node.param_version,
        bucket_key=node.bucket_key)
    ledger.mark_unknown(in_flight_shard, in_flight_node, last_cursor, at_ms,
                        bucket_state.to_dict())
    replay_outcomes = rebuild_from_scratch(spec)
    ledger.verify_replay(replay_outcomes)
    return {
        "unknown_shards": [in_flight_shard],
        "reclaim_first": [in_flight_shard],
        "replay_verified": True,
        "committed": ledger.load_outcomes(),
    }

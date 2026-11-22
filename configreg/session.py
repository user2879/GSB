"""端到端会话编排：采集、判定、账本事务、快照替换与重放（需求 4-8）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .collect import (
    CollectionOutcome,
    Observation,
    SessionParams,
    build_task_plan,
    run_batch,
    split_item_atomic_batches,
)
from .diff import DiffReport, build_diff_report
from .errors import ReplayError
from .ledger import Ledger, SessionIdentity
from .manifest import Manifest, Registry
from .nodepool import NodePool
from .regression import Budget, RegressionReport, evaluate
from .scheduler import Scheduler, now_ms

PRECISION_PARAM_VERSION = 1


@dataclass
class BatchRecord:
    batch_id: int
    task_keys: list[str]
    outcome: CollectionOutcome
    diff_subset: dict[str, Any]
    regression_subset: dict[str, Any]
    content_digest: str


@dataclass
class SessionResult:
    identity: SessionIdentity
    diff_report: DiffReport
    regression_report: RegressionReport
    observations: dict[tuple[int, str, str], Observation]
    batches: list[BatchRecord] = field(default_factory=list)
    replacement: dict[str, Any] | None = None
    unknown_items: list[str] = field(default_factory=list)
    replay_mismatch: dict[str, Any] | None = None
    snapshot_digest: str = ""


def build_identity(
    baseline: Manifest,
    candidate: Manifest,
    params: SessionParams,
) -> SessionIdentity:
    return SessionIdentity(
        baseline_version=baseline.version,
        candidate_version=candidate.version,
        baseline_digest=baseline.source_digest,
        candidate_digest=candidate.source_digest,
        precision=params.precision,
        precision_param_version=PRECISION_PARAM_VERSION,
        sample_seed=params.sample_seed,
    )


def _items_in_tasks(tasks: list[Any]) -> list[str]:
    return sorted({task.payload["item"] for task in tasks})


def _nodes_in_tasks(tasks: list[Any]) -> list[str]:
    return sorted({task.node.node_id for task in tasks})


def _subset_diff(diff_report: DiffReport, items: list[str]) -> dict[str, Any]:
    full = diff_report.to_dict()
    return {
        "baseline_version": full["baseline_version"],
        "candidate_version": full["candidate_version"],
        "precision": full["precision"],
        "item_diffs": {key: full["item_diffs"][key] for key in items if key in full["item_diffs"]},
    }


def _subset_regression(report: RegressionReport, items: list[str]) -> dict[str, Any]:
    full = report.to_dict()
    return {
        "baseline_version": full["baseline_version"],
        "candidate_version": full["candidate_version"],
        "budget": full["budget"],
        "regression_items": [key for key in full["regression_items"] if key in items],
        "item_verdicts": {
            key: full["item_verdicts"][key] for key in items if key in full["item_verdicts"]
        },
    }


def _observation_to_json(obs: Observation) -> dict[str, Any]:
    return {
        "identity_key": [obs.version, obs.item, obs.node_id],
        "observation": {
            "version": obs.version,
            "item": obs.item,
            "node_id": obs.node_id,
            "summary": obs.summary,
            "elapsed_ms": obs.elapsed_ms,
            "attempts": obs.attempts,
            "retry_count": obs.retry_count,
            "global_wait_ms": obs.global_wait_ms,
            "node_wait_ms": obs.node_wait_ms,
        },
    }


class Session:
    def __init__(
        self,
        baseline: Manifest,
        candidate: Manifest,
        registry: Registry,
        params: SessionParams,
        ledger_path: str | Path,
        log_dir: str | Path | None = None,
    ):
        self.baseline = baseline
        self.candidate = candidate
        self.registry = registry
        self.params = params
        self.budget = Budget(params.max_elapsed_delta_ms, params.max_mismatch_nodes)
        self.ledger = Ledger(ledger_path)
        self.identity = build_identity(baseline, candidate, params)
        self.diff_report = build_diff_report(baseline, candidate, params.precision)
        self.pool = NodePool(registry, params.sample_seed, log_dir)
        self.result = SessionResult(
            identity=self.identity,
            diff_report=self.diff_report,
            regression_report=None,  # type: ignore[arg-type]
            observations={},
        )
        self.crash_after_batches: int | None = None

    def ensure_identity(self) -> dict[str, Any] | None:
        if self.ledger.current_identity() is None:
            self.ledger.initialize_identity(self.identity)
            return None
        ok, _ = self.ledger.identity_matches(self.identity)
        if ok:
            return None
        plan = build_task_plan(self.baseline, self.candidate, self.registry, self.params)
        replacement = self.ledger.replace_all(
            self.identity,
            reason="会话版本/清单摘要/精度口径参数与已持久化状态不一致，执行整体替换",
            affected_items=_items_in_tasks(plan),
            affected_nodes=_nodes_in_tasks(plan),
            now_monotonic_ms=now_ms(),
        )
        self.result.replacement = replacement
        return replacement

    def _make_scheduler(self) -> Scheduler:
        return Scheduler(
            global_concurrency=self.params.global_concurrency,
            per_node_concurrency=self.params.per_node_concurrency,
            socket_timeout_ms=self.params.socket_timeout_ms,
            max_attempts=self.params.max_attempts,
            retry_backoff_ms=self.params.retry_backoff_ms,
        )

    def run(self, progress: Any | None = None) -> SessionResult:
        self.ensure_identity()
        self.pool.start_all()
        try:
            self._run_batches(progress)
            self._finalize()
        finally:
            self.pool.stop_all()
            self.ledger.close()
        return self.result

    def _run_batches(self, progress: Any | None) -> None:
        plan = build_task_plan(self.baseline, self.candidate, self.registry, self.params)
        committed = set(self.ledger.committed_task_keys())
        pending_tasks = [task for task in plan if task.task_key not in committed]
        unknown_items = sorted(
            {
                task.payload["item"]
                for task in plan
                if task.task_key not in committed
                and any(committed_key.endswith(":" + task.payload["node_id"]) for committed_key in committed)
            }
        )
        self.result.unknown_items = unknown_items
        batches = split_item_atomic_batches(pending_tasks, self.params.batch_size)
        scheduler = self._make_scheduler()
        for offset, tasks in enumerate(batches):
            batch_id = self.ledger.next_batch_id()
            outcome = run_batch(scheduler, tasks, progress)
            if outcome.failed_tasks:
                failed = ", ".join(sorted(task.task_key for task in outcome.failed_tasks))
                raise RuntimeError(f"批次 {batch_id} 存在重试耗尽的采集任务: {failed}")
            for key, obs in outcome.observations.items():
                self.result.observations[key] = obs
            items = _items_in_tasks(tasks)
            known_observations = self._all_known_observations()
            regression = evaluate(
                self.baseline,
                self.candidate,
                self.diff_report,
                known_observations,
                self.budget,
                self.params,
            )
            diff_subset = _subset_diff(self.diff_report, items)
            regression_subset = _subset_regression(regression, items)
            obs_json = {key: _observation_to_json(obs) for key, obs in outcome.observations.items()}
            digest = self.ledger.append_batch(
                batch_id=batch_id,
                identity=self.identity,
                task_keys=[task.task_key for task in tasks],
                diff_payload=diff_subset,
                regression_payload=regression_subset,
                observations=obs_json,
                scheduler_records=outcome.scheduler_records,
                committed_at_ms=now_ms(),
            )
            self.result.batches.append(
                BatchRecord(batch_id, [task.task_key for task in tasks], outcome, diff_subset, regression_subset, digest)
            )
            if self.crash_after_batches is not None and len(self.result.batches) >= self.crash_after_batches:
                self.pool.terminate_all()
                raise KeyboardInterrupt(
                    f"模拟强制终止：已完整提交 {len(self.result.batches)} 个批次，"
                    "后续配置项尚未提交，重启后须标记未知并重跑"
                )

    def _all_known_observations(self) -> dict[tuple[int, str, str], Observation]:
        observations = dict(self.result.observations)
        for row in self.ledger.committed_batches():
            payload = self.ledger.load_batch_payload(row["batch_id"])
            for identity_key, raw in payload["observations"].items():
                version, item, node_id = identity_key
                observations[(int(version), str(item), str(node_id))] = Observation(
                    version=int(raw["version"]),
                    item=str(raw["item"]),
                    node_id=str(raw["node_id"]),
                    summary=raw["summary"],
                    elapsed_ms=int(raw["elapsed_ms"]),
                    attempts=int(raw["attempts"]),
                    retry_count=int(raw["retry_count"]),
                    global_wait_ms=int(raw["global_wait_ms"]),
                    node_wait_ms=int(raw["node_wait_ms"]),
                )
        return observations

    def _finalize(self) -> None:
        observations = self._all_known_observations()
        regression = evaluate(
            self.baseline,
            self.candidate,
            self.diff_report,
            observations,
            self.budget,
            self.params,
        )
        self.result.regression_report = regression
        snapshot = {
            "identity": self.identity.to_dict(),
            "diff_digest": self.diff_report.report_digest,
            "regression_digest": regression.report_digest,
            "diff_collection": self.diff_report.to_dict(),
            "regression_collection": regression.to_dict(),
            "affected_items": self.diff_report.changed_keys(),
            "affected_nodes": self.registry.node_ids(),
            "batches": [
                {
                    "batch_id": record.batch_id,
                    "task_keys": record.task_keys,
                    "content_digest": record.content_digest,
                }
                for record in self.result.batches
            ],
        }
        self.result.snapshot_digest = self.ledger.save_snapshot(snapshot)

    def replay_and_verify(self, progress: Any | None = None) -> dict[str, Any]:
        """按账本记录的版本与缩放因子重放剩余/全部采集并逐项核对。"""
        stored_identity = self.ledger.current_identity()
        if stored_identity is None:
            raise ReplayError("账本缺少会话身份记录，无法重放", [])
        if stored_identity != self.identity.to_dict():
            raise ReplayError(
                "重放身份与账本不一致",
                [{"field": "identity", "ledger": stored_identity, "current": self.identity.to_dict()}],
            )
        self.pool.start_all()
        try:
            plan = build_task_plan(self.baseline, self.candidate, self.registry, self.params)
            scheduler = self._make_scheduler()
            live_outcome = run_batch(scheduler, plan, progress)
            if live_outcome.failed_tasks:
                raise ReplayError(
                    "重放采集存在失败任务",
                    [{"task_key": task.task_key, "error": task.error} for task in live_outcome.failed_tasks],
                )
            live_regression = evaluate(
                self.baseline,
                self.candidate,
                self.diff_report,
                live_outcome.observations,
                self.budget,
                self.params,
            )
            mismatch = self._compare_with_ledger(live_regression, live_outcome.observations)
            if mismatch is not None:
                raise ReplayError("重放结果与已提交账本不一致", [mismatch])
            return {
                "replayed": True,
                "tasks": len(plan),
                "diff_digest": self.diff_report.report_digest,
                "regression_digest": live_regression.report_digest,
                "mismatch": None,
            }
        finally:
            self.pool.stop_all()

    def _compare_with_ledger(
        self,
        live: RegressionReport,
        live_observations: dict[tuple[int, str, str], Observation],
    ) -> dict[str, Any] | None:
        live_map = live.to_dict()["item_verdicts"]
        for row in self.ledger.committed_batches():
            payload = self.ledger.load_batch_payload(row["batch_id"])
            for item_key in sorted(payload["item_verdicts"]):
                ledger_verdict = payload["item_verdicts"][item_key]
                live_verdict = live_map.get(item_key)
                if live_verdict is None:
                    return {
                        "item": item_key,
                        "node": None,
                        "field": "item_verdict",
                        "problem": "重放结果缺少该配置项",
                    }
                mismatch = _compare_verdict(
                    item_key,
                    ledger_verdict,
                    live_verdict,
                    strict_timing=self.params.strict_timing,
                )
                if mismatch is not None:
                    return mismatch
            for identity_key, raw_obs in sorted(payload["observations"].items()):
                version, item_key, node_id = identity_key
                live_obs = live_observations.get((int(version), str(item_key), str(node_id)))
                if live_obs is None:
                    return {"item": item_key, "node": node_id, "field": "observation", "problem": "重放缺少观测"}
                if raw_obs["summary"] != live_obs.summary:
                    return {
                        "item": item_key,
                        "node": node_id,
                        "field": "summary",
                        "ledger": raw_obs["summary"],
                        "replay": live_obs.summary,
                    }
                if self.params.strict_timing and raw_obs["elapsed_ms"] != live_obs.elapsed_ms:
                    return {
                        "item": item_key,
                        "node": node_id,
                        "field": "elapsed_ms",
                        "ledger": raw_obs["elapsed_ms"],
                        "replay": live_obs.elapsed_ms,
                    }
        return None


def _compare_verdict(
    item_key: str,
    ledger_verdict: dict[str, Any],
    live_verdict: dict[str, Any],
    strict_timing: bool,
) -> dict[str, Any] | None:
    if ledger_verdict["status"] != live_verdict["status"]:
        return {
            "item": item_key,
            "node": None,
            "field": "status",
            "ledger": ledger_verdict["status"],
            "replay": live_verdict["status"],
        }
    ledger_nodes = {entry["node_id"]: entry for entry in ledger_verdict["node_verdicts"]}
    live_nodes = {entry["node_id"]: entry for entry in live_verdict["node_verdicts"]}
    for node_id in sorted(ledger_nodes):
        left = ledger_nodes[node_id]
        right = live_nodes.get(node_id)
        if right is None:
            return {"item": item_key, "node": node_id, "field": "node_verdict", "problem": "重放缺少节点结论"}
        for field_name in (
            "summary_equal",
            "mismatch_fields",
            "baseline_present",
            "candidate_present",
            "elapsed_breach",
        ):
            if left[field_name] != right[field_name]:
                return {
                    "item": item_key,
                    "node": node_id,
                    "field": field_name,
                    "ledger": left[field_name],
                    "replay": right[field_name],
                }
        if strict_timing and left["elapsed_delta_ms"] != right["elapsed_delta_ms"]:
            return {
                "item": item_key,
                "node": node_id,
                "field": "elapsed_delta_ms",
                "ledger": left["elapsed_delta_ms"],
                "replay": right["elapsed_delta_ms"],
            }
    ledger_attr = ledger_verdict.get("attribution")
    live_attr = live_verdict.get("attribution")
    if (ledger_attr is None) != (live_attr is None):
        return {"item": item_key, "node": None, "field": "attribution.presence", "ledger": ledger_attr, "replay": live_attr}
    if ledger_attr is not None:
        for field_name in ("trigger", "cause", "dominant_layer", "path", "layer_contributions"):
            if ledger_attr[field_name] != live_attr[field_name]:
                return {
                    "item": item_key,
                    "node": None,
                    "field": f"attribution.{field_name}",
                    "ledger": ledger_attr[field_name],
                    "replay": live_attr[field_name],
                }
    return None

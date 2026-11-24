"""归并账本引擎：四阶段流水线、确定性并发、快照复用与双路径视图。"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable

from .attribution import attribute
from .commands import apply_commands
from .errors import E_VIEW_STALE, Failure, sort_failures
from .injection import apply_duplicate_requests, load_injection
from .loader import load_commands, load_merge_rules, load_request_stream, load_worker_config
from .models import LedgerEntry, MergeRule, Receipt, ReuseRecord, Request, Worker
from .receipts import parse_all
from .resources import build_batch_resources, propagate_invalidation
from .snapshot import STAGES, SnapshotStore
from .util import sha256_hex


@dataclass(frozen=True, slots=True)
class PipelineState:
    workers: tuple[Worker, ...]
    requests: tuple[Request, ...]
    rules: tuple[MergeRule, ...]
    receipts: tuple[Receipt, ...]
    batch_size: int
    unknown_versions: frozenset[tuple[str, int]]
    conversion_log: tuple[dict, ...]
    todo_queue: tuple[str, ...]
    config_hash: str
    fingerprints: tuple[dict, ...]


def map_reduce_ordered(items: list, worker: Callable, max_workers: int) -> list:
    """并发执行但按输入顺序回填，结果与 max_workers 无关。"""

    if max_workers <= 1:
        return [worker(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker, items))
    return results


class LedgerEngine:
    def __init__(
        self,
        config_path: str,
        samples_dir: str,
        commands_path: str,
        output_dir: str,
        injection_path: str | None = None,
        max_workers: int = 1,
        clock: Callable[[], float] | None = None,
    ):
        self.config_path = config_path
        self.samples_dir = samples_dir
        self.commands_path = commands_path
        self.store = SnapshotStore(output_dir)
        self.injection_path = injection_path
        self.max_workers = max(1, max_workers)
        self.clock = clock or time.monotonic
        self.failures: list[Failure] = []
        self.stage_failures: dict[str, list[Failure]] = {}
        self.reuse_log: list[ReuseRecord] = []
        self.stage_actions: dict[str, str] = {}
        self.injection_hits: list[str] = []
        self._last_view: dict | None = None

    # ------------------------------------------------------------------ 装载
    def load_initial_state(self) -> PipelineState:
        worker_config, worker_failures = load_worker_config(self.config_path)
        self.failures.extend(worker_failures)
        requests, request_failures = load_request_stream(
            f"{self.samples_dir}/requests.jsonl"
        )
        self.failures.extend(request_failures)
        rules, rule_failures = load_merge_rules(
            f"{self.samples_dir}/merge_rules.json", worker_config["workers"]
        )
        self.failures.extend(rule_failures)
        commands, command_failures = load_commands(self.commands_path)
        self.failures.extend(command_failures)

        injection_index: dict = {}
        unknown_versions: set[tuple[str, int]] = set()
        if self.injection_path:
            plan, injection_failures = load_injection(self.injection_path)
            self.failures.extend(injection_failures)
            injection_index = plan["receipt_index"]
            unknown_versions = set(plan["unknown_versions"])
            requests, duplicate_failures, hits = apply_duplicate_requests(
                requests, plan["duplicate_requests"]
            )
            self.failures.extend(duplicate_failures)
            self.injection_hits.extend(hits)

        receipts, receipt_failures, fingerprints = parse_all(
            self.samples_dir, injection_index
        )
        self.failures.extend(receipt_failures)
        self.injection_hits.extend(
            failure.injection_id
            for failure in receipt_failures
            if failure.injection_id
        )

        state = PipelineState(
            workers=worker_config["workers"],
            requests=requests,
            rules=rules,
            receipts=tuple(receipts),
            batch_size=worker_config["batch_size"],
            unknown_versions=frozenset(unknown_versions),
            conversion_log=(),
            todo_queue=(),
            config_hash="",
            fingerprints=(),
        )
        state, command_failures, _, todo_queue = apply_commands(state, commands)
        self.failures.extend(command_failures)
        state = replace(
            state,
            todo_queue=tuple(todo_queue),
            config_hash=sha256_hex(
                {
                    "workers": [_worker_dict(worker) for worker in state.workers],
                    "batch_size": state.batch_size,
                    "rules": [_rule_dict(rule) for rule in state.rules],
                    "requests": [_request_dict(item) for item in state.requests],
                    "commands_cursor": len(commands),
                }
            ),
            fingerprints=tuple(fingerprints),
        )
        self._commands_count = len(commands)
        return state

    # ------------------------------------------------------------------ 阶段
    def stage_parse(self, state: PipelineState) -> dict:
        self.reuse_log = []
        self.stage_actions = {}
        input_hash = sha256_hex(
            [receipt.as_ledger_dict() for receipt in state.receipts]
        )
        cached = self.store.reusable_stage("parse", input_hash)
        if cached is not None:
            self.stage_actions["parse"] = "reuse"
            return cached
        self.stage_actions["parse"] = "computed"

        def work(receipt: Receipt) -> dict:
            return receipt.as_ledger_dict()

        payload = {
            "stage": "parse",
            "input_hash": input_hash,
            "receipts": map_reduce_ordered(list(state.receipts), work, self.max_workers),
        }
        return self._commit("parse", input_hash, payload, state, reused=False)

    def stage_attribution(self, state: PipelineState, parse_payload: dict) -> dict:
        request_seq = {item.request_id: item.seq for item in state.requests}
        input_hash = sha256_hex(
            {
                "parse_hash": parse_payload["input_hash"],
                "unknown_versions": sorted(state.unknown_versions),
                "request_seq": request_seq,
            }
        )
        cached = self.store.reusable_stage("attribution", input_hash)
        if cached is not None:
            self.stage_actions["attribution"] = "reuse"
            self.stage_failures["attribution"] = [
                self._failure_from_dict(item) for item in cached.get("failures", [])
            ]
            return cached
        self.stage_actions["attribution"] = "computed"

        receipts = list(state.receipts)
        entries, attribution_failures, debug = attribute(
            receipts, set(state.unknown_versions), request_seq
        )
        self.stage_failures["attribution"] = attribution_failures
        deduped = _dedup_entries(entries)
        persisted_failures = [failure.as_dict() for failure in attribution_failures]
        payload = {
            "stage": "attribution",
            "input_hash": input_hash,
            "entries": [entry.as_dict() for entry in deduped],
            "debug": debug,
            "failures": persisted_failures,
        }
        return self._commit("attribution", input_hash, payload, state, reused=False)

    def stage_resources(self, state: PipelineState, attribution_payload: dict) -> dict:
        request_tokens = {item.request_id: item.token_count for item in state.requests}
        input_hash = sha256_hex(
            {
                "attribution_hash": attribution_payload["input_hash"],
                "workers": [_worker_dict(worker) for worker in state.workers],
                "tokens": request_tokens,
            }
        )
        cached = self.store.reusable_stage("resources", input_hash)
        if cached is not None:
            for item in cached.get("invalidation_log", []):
                self._append_reuse(item)
            self.stage_actions["resources"] = "reuse"
            return cached
        self.stage_actions["resources"] = "computed"

        resources = build_batch_resources(
            list(state.receipts),
            state.workers,
            set(state.unknown_versions),
            request_tokens,
        )
        entries = [_entry_from_dict(item) for item in attribution_payload["entries"]]
        entries, invalidation_log = propagate_invalidation(entries, resources)
        for record in invalidation_log:
            self.reuse_log.append(record)
        payload = {
            "stage": "resources",
            "input_hash": input_hash,
            "resources": [item.as_dict() for item in resources],
            "entries": [item.as_dict() for item in entries],
            "invalidation_log": [record.as_dict() for record in invalidation_log],
        }
        return self._commit("resources", input_hash, payload, state, reused=False)

    def stage_summary(
        self,
        state: PipelineState,
        parse_payload: dict,
        attribution_payload: dict,
        resources_payload: dict,
        started_at: float,
        budget_ms: float | None,
    ) -> dict:
        input_hash = sha256_hex(
            {
                "parse": parse_payload["input_hash"],
                "attribution": attribution_payload["input_hash"],
                "resources": resources_payload["input_hash"],
                "todo": list(state.todo_queue),
            }
        )
        cached = self.store.reusable_stage("summary", input_hash)
        if cached is not None:
            invalidation = [
                ReuseRecord(
                    stage=item["stage"],
                    key=item["key"],
                    action=item["action"],
                    reason=item["reason"],
                    propagation_path=tuple(item.get("propagation_path", ())),
                )
                for item in cached.get("reuse_log", [])
                if item["stage"] == "invalidation"
            ]
            self.stage_actions = {
                stage: "reuse" for stage in STAGES
            }
            canonical_log = list(invalidation)
            labels = {
                "parse": "解析阶段",
                "attribution": "归属归一阶段",
                "resources": "资源计算阶段",
                "summary": "账本汇总阶段",
            }
            for stage in STAGES:
                canonical_log.append(
                    ReuseRecord(
                        stage,
                        "all",
                        "reuse",
                        f"{labels[stage]}输入哈希未变化，直接复用持久化产物",
                    )
                )
            self.reuse_log = canonical_log
            return cached
        self.stage_actions["summary"] = "computed"

        entries = [_entry_from_dict(item) for item in resources_payload["entries"]]
        resources = [_resource_from_dict(item) for item in resources_payload["resources"]]
        invalid_count = sum(1 for entry in entries if entry.invalid)
        elapsed_ms = max(0, int((self.clock() - started_at) * 1000))
        stale = budget_ms is not None and elapsed_ms > budget_ms
        stage_labels = {
            "parse": "解析阶段",
            "attribution": "归属归一阶段",
            "resources": "资源计算阶段",
            "summary": "账本汇总阶段",
        }
        canonical_log = list(self.reuse_log)
        for stage in STAGES:
            action = self.stage_actions.get(stage, "computed")
            reason = (
                f"{stage_labels[stage]}输入哈希未变化，直接复用持久化产物"
                if action == "reuse"
                else f"{stage_labels[stage]}本轮重新计算并整体替换提交"
            )
            canonical_log.append(ReuseRecord(stage, "all", action, reason))
        self.reuse_log = canonical_log

        payload = {
            "stage": "summary",
            "input_hash": input_hash,
            "stale": stale,
            "totals": {
                "request_count": len(state.requests),
                "receipt_count": len(state.receipts),
                "ledger_entry_count": len(entries),
                "invalid_entry_count": invalid_count,
                "invalid_batch_count": sum(1 for item in resources if item.invalid),
                "failure_count": len(self.all_failures()),
            },
            "attribution_summary": [
                {
                    "request_id": entry.request_id,
                    "batch_id": entry.batch_id,
                    "worker_id": entry.worker_id,
                    "param_version": entry.param_version,
                    "invalid": entry.invalid,
                    "receipt_ref": entry.receipt_ref,
                    "basis": entry.basis,
                }
                for entry in entries
            ],
            "todo_queue": list(state.todo_queue),
            "reuse_log": [record.as_dict() for record in self.reuse_log],
        }
        committed = self._commit("summary", input_hash, payload, state, reused=False)
        committed["elapsed_ms"] = elapsed_ms
        return committed

    def _commit(self, stage: str, input_hash: str, payload: dict, state: PipelineState, reused: bool) -> dict:
        self.store.commit_stage(
            stage,
            input_hash,
            payload,
            reused,
            state.config_hash,
            list(state.fingerprints),
            self._commands_count,
        )
        return payload

    # ------------------------------------------------------------------ 运行
    def run_authoritative(self, budget_ms: float | None = None) -> dict:
        """固定裁决顺序跑完全部四个阶段，产出权威账本报告。"""

        started_at = self.clock()
        state = self.load_initial_state()
        parse_payload = self.stage_parse(state)
        attribution_payload = self.stage_attribution(state, parse_payload)
        resources_payload = self.stage_resources(state, attribution_payload)
        summary_payload = self.stage_summary(
            state,
            parse_payload,
            attribution_payload,
            resources_payload,
            started_at,
            budget_ms,
        )
        report = self._build_report(
            state,
            parse_payload,
            attribution_payload,
            resources_payload,
            summary_payload,
            mode="authoritative",
        )
        self._last_view = report
        return report

    def fast_view(self, budget_ms: float) -> dict:
        """快速路径：预算内先返回已完成阶段结论，超预算回退最近结论并标 stale。"""

        started_at = self.clock()
        state = self.load_initial_state()
        parse_payload = self.stage_parse(state)
        if self.max_workers >= 1:
            self._tick()
        if self._elapsed(started_at) > budget_ms:
            return self._stale_fallback(budget_ms, state, progress=1, completed=STAGES[:1])

        attribution_payload = self.stage_attribution(state, parse_payload)
        self._tick()
        if self._elapsed(started_at) > budget_ms:
            return self._stale_fallback(budget_ms, state, progress=2, completed=STAGES[:2])

        resources_payload = self.stage_resources(state, attribution_payload)
        self._tick()
        if self._elapsed(started_at) > budget_ms:
            return self._stale_fallback(budget_ms, state, progress=3, completed=STAGES[:3])

        summary_payload = self.stage_summary(
            state, parse_payload, attribution_payload, resources_payload, started_at, budget_ms
        )
        report = self._build_report(
            state,
            parse_payload,
            attribution_payload,
            resources_payload,
            summary_payload,
            mode="fast",
        )
        if summary_payload.get("stale"):
            report["view_flags"] = [E_VIEW_STALE]
            if self._last_view is not None:
                report = self._last_view
                report["view_flags"] = [E_VIEW_STALE]
        self._last_view = report
        return report

    def _stale_fallback(
        self, budget_ms: float, state: PipelineState, progress: int, completed: list[str]
    ) -> dict:
        if self._last_view is not None:
            report = dict(self._last_view)
        else:
            report = {
                "mode": "fast",
                "view_flags": [E_VIEW_STALE],
                "budget_ms": budget_ms,
                "progress": {"completed_stages": completed, "stage_index": progress},
                "attribution_summary": [],
                "totals": {"invalid_entry_count": 0},
            }
        report["view_flags"] = report.get("view_flags", []) + [E_VIEW_STALE]
        report["progress"] = {
            "completed_stages": completed,
            "stage_index": progress,
            "total_stages": len(STAGES),
        }
        return report

    def _elapsed(self, started_at: float) -> float:
        return (self.clock() - started_at) * 1000.0

    def _tick(self) -> None:
        """测试用假时钟由调用方推进；真实 monotonic 时钟自然流逝，无需操作。"""

        if hasattr(self.clock, "now"):
            self.clock.now += 1.0

    def _failure_from_dict(self, item: dict) -> Failure:
        return Failure(
            code=item["code"],
            source=item["source"],
            line=item["line"],
            message=item["message"],
            injection_id=item.get("injection_id"),
        )

    def _append_reuse(self, item: dict) -> None:
        self.reuse_log.append(
            ReuseRecord(
                stage=item["stage"],
                key=item["key"],
                action=item["action"],
                reason=item["reason"],
                propagation_path=tuple(item.get("propagation_path", ())),
            )
        )

    def all_failures(self) -> list[Failure]:
        combined = list(self.failures)
        for stage in STAGES:
            combined.extend(self.stage_failures.get(stage, []))
        return sort_failures(combined)

    def _build_report(
        self,
        state: PipelineState,
        parse_payload: dict,
        attribution_payload: dict,
        resources_payload: dict,
        summary_payload: dict,
        mode: str,
    ) -> dict:
        failures = self.all_failures()
        entries = [_entry_from_dict(item) for item in resources_payload["entries"]]
        if self.injection_path and any(
            "未知参数版本" in (item.get("invalid_reason") or "")
            for item in resources_payload["resources"]
        ):
            plan, _ = load_injection(self.injection_path)
            self.injection_hits.extend(
                entry["id"]
                for entry in plan["entries"]
                if entry["type"] == "unknown_param_version"
            )
        report = {
            "mode": mode,
            "global_version": self.store.manifest["global_version"],
            "command_cursor": self.store.manifest["command_cursor"],
            "config_hash": state.config_hash,
            "receipt_fingerprints": list(state.fingerprints),
            "snapshot_manifest": self.store.manifest,
            "totals": summary_payload["totals"],
            "attribution_summary": summary_payload["attribution_summary"],
            "ledger": [entry.as_dict() for entry in entries],
            "resources": resources_payload["resources"],
            "conversion_log": list(state.conversion_log),
            "reuse_log": [record.as_dict() for record in self.reuse_log],
            "todo_queue": list(state.todo_queue),
            "failures": [failure.as_dict() for failure in failures],
            "injection": {
                "enabled": self.injection_path is not None,
                "hit_ids": sorted(set(self.injection_hits)),
            },
            "stage_debug": {
                "parse_hash": parse_payload["input_hash"],
                "attribution_hash": attribution_payload["input_hash"],
                "resource_hash": resources_payload["input_hash"],
                "summary_hash": summary_payload["input_hash"],
            },
            "view_flags": [],
        }
        return report


def _dedup_entries(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    best: dict[str, LedgerEntry] = {}
    for entry in entries:
        current = best.get(entry.request_id)
        if current is None or entry.receipt_ref < current.receipt_ref:
            best[entry.request_id] = entry
    return sorted(best.values(), key=lambda entry: (entry.seq, entry.request_id))


def _worker_dict(worker: Worker) -> dict:
    return {
        "worker_id": worker.worker_id,
        "port": worker.port,
        "model_dim": worker.model_dim,
        "hidden_dim": worker.hidden_dim,
        "param_version": worker.param_version,
        "quant": worker.quant,
    }


def _rule_dict(rule: MergeRule) -> dict:
    return {
        "rule_id": rule.rule_id,
        "worker_ids": list(rule.worker_ids),
        "batch_size": rule.batch_size,
    }


def _request_dict(request: Request) -> dict:
    return {
        "seq": request.seq,
        "request_id": request.request_id,
        "arrival_ms": request.arrival_ms,
        "token_count": request.token_count,
    }


def _entry_from_dict(item: dict) -> LedgerEntry:
    return LedgerEntry(
        request_id=item["request_id"],
        seq=item["seq"],
        batch_id=item["batch_id"],
        worker_id=item["worker_id"],
        param_version=item["param_version"],
        source_version=item["source_version"],
        duration_ms=item["duration_ms"],
        output_digest=item["output_digest"],
        rank_in_batch=item["rank_in_batch"],
        integrity_tag=item["integrity_tag"],
        basis=item["basis"],
        candidate_count=item["candidate_count"],
        superseded_refs=tuple(item["superseded_refs"]),
        receipt_ref=item["receipt_ref"],
        invalid=item["invalid"],
        invalid_reason=item["invalid_reason"],
        propagation_path=tuple(item["propagation_path"]),
    )


def _resource_from_dict(item: dict) -> "object":
    from .models import BatchResource

    return BatchResource(
        batch_id=item["batch_id"],
        worker_id=item["worker_id"],
        param_version=item["param_version"],
        request_count=item["request_count"],
        total_duration_ms=item["total_duration_ms"],
        max_duration_ms=item["max_duration_ms"],
        macs_fixed=item["macs_fixed"],
        utilization_fixed=item["utilization_fixed"],
        macs_per_ms_fixed=item["macs_per_ms_fixed"],
        quant=item["quant"],
        invalid=item["invalid"],
        invalid_reason=item["invalid_reason"],
        propagation_path=tuple(item["propagation_path"]),
    )

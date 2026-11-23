"""Incremental, content-addressed execution engine with deterministic scheduling."""

from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from .cache_keys import load_artifact, stage_keys
from .config import WorkbenchConfig
from .encoding import encode_pair
from .error_tracking import track_errors
from .hashing import sha256_json
from .models import Issue, Sample
from .quantization import score_fixed_path
from .snapshot import LoadedSnapshot, SnapshotStore


STAGES = ("loading", "position_encoding", "scoring", "error_tracking")


@dataclass
class SamplePlan:
    sample: Sample
    keys: dict[str, Any]
    needed: tuple[str, ...]
    complete: bool
    reused_stages: tuple[str, ...]
    recompute_stages: tuple[str, ...]
    cached_pe: dict[str, Any] | None = None
    cached_score: dict[str, Any] | None = None
    cached_error: dict[str, Any] | None = None


@dataclass
class RunResult:
    resume_state: str
    config_hash: str
    completed_samples: list[str]
    pending_samples: list[str]
    pending_issues: list[Issue]
    consumed_points: int
    point_budget: int
    point_budget_remaining: int
    sample_quota: int
    resume_cursor: str | None
    reused: list[dict[str, Any]]
    recomputed: list[dict[str, Any]]
    scheduler: dict[str, Any]
    snapshot_manifest: dict[str, Any]
    snapshot_id: str
    config: WorkbenchConfig = field(repr=False)


class WorkbenchEngine:
    def __init__(
        self,
        config: WorkbenchConfig,
        samples: list[Sample],
        state_dir: str = "state",
        *,
        sample_content_hash: str | None = None,
    ):
        self.config = config
        self.samples = sorted(samples, key=lambda sample: sample.sample_id)
        self.samples_by_id = {sample.sample_id: sample for sample in self.samples}
        self.sample_content_hash = sample_content_hash or sha256_json(
            [sample.to_canonical_dict() for sample in self.samples]
        )
        self.store = SnapshotStore(state_dir)
        self.loaded_snapshot: LoadedSnapshot = self.store.load_current(
            self._config_hash(self.config)
        )
        self.caches: dict[str, dict[str, dict[str, Any]]] = {
            stage: dict(self.loaded_snapshot.stages.get(stage, {})) for stage in STAGES
        }
        self._last_manifest: dict[str, Any] = dict(self.loaded_snapshot.manifest)
        self._durable_consumed_budget = int(
            self.loaded_snapshot.manifest.get("consumed_budget", 0) or 0
        )
        self._sensitivity_cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._last_run: RunResult | None = None

    @staticmethod
    def _config_hash(config: WorkbenchConfig) -> str:
        return sha256_json({"schema": "workbench-config-v1", "config": config.to_dict()})

    @property
    def config_hash(self) -> str:
        return self._config_hash(self.config)

    @property
    def resume_state(self) -> str:
        if self.loaded_snapshot.state == "cold":
            return "cold"
        if self.loaded_snapshot.state == "rebuild":
            return "rebuild"
        expected = self.loaded_snapshot.manifest.get("config_hash")
        return "exact_resume" if expected == self.config_hash else "incremental"

    def set_config(self, config: WorkbenchConfig) -> None:
        self.config = config
        self._sensitivity_cache.clear()

    def _normalize_cached_error(
        self, cached: dict[str, Any], config: WorkbenchConfig
    ) -> dict[str, Any]:
        if cached.get("train_context_length") == config.train_context_length:
            return cached
        normalized = copy.deepcopy(cached)
        normalized["train_context_length"] = config.train_context_length
        return normalized

    def _build_plan(self, sample: Sample) -> SamplePlan:
        keys = stage_keys(sample, self.config)
        cached_pe = self.caches["position_encoding"].get(keys["position_encoding"])
        cached_score = self.caches["scoring"].get(keys["scoring"])
        cached_error = self.caches["error_tracking"].get(keys["error_tracking"])
        if cached_error is not None:
            normalized_error = self._normalize_cached_error(cached_error, self.config)
            if normalized_error is not cached_error:
                self.caches["error_tracking"][keys["error_tracking"]] = normalized_error
            cached_error = normalized_error
        availability = {
            "position_encoding": cached_pe is not None,
            "scoring": cached_score is not None,
            "error_tracking": cached_error is not None,
        }
        recompute = tuple(stage for stage in STAGES[1:] if not availability[stage])
        reused = tuple(stage for stage in STAGES[1:] if availability[stage])
        complete = not recompute
        return SamplePlan(
            sample=sample,
            keys=keys,
            needed=recompute,
            complete=complete,
            reused_stages=reused,
            recompute_stages=recompute,
            cached_pe=cached_pe,
            cached_score=cached_score,
            cached_error=cached_error,
        )

    def _stage_reference(
        self, sample: Sample, stage: str, cache_key: str
    ) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "scheme": self.config.scheme,
            "stage": stage,
            "cache_key": cache_key,
        }

    def _compute_sample(self, plan: SamplePlan) -> dict[str, Any]:
        sample = plan.sample
        keys = plan.keys
        position_artifact = plan.cached_pe
        score_artifact = plan.cached_score
        error_artifact = plan.cached_error
        computed: list[str] = []

        if position_artifact is None:
            position_artifact = encode_pair(
                sample.sample_id,
                sample.query_position,
                sample.key_positions,
                sample.query_vector,
                sample.key_vectors,
                self.config,
            )
            position_artifact.update(
                {
                    "stage": "position_encoding",
                    "cache_key": keys["position_encoding"],
                    "parameters": keys["pe_parameters"],
                    "sort_key": f"{sample.sample_id}|position_encoding|{self.config.scheme}",
                }
            )
            computed.append("position_encoding")

        if score_artifact is None:
            score_artifact = score_fixed_path(
                position_artifact["query"],
                position_artifact["keys"],
                sample.query_position,
                sample.key_positions,
                dimension=self.config.dimension,
                dimension_block_size=self.config.dimension_block_size,
                quantization_level=self.config.quantization_level,
                fixed_decimal_bits=self.config.fixed_decimal_bits,
            )
            score_artifact.update(
                {
                    "stage": "scoring",
                    "cache_key": keys["scoring"],
                    "sample_id": sample.sample_id,
                    "scheme": self.config.scheme,
                    "scheme_display": self.config.scheme_display,
                    "precision_level": self.config.precision_level,
                    "position_encoding_key": keys["position_encoding"],
                    "quantization_level": self.config.quantization_level,
                    "fixed_decimal_bits": self.config.fixed_decimal_bits,
                    "dimension": self.config.dimension,
                    "sort_key": f"{sample.sample_id}|scoring|{self.config.scheme}",
                }
            )
            computed.append("scoring")

        if error_artifact is None:
            error_artifact = track_errors(
                sample,
                position_artifact,
                score_artifact,
                self.config,
                cache_key=keys["error_tracking"],
            )
            computed.append("error_tracking")

        assert position_artifact is not None
        assert score_artifact is not None
        assert error_artifact is not None
        return {
            "sample_id": sample.sample_id,
            "position_encoding": position_artifact,
            "scoring": score_artifact,
            "error_tracking": error_artifact,
            "computed": computed,
        }

    def _active_entries(
        self, plans: dict[str, SamplePlan]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        wanted: dict[str, set[str]] = {stage: set() for stage in STAGES}
        moment = {
            "loading": lambda plan: plan.keys["load"],
            "position_encoding": lambda plan: plan.keys["position_encoding"],
            "scoring": lambda plan: plan.keys["scoring"],
            "error_tracking": lambda plan: plan.keys["error_tracking"],
        }
        for plan in plans.values():
            for stage in STAGES:
                wanted[stage].add(moment[stage](plan))
        return {
            stage: {
                key: value
                for key, value in self.caches[stage].items()
                if key in wanted[stage]
            }
            for stage in STAGES
        }

    def _completed_ids(self, plans: dict[str, SamplePlan]) -> list[str]:
        completed: list[str] = []
        for sample_id, plan in plans.items():
            if all(
                plan.keys[stage] in self.caches[stage]
                for stage in ("position_encoding", "scoring", "error_tracking")
            ):
                completed.append(sample_id)
        return sorted(completed)

    def _pending_issue(self, sample_id: str, detail: str) -> Issue:
        return Issue(
            code="E_POINT_BUDGET_PENDING",
            sample_id=sample_id,
            field="point_budget",
            message=detail,
            severity="pending",
        )

    def run(
        self,
        *,
        point_budget: int | None = None,
        sample_quota: int | None = None,
        snapshot_interval: int | None = None,
        command_cursor: int | None = None,
    ) -> RunResult:
        budget = int(point_budget if point_budget is not None else self.config.point_budget)
        quota = int(sample_quota if sample_quota is not None else self.config.sample_quota)
        interval = int(
            snapshot_interval
            if snapshot_interval is not None
            else self.config.snapshot_interval
        )
        if budget <= 0 or quota <= 0 or interval <= 0:
            raise ValueError("point_budget, sample_quota, and snapshot_interval must be positive")

        plans = {sample.sample_id: self._build_plan(sample) for sample in self.samples}
        reused: list[dict[str, Any]] = []
        recomputed: list[dict[str, Any]] = []
        for sample in self.samples:
            plan = plans[sample.sample_id]
            for stage in plan.reused_stages:
                reused.append(
                    self._stage_reference(sample, stage, plan.keys[stage])
                )

        remaining = budget
        allocated: list[SamplePlan] = []
        pending_details: dict[str, str] = {}
        budget_exhausted = False
        for sample in self.samples:
            plan = plans[sample.sample_id]
            if plan.complete:
                continue
            cost = sample.point_cost
            if budget_exhausted:
                pending_details[sample.sample_id] = "global point budget exhausted"
                continue
            if cost > quota:
                pending_details[sample.sample_id] = (
                    f"sample point cost {cost} exceeds per-sample quota {quota}"
                )
                continue
            if cost > remaining:
                budget_exhausted = True
                pending_details[sample.sample_id] = (
                    f"global point budget exhausted; {remaining} point(s) remain"
                )
                continue
            remaining -= cost
            allocated.append(plan)

        scheduler = {
            "policy": "stable_sample_id_order",
            "max_workers_limit": self.config.thread_pool_size,
            "max_workers_observed": 0,
            "tasks_submitted": len(allocated),
            "tasks_reused_complete": sum(1 for plan in plans.values() if plan.complete),
            "point_budget": budget,
            "points_reserved": budget - remaining,
            "sample_quota": quota,
            "snapshot_interval": interval,
        }
        run_start_durable = self._durable_consumed_budget
        run_consumed_total = 0
        checkpoints: list[str] = []
        observed = 0
        active_workers = 0

        def wrapped(plan: SamplePlan) -> dict[str, Any]:
            nonlocal observed, active_workers
            with self._lock:
                active_workers += 1
                observed = max(observed, active_workers)
            try:
                return self._compute_sample(plan)
            finally:
                with self._lock:
                    active_workers -= 1

        executor = ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size,
            thread_name_prefix="pe-diag",
        )
        try:
            for chunk_start in range(0, len(allocated), interval):
                chunk = allocated[chunk_start : chunk_start + interval]
                futures = {executor.submit(wrapped, plan): plan for plan in chunk}
                outcomes: list[tuple[SamplePlan, dict[str, Any]]] = []
                for future in as_completed(futures):
                    plan = futures[future]
                    outcomes.append((plan, future.result()))
                outcomes.sort(key=lambda item: item[0].sample.sample_id)
                for plan, outcome in outcomes:
                    sample = plan.sample
                    for stage in ("position_encoding", "scoring", "error_tracking"):
                        artifact = outcome[stage]
                        self.caches[stage][plan.keys[stage]] = artifact
                    for stage in outcome["computed"]:
                        recomputed.append(
                            self._stage_reference(sample, stage, plan.keys[stage])
                        )
                run_consumed_total += sum(plan.sample.point_cost for plan, _ in outcomes)
                completed_now = self._completed_ids(plans)
                pending_now = sorted(
                    sample_id
                    for sample_id in plans
                    if sample_id not in set(completed_now)
                )
                cursor = pending_now[0] if pending_now else None
                if chunk_start + interval < len(allocated):
                    checkpoint = self._commit_snapshot(
                        plans=plans,
                        completed=completed_now,
                        pending=pending_now,
                        resume_cursor=cursor,
                        command_cursor=command_cursor,
                        consumed_budget_total=run_start_durable + run_consumed_total,
                        last_run_consumed_points=run_consumed_total,
                    )
                    checkpoints.append(checkpoint["snapshot_id"])
        finally:
            executor.shutdown(wait=True)

        scheduler["max_workers_observed"] = observed
        completed = self._completed_ids(plans)
        completed_set = set(completed)
        pending = sorted(sample_id for sample_id in plans if sample_id not in completed_set)
        pending_issues: list[Issue] = []
        for sample_id in pending:
            detail = pending_details.get(sample_id, "sample has incomplete stage artifacts")
            pending_issues.append(self._pending_issue(sample_id, detail))
        cursor = pending[0] if pending else None
        checkpoint = self._commit_snapshot(
            plans=plans,
            completed=completed,
            pending=pending,
            resume_cursor=cursor,
            command_cursor=command_cursor,
            consumed_budget_total=run_start_durable + run_consumed_total,
            last_run_consumed_points=run_consumed_total,
        )
        self._durable_consumed_budget = int(checkpoint["manifest"]["consumed_budget"])
        checkpoints.append(checkpoint["snapshot_id"])
        scheduler["checkpoints"] = checkpoints
        scheduler["points_consumed_this_run"] = run_consumed_total
        reused.sort(key=lambda item: (item["sample_id"], item["stage"]))
        recomputed.sort(key=lambda item: (item["sample_id"], item["stage"]))
        result = RunResult(
            resume_state=self.resume_state,
            config_hash=self.config_hash,
            completed_samples=completed,
            pending_samples=pending,
            pending_issues=pending_issues,
            consumed_points=run_consumed_total,
            point_budget=budget,
            point_budget_remaining=remaining,
            sample_quota=quota,
            resume_cursor=cursor,
            reused=reused,
            recomputed=recomputed,
            scheduler=scheduler,
            snapshot_manifest=checkpoint["manifest"],
            snapshot_id=checkpoint["snapshot_id"],
            config=self.config,
        )
        self._last_run = result
        trace = {
            "resume_state": result.resume_state,
            "config_hash": result.config_hash,
            "sample_content_hash": self.sample_content_hash,
            "snapshot_id": result.snapshot_id,
            "snapshot_version": result.snapshot_manifest.get("snapshot_version"),
            "completed_samples": result.completed_samples,
            "pending_samples": result.pending_samples,
            "resume_cursor": result.resume_cursor,
            "scheduler": result.scheduler,
            "reused": result.reused,
            "recomputed": result.recomputed,
            "pending_issues": [issue.to_dict() for issue in result.pending_issues],
        }
        self.store.write_trace(trace)
        return result

    def _commit_snapshot(
        self,
        *,
        plans: dict[str, SamplePlan],
        completed: list[str],
        pending: list[str],
        resume_cursor: str | None,
        command_cursor: int | None,
        consumed_budget_total: int,
        last_run_consumed_points: int,
    ) -> dict[str, Any]:
        stages = self._active_entries(plans)
        self.caches["loading"] = {
            key: value for key, value in self.caches["loading"].items() if key in stages.get("loading", {})
        } if "loading" in stages else self.caches["loading"]
        for sample in self.samples:
            key = plans[sample.sample_id].keys["load"]
            if key not in self.caches["loading"]:
                self.caches["loading"][key] = load_artifact(sample, key)
            loading_entry = dict(self.caches["loading"][key])
            loading_entry["scheme"] = self.config.scheme
            self.caches["loading"][key] = loading_entry
        stages = self._active_entries(plans)
        schemes = [self.config.scheme] if completed else []
        previous = LoadedSnapshot(
            state=self.loaded_snapshot.state,
            snapshot_id=self.loaded_snapshot.snapshot_id,
            manifest=self._last_manifest,
            stages={},
        )
        committed = self.store.commit(
            config=self.config.to_dict(),
            config_hash=self.config_hash,
            stages=stages,
            completed_schemes=schemes,
            completed_samples=completed,
            consumed_budget_total=consumed_budget_total,
            last_run_consumed_points=last_run_consumed_points,
            pending_samples=pending,
            resume_cursor=resume_cursor,
            command_cursor=command_cursor,
            sample_content_hash=self.sample_content_hash,
            previous_snapshot=previous,
            extra_manifest={
                "config_scheme": self.config.scheme,
                "config_precision_level": self.config.precision_level,
                "config_quantization_level": self.config.quantization_level,
                "config_fixed_decimal_bits": self.config.fixed_decimal_bits,
            },
        )
        self._last_manifest = dict(committed["manifest"])
        return committed

    def active_stage_keys(self) -> dict[str, set[str]]:
        plans = {sample.sample_id: self._build_plan(sample) for sample in self.samples}
        return {
            stage: {
                plan.keys["load" if stage == "loading" else stage]
                for plan in plans.values()
            }
            for stage in STAGES
        }

    def diagnostics_for_completed(self, completed_ids: list[str]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for sample_id in sorted(completed_ids):
            sample = self.samples_by_id[sample_id]
            keys = stage_keys(sample, self.config)
            pe = self.caches["position_encoding"].get(keys["position_encoding"])
            score = self.caches["scoring"].get(keys["scoring"])
            error = self.caches["error_tracking"].get(keys["error_tracking"])
            if pe is None or score is None or error is None:
                continue
            records.append(
                {
                    "sample_id": sample_id,
                    "query_position": sample.query_position,
                    "key_positions": list(sample.key_positions),
                    "point_cost": sample.point_cost,
                    "scheme": self.config.scheme,
                    "precision_level": self.config.precision_level,
                    "quantization_level": self.config.quantization_level,
                    "fixed_decimal_bits": self.config.fixed_decimal_bits,
                    "position": pe["summary"],
                    "scores": [
                        {
                            "index": key_record["index"],
                            "key_position": key_record["key_position"],
                            "relative_distance": key_record["relative_distance"],
                            "raw_dot": key_record["raw_dot"],
                            "score": key_record["score"],
                            "weight": key_record["weight"],
                            "accumulation_blocks": len(key_record["accumulation"]),
                        }
                        for key_record in score["keys"]
                    ],
                    "error": error,
                    "first_exceeded_stage": error["first_exceeded_stage"],
                    "first_exceeded_value": error["first_exceeded_value"],
                    "stage_errors": error["stage_errors"],
                    "attention_overtolerance": error["attention_overtolerance"],
                }
            )
        return records

    @property
    def last_run(self) -> RunResult | None:
        return self._last_run

    @property
    def cached_sensitivity(self) -> dict[str, dict[str, Any]]:
        return self._sensitivity_cache

    def cache_sensitivity(self, key: str, value: dict[str, Any]) -> None:
        self._sensitivity_cache[key] = value

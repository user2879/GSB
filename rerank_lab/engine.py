"""Experiment engine: cache invalidation, budgeted scheduling and commands."""

from __future__ import annotations

import copy
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from .cache import DeterministicLRUCache
from .canonical import (
    decimal_text,
    derive_cache_key,
    digest,
    round_half_away,
)
from .errors import (
    E_GLOBAL_BUDGET_PENDING,
    E_INPUT_INVALID,
    E_REQUEST_BUDGET_EXHAUSTED,
    E_RESULT_STALE,
    E_STAGE_EXCEPTION,
    Failure,
    sorted_failures,
)
from .loader import _parse_candidate_record, load_sample_dir
from .models import (
    Evaluation,
    FeatureCatalog,
    LabConfig,
    LoadedSample,
    RawCandidate,
    RequestPlan,
)
from .normalize import compute_normalization
from .rerank import attach_reduction_cache_keys, rerank_request
from .scoring import (
    assemble_fusion_from_entries,
    fuse_candidate,
    fusion_parameters,
)
from .snapshot import SnapshotManager


def _norm_command_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _request_candidates(
    grouped: dict[str, dict[str, RawCandidate]], request_id: str
) -> tuple[RawCandidate, ...]:
    return tuple(grouped[request_id][item_id] for item_id in sorted(grouped[request_id]))


def _plan_keys(
    request_id: str,
    candidates: tuple[RawCandidate, ...],
    config: LabConfig,
    catalog: FeatureCatalog,
) -> dict[str, Any]:
    request_digest = digest(
        {
            "request_id": request_id,
            "catalog": list(catalog.feature_order),
            "candidates": [candidate.content_dict() for candidate in candidates],
        }
    )
    quant = config.quantisation_dict()
    stats_key = derive_cache_key(
        "normalization",
        request_digest,
        {
            "normalization": config.normalization,
            "feature_order": list(catalog.feature_order),
            "groups": [
                {
                    "name": group.name,
                    "features": list(group.features),
                    "conflict": group.conflict,
                }
                for group in catalog.groups
            ],
            "algorithm": "fraction-exact-v1",
        },
        quant,
    )
    fusion_keys = {
        candidate.item_id: derive_cache_key(
            "fusion",
            stats_key,
            {
                "candidate": candidate.content_fingerprint,
                "feature_weights": config.fusion_dict()["feature_weights"],
                "fusion_temperature": str(config.fusion_temperature),
                "score_min": str(config.score_min),
                "score_max": str(config.score_max),
                "algorithm": "ordered-integer-mac-v1",
            },
            quant,
        )
        for candidate in candidates
    }
    fusion_set = digest(
        {
            "request_id": request_id,
            "feature_dimension": catalog.dimension,
            "fusion_keys": [[item_id, fusion_keys[item_id]] for item_id in sorted(fusion_keys)],
        }
    )
    rerank_key = derive_cache_key(
        "rerank",
        fusion_set,
        {
            "rerank_window_size": config.rerank_window_size,
            "diversity_penalty": str(config.diversity_penalty),
            "similarity_threshold": str(config.similarity_threshold),
            "similarity_features": list(catalog.similarity_features),
            "lower_bound_fixed": config.score_min,
            "algorithm": "exact-integer-similarity-v1",
        },
        quant,
    )
    final_key = derive_cache_key(
        "final_sort",
        rerank_key,
        {"order": "score_desc_item_id_asc", "algorithm": "stable-final-order-v1"},
        quant,
    )
    return {
        "request_digest": request_digest,
        "stats_key": stats_key,
        "fusion_keys": fusion_keys,
        "fusion_set": fusion_set,
        "rerank_key": rerank_key,
        "final_key": final_key,
    }


def _score_rows(
    candidate_ids: Iterable[str],
    fusion_entry: dict[str, Any],
    rerank_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    after = rerank_entry["scores_after_rerank"]
    for item_id in sorted(candidate_ids):
        score = fusion_entry["candidates"][item_id]
        rerank_score = int(after[item_id])
        rows.append(
            {
                "item_id": item_id,
                "score_fixed": rerank_score,
                "fusion_score_fixed": score["score_fixed"],
                "raw_score_fixed": score["raw_score_fixed"],
                "rerank_adjustment_fixed": rerank_score - score["score_fixed"],
                "clamped": score["clamped"],
                "clamp_reason": score["clamp_reason"],
                "feature_dimension": score["feature_dimension"],
                "contributions": score["contributions"],
            }
        )
    return rows


def build_pure_request_result(
    request_id: str,
    candidate_ids: list[str],
    keys: dict[str, Any],
    normalization_entry: dict[str, Any],
    fusion_entry: dict[str, Any],
    rerank_entry: dict[str, Any],
    catalog: FeatureCatalog,
    config: LabConfig,
) -> dict[str, Any]:
    attach_reduction_cache_keys(rerank_entry, keys["rerank_key"], config)
    sorted_ids = sorted(candidate_ids)
    feature_dimension = catalog.dimension
    cache_refs = {
        "normalization": {
            "key": keys["stats_key"],
            "feature_dimension": feature_dimension,
            "candidate_ids": sorted_ids,
        },
        "fusion": [
            {
                "item_id": item_id,
                "key": keys["fusion_keys"][item_id],
                "feature_dimension": feature_dimension,
                "candidate_ids": sorted_ids,
            }
            for item_id in sorted_ids
        ],
        "rerank": {
            "key": keys["rerank_key"],
            "feature_dimension": feature_dimension,
            "candidate_ids": sorted_ids,
        },
        "final_sort": {
            "key": keys["final_key"],
            "feature_dimension": feature_dimension,
            "candidate_ids": sorted_ids,
        },
        "reductions": [
            {
                "event_key": event["event_key"],
                "cache_key": event["cache_key"],
                "item_id": event["item_id"],
                "feature_dimension": feature_dimension,
                "candidate_ids": sorted_ids,
            }
            for event in rerank_entry["reductions"]
        ],
    }
    return {
        "request_id": request_id,
        "status": "completed",
        "feature_dimension": feature_dimension,
        "candidate_ids": sorted_ids,
        "candidate_count": len(sorted_ids),
        "cache_keys": {
            "normalization": keys["stats_key"],
            "fusion": {
                item_id: keys["fusion_keys"][item_id] for item_id in sorted_ids
            },
            "fusion_set": keys["fusion_set"],
            "rerank": keys["rerank_key"],
            "final_sort": keys["final_key"],
            "reductions": [
                event["cache_key"] for event in rerank_entry["reductions"]
            ],
        },
        "cache_refs": cache_refs,
        "normalization": normalization_entry,
        "fusion": fusion_entry,
        "rerank": rerank_entry,
        "scores": _score_rows(sorted_ids, fusion_entry, rerank_entry),
        "order": rerank_entry["order"],
        "top_item_id": rerank_entry["order"][0] if rerank_entry["order"] else None,
        "score_scale": fusion_entry["score_scale"],
        "score_min_fixed": fusion_entry["score_min_fixed"],
        "score_max_fixed": fusion_entry["score_max_fixed"],
    }


class CommandOutcome:
    def __init__(
        self,
        command_type: str,
        accepted: bool,
        changed_fields: list[str] | None = None,
        failure: Failure | None = None,
        affected_hint: list[str] | None = None,
    ) -> None:
        self.command_type = command_type
        self.accepted = accepted
        self.changed_fields = changed_fields or []
        self.failure = failure
        self.affected_hint = affected_hint or []

    def to_dict(self) -> dict[str, Any]:
        out = {
            "type": self.command_type,
            "accepted": self.accepted,
            "changed_fields": self.changed_fields,
        }
        if self.affected_hint:
            out["affected_request_hint"] = sorted(self.affected_hint)
        if self.failure is not None:
            out["failure"] = self.failure.to_dict()
        return out


class ExperimentEngine:
    def __init__(
        self,
        loaded: LoadedSample,
        output_dir: str | Path,
        *,
        clean: bool = True,
        injections: list[dict[str, Any]] | None = None,
        enable_snapshots: bool = True,
    ) -> None:
        self.loaded = loaded
        self.output_dir = Path(output_dir)
        self.clean = clean
        self.enable_snapshots = enable_snapshots
        self.config = loaded.config
        self.catalog = loaded.catalog
        self.grouped: dict[str, dict[str, RawCandidate]] = {
            request_id: dict(items)
            for request_id, items in loaded.grouped_candidates().items()
        }
        self.injections = list(injections if injections is not None else loaded.injections)
        self.injection_by_request: dict[str, list[dict[str, Any]]] = {}
        for injection in sorted(
            self.injections, key=lambda item: (item["request_id"], item["injection_id"])
        ):
            self.injection_by_request.setdefault(injection["request_id"], []).append(injection)
        self.injection_hits: dict[str, dict[str, Any]] = {}
        self.base_failures = list(loaded.failures)
        self.command_failures: list[Failure] = []
        self.version = 0
        self.last_eval: Evaluation | None = None
        self.last_final_keys: dict[str, str] = {}
        self.latest_summary: dict[str, Any] | None = None
        self.command_records: list[dict[str, Any]] = []
        self.cache = DeterministicLRUCache(self.config.cache_residency_budget if self.config else 1)
        self.groups_hash = digest(self.loaded.raw_groups)
        self.config_hash = self.config.hash if self.config else digest({})
        self.candidates_hash = digest(
            [
                candidate.content_dict()
                for request_id in sorted(self.grouped)
                for item_id in sorted(self.grouped[request_id])
                for candidate in [self.grouped[request_id][item_id]]
            ]
        )
        self.snapshot = SnapshotManager(
            self.output_dir,
            config_hash=self.config_hash,
            groups_hash=self.groups_hash,
            candidates_hash=self.candidates_hash,
            config=self.config,
            catalog=self.catalog,
        )
        self.start_decision: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Geometry / cache helpers

    def _refresh_hashes(self) -> None:
        self.config_hash = self.config.hash if self.config else digest({})
        self.candidates_hash = digest(
            [
                candidate.content_dict()
                for request_id in sorted(self.grouped)
                for item_id in sorted(self.grouped[request_id])
                for candidate in [self.grouped[request_id][item_id]]
            ]
        )
        self.snapshot.config_hash = self.config_hash
        self.snapshot.candidates_hash = self.candidates_hash
        self.snapshot.config = self.config
        self.snapshot.catalog = self.catalog

    def _normalise_injection_faults(self, request_id: str) -> list[dict[str, Any]]:
        return self.injection_by_request.get(request_id, [])

    def _record_injection_hit(
        self,
        injection: dict[str, Any],
        code: str,
        detail: str,
        observed: dict[str, Any] | None = None,
    ) -> None:
        self.injection_hits[injection["injection_id"]] = {
            **injection,
            "code": code,
            "detail": detail,
            "observed": observed or {},
            "applied": True,
        }

    def _hard_injection_failure(self, request_id: str) -> Failure | None:
        for injection in self._normalise_injection_faults(request_id):
            if injection["fault"] == "quota_below_minimum":
                continue
            observed: dict[str, Any] = {}
            if injection["fault"] == "non_finite_feature":
                code = E_INPUT_INVALID
                candidates = self.grouped.get(request_id, {})
                item_id = sorted(candidates)[0] if candidates else None
                detail = (
                    "injected non-finite dense feature was rejected before scoring"
                    if item_id is None
                    else f"injected non-finite dense feature into candidate {item_id!r} was rejected before scoring"
                )
                observed = {
                    "candidate_id": item_id,
                    "field": "dense_features[0]",
                    "value": "NaN",
                }
            elif injection["fault"] == "weight_length_mismatch":
                code = E_INPUT_INVALID
                detail = "injected feature weight length does not match feature dimension"
                observed = {
                    "expected_weight_length": self.catalog.dimension,
                    "injected_weight_length": max(0, self.catalog.dimension - 1),
                }
            else:  # pragma: no cover - loader rejects this
                continue
            self._record_injection_hit(injection, code, detail, observed)
            return Failure(
                code=code,
                stage=injection["stage"],
                detail=detail,
                request_id=request_id,
                injection_id=injection["injection_id"],
            )
        return None

    def _quota_injection(self, request_id: str) -> dict[str, Any] | None:
        for injection in self._normalise_injection_faults(request_id):
            if injection["fault"] == "quota_below_minimum":
                return injection
        return None

    # ------------------------------------------------------------------
    # Evaluation

    def _build_plan(self, request_id: str) -> RequestPlan:
        assert self.config is not None
        candidates = _request_candidates(self.grouped, request_id)
        keys = _plan_keys(request_id, candidates, self.config, self.catalog)
        hard_failure = self._hard_injection_failure(request_id)
        if hard_failure is not None:
            return RequestPlan(
                request_id=request_id,
                candidates=candidates,
                request_digest=keys["request_digest"],
                stats_key=keys["stats_key"],
                stats_entry=None,
                fusion_keys=keys["fusion_keys"],
                fusion_entries={},
                rerank_key=keys["rerank_key"],
                rerank_entry=None,
                final_key=keys["final_key"],
                final_entry=None,
                missing_fusion=[],
                need_stats=False,
                need_rerank=False,
                cost=0,
                budget_status="failed",
                failure=hard_failure,
            )

        final_entry = self.cache.get(keys["final_key"])
        if final_entry is not None:
            return RequestPlan(
                request_id=request_id,
                candidates=candidates,
                request_digest=keys["request_digest"],
                stats_key=keys["stats_key"],
                stats_entry=None,
                fusion_keys=keys["fusion_keys"],
                fusion_entries={},
                rerank_key=keys["rerank_key"],
                rerank_entry=None,
                final_key=keys["final_key"],
                final_entry=final_entry,
                missing_fusion=[],
                need_stats=False,
                need_rerank=False,
                cost=0,
                budget_status="cache_hit",
            )

        stats_entry = self.cache.get(keys["stats_key"])
        fusion_entries: dict[str, dict[str, Any]] = {}
        for item_id in sorted(keys["fusion_keys"]):
            entry = self.cache.get(keys["fusion_keys"][item_id])
            if entry is not None:
                fusion_entries[item_id] = entry
        rerank_entry = self.cache.get(keys["rerank_key"])
        missing_fusion = [
            item_id for item_id in sorted(keys["fusion_keys"]) if item_id not in fusion_entries
        ]
        if rerank_entry is not None:
            cost = 0
            need_stats = False
            need_rerank = False
        else:
            pair_count = (
                min(len(candidates), self.config.rerank_window_size)
                * (min(len(candidates), self.config.rerank_window_size) - 1)
                // 2
            )
            cost = (len(missing_fusion) + pair_count) * self.catalog.dimension
            need_stats = True
            need_rerank = True
        return RequestPlan(
            request_id=request_id,
            candidates=candidates,
            request_digest=keys["request_digest"],
            stats_key=keys["stats_key"],
            stats_entry=stats_entry,
            fusion_keys=keys["fusion_keys"],
            fusion_entries=fusion_entries,
            rerank_key=keys["rerank_key"],
            rerank_entry=rerank_entry,
            final_key=keys["final_key"],
            final_entry=None,
            missing_fusion=missing_fusion,
            need_stats=need_stats,
            need_rerank=need_rerank,
            cost=cost,
            budget_status="new",
        )

    def _allocate_budget(self, plans: dict[str, RequestPlan]) -> None:
        assert self.config is not None
        remaining = self.config.global_mac_budget
        budget_closed = False
        for request_id in sorted(plans):
            plan = plans[request_id]
            if plan.failure is not None:
                plan.budget_status = "failed"
                continue
            quota_injection = self._quota_injection(request_id)
            if quota_injection is not None:
                detail = "injected per-request quota is below the minimum fusion MAC amount"
                self._record_injection_hit(
                    quota_injection,
                    E_REQUEST_BUDGET_EXHAUSTED,
                    detail,
                    observed={
                        "minimum_mac_units": self.catalog.dimension,
                        "effective_quota": max(0, self.catalog.dimension - 1),
                    },
                )
                plan.failure = Failure(
                    E_REQUEST_BUDGET_EXHAUSTED,
                    quota_injection["stage"],
                    detail,
                    request_id=request_id,
                    injection_id=quota_injection["injection_id"],
                )
                plan.budget_status = "failed"
                continue
            if plan.final_entry is not None:
                plan.admitted = True
                plan.budget_status = "cache_hit"
                continue
            if plan.rerank_entry is not None:
                plan.admitted = True
                plan.budget_status = "reused_rerank"
                continue
            if plan.cost > self.config.per_request_quota:
                plan.failure = Failure(
                    E_REQUEST_BUDGET_EXHAUSTED,
                    "budget",
                    f"request cost {plan.cost} exceeds per-request quota {self.config.per_request_quota}",
                    request_id=request_id,
                )
                plan.budget_status = "failed"
                continue
            if budget_closed or plan.cost > remaining:
                plan.budget_status = "pending"
                continue
            remaining -= plan.cost
            plan.admitted = True
            plan.budget_status = "computed" if plan.cost else "reused_stage"
        self._remaining_after_allocation = remaining

    def _run_tasks(
        self,
        plans: dict[str, RequestPlan],
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, dict[str, Any]]],
        dict[str, dict[str, Any]],
        dict[str, Failure],
        list[dict[str, Any]],
    ]:
        assert self.config is not None
        stats_by_request: dict[str, dict[str, Any]] = {
            request_id: plan.stats_entry
            for request_id, plan in plans.items()
            if plan.stats_entry is not None
        }
        fusion_by_request: dict[str, dict[str, dict[str, Any]]] = {
            request_id: dict(plan.fusion_entries) for request_id, plan in plans.items()
        }
        rerank_by_request: dict[str, dict[str, Any]] = {
            request_id: plan.rerank_entry
            for request_id, plan in plans.items()
            if plan.rerank_entry is not None
        }
        errors: dict[str, Failure] = {}
        task_records: list[dict[str, Any]] = []

        active = [
            plan
            for request_id, plan in sorted(plans.items())
            if plan.admitted
            and plan.failure is None
            and plan.final_entry is None
            and plan.rerank_entry is None
        ]
        if not active:
            return stats_by_request, fusion_by_request, rerank_by_request, errors, task_records

        stats_needed = [
            plan for plan in active if plan.need_stats and plan.stats_entry is None
        ]
        fusion_needed = [
            (plan, item_id)
            for plan in active
            for item_id in plan.missing_fusion
        ]
        rerank_needed = [plan for plan in active if plan.need_rerank]
        candidate_tasks = [
            {
                "task_key": digest(
                    {
                        "stage": "fusion",
                        "request_id": plan.request_id,
                        "item_id": item_id,
                    }
                ),
                "request_id": plan.request_id,
                "item_id": item_id,
            }
            for plan, item_id in fusion_needed
        ]
        task_records.extend(candidate_tasks)

        with ThreadPoolExecutor(
            max_workers=self.config.concurrency,
            thread_name_prefix="rerank-fusion",
        ) as executor:
            stats_futures: dict[str, Future[Any]] = {
                plan.request_id: executor.submit(
                    compute_normalization,
                    plan.candidates,
                    self.catalog,
                    self.config.normalization,
                    self.config.fixed_decimal_places,
                    self.config.quantization_step,
                )
                for plan in stats_needed
            }
            for request_id in sorted(stats_futures):
                try:
                    stats_by_request[request_id] = stats_futures[request_id].result()
                except Exception as exc:  # isolated request failure
                    errors[request_id] = Failure(
                        E_STAGE_EXCEPTION,
                        "normalize",
                        f"{type(exc).__name__}: {exc}",
                        request_id=request_id,
                    )

            params = fusion_parameters(self.config)
            fusion_futures: dict[tuple[str, str], Future[Any]] = {}
            for plan, item_id in sorted(
                fusion_needed, key=lambda item: (item[0].request_id, item[1])
            ):
                if plan.request_id in errors:
                    continue
                stats = stats_by_request.get(plan.request_id)
                if stats is None:
                    errors[plan.request_id] = Failure(
                        E_STAGE_EXCEPTION,
                        "normalize",
                        "normalization artifact missing",
                        request_id=plan.request_id,
                    )
                    continue
                candidate = self.grouped[plan.request_id][item_id]
                vector = list(stats["normalized_fixed"][item_id])
                fusion_futures[(plan.request_id, item_id)] = executor.submit(
                    fuse_candidate,
                    candidate,
                    vector,
                    self.catalog,
                    self.config,
                    params,
                )
            for request_id, item_id in sorted(fusion_futures):
                if request_id in errors:
                    continue
                try:
                    fusion_by_request.setdefault(request_id, {})[item_id] = (
                        fusion_futures[(request_id, item_id)].result()
                    )
                except Exception as exc:
                    errors[request_id] = Failure(
                        E_STAGE_EXCEPTION,
                        "fusion",
                        f"{type(exc).__name__}: {exc}",
                        request_id=request_id,
                        item_id=item_id,
                    )

            rerank_futures: dict[str, Future[Any]] = {}
            for plan in sorted(rerank_needed, key=lambda item: item.request_id):
                if plan.request_id in errors:
                    continue
                stats = stats_by_request.get(plan.request_id)
                if stats is None:
                    errors[plan.request_id] = Failure(
                        E_STAGE_EXCEPTION,
                        "normalize",
                        "normalization artifact missing for rerank",
                        request_id=plan.request_id,
                    )
                    continue
                entries = fusion_by_request.get(plan.request_id, {})
                if any(item_id not in entries for item_id in plan.fusion_keys):
                    errors[plan.request_id] = Failure(
                        E_STAGE_EXCEPTION,
                        "fusion",
                        "fusion artifact missing for rerank",
                        request_id=plan.request_id,
                    )
                    continue
                fusion_entry = assemble_fusion_from_entries(
                    plan.candidates, self.catalog, self.config, entries
                )
                rerank_futures[plan.request_id] = executor.submit(
                    rerank_request,
                    plan.candidates,
                    fusion_entry,
                    stats,
                    self.catalog,
                    self.config,
                )
            for request_id in sorted(rerank_futures):
                if request_id in errors:
                    continue
                try:
                    rerank_by_request[request_id] = rerank_futures[request_id].result()
                except Exception as exc:
                    errors[request_id] = Failure(
                        E_STAGE_EXCEPTION,
                        "rerank",
                        f"{type(exc).__name__}: {exc}",
                        request_id=request_id,
                    )
        return stats_by_request, fusion_by_request, rerank_by_request, errors, task_records

    def _evaluate(
        self,
        version: int,
        previous_keys: dict[str, str] | None = None,
    ) -> Evaluation:
        assert self.config is not None
        plans = {
            request_id: self._build_plan(request_id) for request_id in sorted(self.grouped)
        }
        current_keys = {request_id: plan.final_key for request_id, plan in plans.items()}
        previous_keys = previous_keys or {}
        affected = sorted(
            set(
                request_id
                for request_id in current_keys
                if previous_keys.get(request_id) != current_keys[request_id]
            )
            | (set(previous_keys) - set(current_keys))
        )
        self._allocate_budget(plans)
        (
            stats_by_request,
            fusion_by_request,
            rerank_by_request,
            stage_errors,
            task_records,
        ) = self._run_tasks(plans)

        results: dict[str, dict[str, Any]] = {}
        failures: list[Failure] = []
        pending: list[dict[str, Any]] = []
        completed: list[str] = []
        failed: list[str] = []
        reuse_requests: list[str] = []
        recompute_requests: list[str] = []
        cache_puts: dict[str, tuple[str, dict[str, Any]]] = {}

        for request_id in sorted(plans):
            plan = plans[request_id]
            if plan.failure is not None:
                failures.append(plan.failure)
                failed.append(request_id)
                continue
            if plan.budget_status == "pending":
                pending.append(
                    {
                        "request_id": request_id,
                        "code": E_GLOBAL_BUDGET_PENDING,
                        "stage": "budget",
                        "required_mac_units": plan.cost,
                        "remaining_global_mac_units": getattr(
                            self, "_remaining_after_allocation", 0
                        ),
                    }
                )
                continue
            if request_id in stage_errors:
                failures.append(stage_errors[request_id])
                failed.append(request_id)
                continue

            pure: dict[str, Any] | None = None
            if plan.final_entry is not None:
                pure = plan.final_entry
            elif plan.rerank_entry is not None:
                pure = plan.rerank_entry
                cache_puts[plan.final_key] = ("final_sort", pure)
            else:
                stats = stats_by_request.get(request_id)
                entries = fusion_by_request.get(request_id, {})
                rerank_entry = rerank_by_request.get(request_id)
                if stats is None or rerank_entry is None or any(
                    item_id not in entries for item_id in plan.fusion_keys
                ):
                    failures.append(
                        Failure(
                            E_STAGE_EXCEPTION,
                            "fusion",
                            "one or more stage artifacts are missing",
                            request_id=request_id,
                        )
                    )
                    failed.append(request_id)
                    continue
                fusion_entry = assemble_fusion_from_entries(
                    plan.candidates, self.catalog, self.config, entries
                )
                pure = build_pure_request_result(
                    request_id,
                    list(plan.fusion_keys),
                    {
                        "stats_key": plan.stats_key,
                        "fusion_keys": plan.fusion_keys,
                        "fusion_set": digest(
                            [
                                [item_id, plan.fusion_keys[item_id]]
                                for item_id in sorted(plan.fusion_keys)
                            ]
                        ),
                        "rerank_key": plan.rerank_key,
                        "final_key": plan.final_key,
                    },
                    stats,
                    fusion_entry,
                    rerank_entry,
                    self.catalog,
                    self.config,
                )
                cache_puts[plan.stats_key] = ("normalization", stats)
                for item_id in sorted(plan.fusion_keys):
                    cache_puts[plan.fusion_keys[item_id]] = (
                        "fusion",
                        entries[item_id],
                    )
                cache_puts[plan.rerank_key] = ("rerank", pure)
                cache_puts[plan.final_key] = ("final_sort", pure)

            fusion_hits = sorted(
                item_id
                for item_id in plan.fusion_keys
                if item_id in plan.fusion_entries
            )
            fusion_recomputed = sorted(
                item_id
                for item_id in plan.fusion_keys
                if item_id in plan.missing_fusion and plan.need_rerank
            )
            recomputed_stats = bool(plan.need_stats and plan.stats_entry is None)
            recomputed_rerank = bool(plan.need_rerank and plan.rerank_entry is None)
            telemetry = {
                "budget": {
                    "status": plan.budget_status,
                    "estimated_mac_units": plan.cost,
                    "charged_mac_units": plan.cost if plan.cost else 0,
                    "per_request_quota": self.config.per_request_quota,
                    "global_mac_budget": self.config.global_mac_budget,
                },
                "cache_reuse": {
                    "final_sort": plan.final_entry is not None,
                    "rerank": plan.rerank_entry is not None,
                    "normalization": plan.stats_entry is not None,
                    "fusion": fusion_hits,
                },
                "recomputed": {
                    "normalization": recomputed_stats,
                    "fusion": fusion_recomputed,
                    "rerank": recomputed_rerank,
                },
            }
            result = copy.deepcopy(pure)
            result.update(telemetry)
            results[request_id] = result
            completed.append(request_id)
            if plan.final_entry is not None or plan.rerank_entry is not None:
                reuse_requests.append(request_id)
            if recomputed_stats or fusion_recomputed or recomputed_rerank:
                recompute_requests.append(request_id)

        evictions: list[dict[str, Any]] = []
        for block_key in sorted(cache_puts):
            stage, value = cache_puts[block_key]
            evicted = self.cache.put(block_key, stage, value)
            for event in evicted:
                item = dict(event)
                item["version"] = version
                evictions.append(item)

        charged = sum(
            plan.cost
            for plan in plans.values()
            if plan.admitted and plan.cost > 0
        )
        remaining = max(0, self.config.global_mac_budget - charged)
        logical_latency = Fraction(charged, 1) / self.config.latency_units_per_ms
        evaluation = Evaluation(
            version=version,
            requests=results,
            plans=plans,
            failures=sorted_failures(failures),
            pending=pending,
            admitted=sorted(
                request_id for request_id, plan in plans.items() if plan.admitted
            ),
            completed=sorted(completed),
            failed=sorted(set(failed)),
            mac_units_charged=charged,
            mac_units_remaining=remaining,
            evictions=evictions,
            reuse_requests=sorted(reuse_requests),
            recompute_requests=sorted(recompute_requests),
            affected_requests=affected,
            logical_latency_ms=logical_latency,
        )
        evaluation.task_records = task_records  # type: ignore[attr-defined]
        return evaluation

    # ------------------------------------------------------------------
    # Commands

    def _config_failure(self, detail: str, code: str = E_INPUT_INVALID) -> Failure:
        return Failure(code, "validation", detail)

    def _parse_weights(self, payload: Any) -> tuple[Fraction, ...]:
        assert self.config is not None
        if isinstance(payload, dict):
            if set(payload) != set(self.catalog.feature_order):
                missing = sorted(set(self.catalog.feature_order) - set(payload))
                unknown = sorted(set(payload) - set(self.catalog.feature_order))
                raise ValueError(f"weight keys mismatch; missing={missing} unknown={unknown}")
            return tuple(Fraction(str(payload[name])) for name in self.catalog.feature_order)
        if isinstance(payload, list):
            if len(payload) != self.catalog.dimension:
                raise ValueError(
                    f"weight length {len(payload)} does not match feature dimension {self.catalog.dimension}"
                )
            return tuple(Fraction(str(value)) for value in payload)
        raise ValueError("weights must be an object or array")

    def _apply_command(self, command: dict[str, Any]) -> CommandOutcome:
        assert self.config is not None
        command_type = _norm_command_name(
            command.get("command", command.get("type", command.get("op", "")))
        )
        aliases = {
            "modify_weights": "set_weights",
            "update_weights": "set_weights",
            "change_weights": "set_weights",
            "switch_normalization": "set_normalization",
            "set_quantization": "set_quantization",
            "adjust_quantization": "set_quantization",
            "set_fixed_point": "set_quantization",
            "set_rerank": "set_rerank",
            "modify_rerank": "set_rerank",
            "update_rerank": "set_rerank",
            "add": "add_candidate",
            "remove": "remove_candidate",
        }
        command_type = aliases.get(command_type, command_type)
        try:
            if command_type == "set_weights":
                payload = command.get("weights", command.get("feature_weights"))
                weights = self._parse_weights(payload)
                old = self.config.feature_weights
                self.config = self.config.with_weights(weights)
                changed = ["feature_weights"] if old != weights else []
                return CommandOutcome(command_type, True, changed)
            if command_type == "set_normalization":
                value = _norm_command_name(
                    command.get("normalization", command.get("value", command.get("method")))
                )
                value = {"min_max": "minmax", "standard": "zscore"}.get(value, value)
                if value not in ("minmax", "zscore"):
                    raise ValueError("normalization must be minmax or zscore")
                changed = [] if value == self.config.normalization else ["normalization"]
                self.config = self.config.with_values(normalization=value)
                return CommandOutcome(command_type, True, changed)
            if command_type == "set_quantization":
                places_raw = command.get(
                    "fixed_decimal_places",
                    command.get("fixed_point_places", command.get("width")),
                )
                step_raw = command.get(
                    "quantization_step",
                    command.get("quantization_level", command.get("quant_level")),
                )
                places = self.config.fixed_decimal_places
                step = self.config.quantization_step
                changed: list[str] = []
                if places_raw is not None:
                    if isinstance(places_raw, bool) or not isinstance(places_raw, int) or places_raw <= 0:
                        raise ValueError("fixed_decimal_places must be a positive integer")
                    if places_raw != places:
                        changed.append("fixed_decimal_places")
                    places = places_raw
                if step_raw is not None:
                    if isinstance(step_raw, bool) or not isinstance(step_raw, int) or step_raw <= 0:
                        raise ValueError("quantization_step must be a positive integer")
                    if step_raw != step:
                        changed.append("quantization_step")
                    step = step_raw
                if round_half_away(self.config.fusion_temperature * (10**places)) == 0:
                    raise ValueError(
                        "fusion_temperature quantises to zero at fixed_decimal_places"
                    )
                self.config = self.config.with_values(
                    fixed_decimal_places=places, quantization_step=step
                )
                return CommandOutcome(command_type, True, changed)
            if command_type == "set_rerank":
                window_raw = command.get(
                    "rerank_window_size", command.get("window_size", command.get("window"))
                )
                penalty_raw = command.get(
                    "diversity_penalty",
                    command.get("diversity_penalty_coefficient", command.get("penalty")),
                )
                changed = []
                window = self.config.rerank_window_size
                penalty = self.config.diversity_penalty
                if window_raw is not None:
                    if isinstance(window_raw, bool) or not isinstance(window_raw, int) or window_raw <= 0:
                        raise ValueError("rerank_window_size must be a positive integer")
                    if window_raw != window:
                        changed.append("rerank_window_size")
                    window = window_raw
                if penalty_raw is not None:
                    try:
                        parsed_penalty = Fraction(str(penalty_raw))
                    except Exception as exc:
                        raise ValueError("diversity_penalty must be finite") from exc
                    if parsed_penalty < 0:
                        raise ValueError("diversity_penalty must be non-negative")
                    if parsed_penalty != penalty:
                        changed.append("diversity_penalty")
                    penalty = parsed_penalty
                self.config = self.config.with_values(
                    rerank_window_size=window, diversity_penalty=penalty
                )
                return CommandOutcome(command_type, True, changed)
            if command_type == "add_candidate":
                candidate_raw = command.get("candidate")
                if candidate_raw is None:
                    candidate_raw = {
                        key: value
                        for key, value in command.items()
                        if key not in ("command", "type", "op", "_line")
                    }
                parse_failures: list[Failure] = []
                candidate = _parse_candidate_record(
                    candidate_raw,
                    int(command.get("_line", 0) or 0),
                    parse_failures,
                )
                if candidate is None:
                    detail = parse_failures[0].detail if parse_failures else "invalid candidate"
                    return CommandOutcome(
                        command_type,
                        False,
                        failure=self._config_failure(detail),
                    )
                unknown = sorted(set(candidate.features) - set(self.catalog.feature_order))
                if unknown:
                    return CommandOutcome(
                        command_type,
                        False,
                        failure=self._config_failure(
                            f"candidate uses undeclared features: {unknown}"
                        ),
                    )
                self.grouped.setdefault(candidate.request_id, {})[
                    candidate.item_id
                ] = candidate
                self.grouped[candidate.request_id] = {
                    item_id: self.grouped[candidate.request_id][item_id]
                    for item_id in sorted(self.grouped[candidate.request_id])
                }
                return CommandOutcome(
                    command_type,
                    True,
                    ["candidates"],
                    affected_hint=[candidate.request_id],
                )
            if command_type == "remove_candidate":
                request_id = command.get("request_id", command.get("requestId"))
                item_id = command.get("item_id", command.get("itemId"))
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("remove_candidate requires request_id")
                if not isinstance(item_id, str) or not item_id:
                    raise ValueError("remove_candidate requires item_id")
                if request_id not in self.grouped or item_id not in self.grouped[request_id]:
                    return CommandOutcome(
                        command_type,
                        False,
                        failure=self._config_failure(
                            f"candidate {request_id}/{item_id} does not exist"
                        ),
                    )
                del self.grouped[request_id][item_id]
                if not self.grouped[request_id]:
                    del self.grouped[request_id]
                return CommandOutcome(
                    command_type,
                    True,
                    ["candidates"],
                    affected_hint=[request_id],
                )
            if not command_type:
                raise ValueError("command is missing a type")
            raise ValueError(f"unsupported command type {command_type!r}")
        except (TypeError, ValueError, KeyError, ArithmeticError) as exc:
            return CommandOutcome(
                command_type or "invalid",
                False,
                failure=self._config_failure(str(exc)),
            )

    # ------------------------------------------------------------------
    # Interactive summaries and report

    def _failed_request_ids(self, evaluation: Evaluation) -> set[str]:
        failed = set(evaluation.failed)
        for failure in list(self.base_failures) + list(evaluation.failures):
            if (
                failure.request_id is not None
                and failure.request_id not in self.grouped
            ):
                failed.add(failure.request_id)
        return failed

    def _summary_from_evaluation(
        self, evaluation: Evaluation, version: int, source_version: int
    ) -> dict[str, Any]:
        completed_ids = list(evaluation.completed)
        top_item = None
        for request_id in completed_ids:
            order = evaluation.requests[request_id].get("order", [])
            if order:
                top_item = order[0]
                break
        failed_ids = self._failed_request_ids(evaluation)
        total = len(
            set(self.grouped)
            | {
                failure.request_id
                for failure in list(self.base_failures) + list(evaluation.failures)
                if failure.request_id is not None
            }
        )
        failed = len(failed_ids)
        pending = len(evaluation.pending)
        eligible = max(0, total - failed)
        return {
            "version": version,
            "version_source": source_version,
            "completed_requests": len(completed_ids),
            "completed_request_ids": completed_ids,
            "current_top_item": top_item,
            "failed_requests": failed,
            "pending_requests": pending,
            "total_requests": total,
            "eligible_requests": eligible,
            "progress": {
                "completed": len(completed_ids),
                "eligible": eligible,
                "basis_points": (
                    (len(completed_ids) * 10000 // eligible) if eligible else 0
                ),
            },
            "affected_requests": list(evaluation.affected_requests),
            "error_codes": sorted(
                {failure.code for failure in evaluation.failures}
                | {item["code"] for item in evaluation.pending}
            ),
        }

    def _make_command_response(
        self,
        version: int,
        evaluation: Evaluation,
        previous_summary: dict[str, Any] | None,
        outcome: CommandOutcome,
    ) -> dict[str, Any]:
        assert self.config is not None
        active_requests = len(evaluation.plans)
        batch_size = self.config.min_command_batch
        scope = "min_command_batch" if active_requests < batch_size else "single_command"
        sample_units = evaluation.mac_units_charged * (
            batch_size if scope == "min_command_batch" else 1
        )
        estimated_ms = Fraction(sample_units, 1) / self.config.latency_units_per_ms
        budget = self.config.interaction_latency_budget_ms
        stale = estimated_ms > budget
        measurement = {
            "mode": "deterministic_mac_units",
            "units_per_ms": str(self.config.latency_units_per_ms),
            "mac_units": sample_units,
            "estimated_ms": decimal_text(estimated_ms, places=6),
            "budget_ms": str(budget),
            "scope": scope,
            "active_requests": active_requests,
            "min_command_batch": batch_size,
            "command_batch_size": batch_size if scope == "min_command_batch" else 1,
            "wall_clock_used": False,
        }
        if stale:
            if previous_summary is not None:
                summary = copy.deepcopy(previous_summary)
                source_version = previous_summary["version"]
                status = "stale"
                errors = [E_RESULT_STALE]
            else:
                summary = {
                    "version": None,
                    "version_source": None,
                    "completed_requests": 0,
                    "completed_request_ids": [],
                    "current_top_item": None,
                    "failed_requests": 0,
                    "pending_requests": 0,
                    "total_requests": len(self.grouped),
                    "eligible_requests": len(self.grouped),
                    "progress": {
                        "completed": 0,
                        "eligible": len(self.grouped),
                        "basis_points": 0,
                    },
                    "affected_requests": list(evaluation.affected_requests),
                    "error_codes": [],
                }
                source_version = None
                status = "stale"
                errors = [E_RESULT_STALE]
        else:
            summary = self._summary_from_evaluation(evaluation, version, version)
            source_version = version
            status = "current"
            errors = []
        return {
            "status": status,
            "requested_version": version,
            "version_source": source_version,
            "computed_version": version,
            "latest_completed_version": (
                previous_summary["version"] if stale and previous_summary else version
            ),
            "error_codes": errors,
            "summary": summary,
            "affected_requests": list(evaluation.affected_requests),
            "progress": summary.get("progress"),
            "measurement": measurement,
            "command": outcome.to_dict(),
        }

    def _commit_snapshot(
        self, version: int, command_index: int | None
    ) -> dict[str, Any] | None:
        if not self.enable_snapshots or self.last_eval is None:
            return None
        self._refresh_hashes()
        return self.snapshot.commit(
            version=version,
            command_index=command_index,
            candidates=[
                candidate
                for request_id in sorted(self.grouped)
                for item_id in sorted(self.grouped[request_id])
                for candidate in [self.grouped[request_id][item_id]]
            ],
            evaluation=self.last_eval,
            failures=self._combined_failures(self.last_eval),
        )

    def _combined_failures(self, evaluation: Evaluation) -> list[Failure]:
        merged = list(self.base_failures) + list(self.command_failures)
        merged.extend(evaluation.failures)
        return sorted_failures(merged)

    def _schedule_report(self, evaluation: Evaluation) -> dict[str, Any]:
        assert self.config is not None
        plan_rows = []
        for request_id in sorted(evaluation.plans):
            plan = evaluation.plans[request_id]
            plan_rows.append(
                {
                    "request_id": request_id,
                    "status": plan.budget_status,
                    "estimated_mac_units": plan.cost,
                    "admitted": plan.admitted,
                    "failure_code": plan.failure.code if plan.failure else None,
                }
            )
        candidate_tasks = getattr(evaluation, "task_records", [])
        return {
            "executor": "ThreadPoolExecutor",
            "concurrency": self.config.concurrency,
            "max_concurrency": 32,
            "budget_policy": {
                "charged_work": "fusion feature MACs plus window pair similarity MACs",
                "normalization": "not charged",
                "order": "request_id ascending",
                "per_request_quota": self.config.per_request_quota,
                "global_mac_budget": self.config.global_mac_budget,
            },
            "mac_units_charged": evaluation.mac_units_charged,
            "mac_units_remaining": evaluation.mac_units_remaining,
            "admitted_requests": list(evaluation.admitted),
            "pending_requests": [item["request_id"] for item in evaluation.pending],
            "quota_failed_requests": [
                failure.request_id
                for failure in evaluation.failures
                if failure.code == E_REQUEST_BUDGET_EXHAUSTED
            ],
            "plans": plan_rows,
            "candidate_tasks": sorted(
                candidate_tasks,
                key=lambda item: (item["request_id"], item["item_id"]),
            ),
        }

    def _fresh_projection(self) -> tuple[bool, str, str]:
        """Compare final incrementally-produced stage output to a cold run."""

        if self.config is None or self.last_eval is None or not self.enable_snapshots:
            return True, digest({}), digest({})
        saved_cache = self.cache
        saved_eval = self.last_eval
        saved_keys = self.last_final_keys
        saved_hits = copy.deepcopy(self.injection_hits)
        try:
            self.cache = DeterministicLRUCache(self.config.cache_residency_budget)
            fresh = self._evaluate(self.version, previous_keys={})
            fresh_projection = self._projection(fresh)
            incremental_projection = self._projection(saved_eval)
            self.last_eval = saved_eval
            self.last_final_keys = saved_keys
            self.injection_hits = saved_hits
            return (
                fresh_projection == incremental_projection,
                digest(incremental_projection),
                digest(fresh_projection),
            )
        finally:
            self.cache = saved_cache
            self.last_eval = saved_eval
            self.last_final_keys = saved_keys
            self.injection_hits = saved_hits

    def _projection(self, evaluation: Evaluation) -> dict[str, Any]:
        requests = {}
        for request_id in sorted(evaluation.completed):
            result = evaluation.requests[request_id]
            requests[request_id] = {
                "feature_dimension": result["feature_dimension"],
                "candidate_ids": result["candidate_ids"],
                "cache_keys": result["cache_keys"],
                "cache_refs": result["cache_refs"],
                "normalization": result["normalization"],
                "scores": result["scores"],
                "order": result["order"],
                "rerank": result["rerank"],
            }
        return {
            "requests": requests,
            "failures": [failure.to_dict() for failure in evaluation.failures],
            "pending": evaluation.pending,
        }

    def run(self) -> dict[str, Any]:
        assert self.config is not None
        assert self.catalog is not None
        self._refresh_hashes()
        self.start_decision = self.snapshot.prepare(self.clean)
        initial = self._evaluate(0, previous_keys={})
        self.last_eval = initial
        self.last_final_keys = {
            request_id: plan.final_key for request_id, plan in initial.plans.items()
        }
        self.latest_summary = self._summary_from_evaluation(initial, 0, 0)
        initial_snapshot = self._commit_snapshot(0, None)

        for index, command in enumerate(self.loaded.commands, start=1):
            self.version = index
            previous_summary = copy.deepcopy(self.latest_summary)
            outcome = self._apply_command(command)
            if outcome.failure is not None:
                self.command_failures.append(outcome.failure)
            self._refresh_hashes()
            evaluation = self._evaluate(index, previous_keys=self.last_final_keys)
            self.last_eval = evaluation
            self.last_final_keys = {
                request_id: plan.final_key for request_id, plan in evaluation.plans.items()
            }
            response = self._make_command_response(
                index, evaluation, previous_summary, outcome
            )
            snapshot_record = None
            if index % self.config.snapshot_commit_interval_commands == 0:
                snapshot_record = self._commit_snapshot(index, index)
            self.command_records.append(
                {
                    "index": index,
                    "version": index,
                    "command": outcome.to_dict(),
                    "affected_requests": list(evaluation.affected_requests),
                    "evaluation": {
                        "completed_requests": len(evaluation.completed),
                        "pending_requests": len(evaluation.pending),
                        "failed_requests": len(evaluation.failed),
                        "mac_units_charged": evaluation.mac_units_charged,
                        "mac_units_remaining": evaluation.mac_units_remaining,
                        "logical_latency_ms": decimal_text(
                            evaluation.logical_latency_ms, places=6
                        ),
                        "reuse_requests": list(evaluation.reuse_requests),
                        "recompute_requests": list(evaluation.recompute_requests),
                        "request_cache_telemetry": {
                            request_id: {
                                "cache_reuse": evaluation.requests[request_id][
                                    "cache_reuse"
                                ],
                                "recomputed": evaluation.requests[request_id][
                                    "recomputed"
                                ],
                            }
                            for request_id in sorted(evaluation.requests)
                        },
                    },
                    "response": response,
                    "snapshot": snapshot_record,
                }
            )
            self.latest_summary = self._summary_from_evaluation(
                evaluation, index, index
            )

        fresh_match, incremental_hash, fresh_hash = self._fresh_projection()
        latest_manifest = self.snapshot.latest_manifest()
        failures = self._combined_failures(self.last_eval)
        final_summary = self._summary_from_evaluation(
            self.last_eval, self.version, self.version
        )
        failed_codes = sorted({failure.code for failure in failures})
        pending_codes = sorted({item["code"] for item in self.last_eval.pending})
        summary = {
            **final_summary,
            "status": "ok" if not failures and not self.last_eval.pending else "partial",
            "error_codes": sorted(set(failed_codes) | set(pending_codes)),
            "affected_requests": list(self.last_eval.affected_requests),
            "version": self.version,
            "version_source": self.version,
        }
        reuse_count = sum(
            1
            for result in self.last_eval.requests.values()
            if result.get("cache_reuse", {}).get("final_sort")
            or result.get("cache_reuse", {}).get("rerank")
        )
        recompute_count = len(self.last_eval.recompute_requests)
        return {
            "schema_version": "1.0",
            "tool": "deterministic-rerank-lab",
            "run": {
                "mode": "incremental",
                "clean": self.clean,
                "command_count": len(self.loaded.commands),
                "initial_version": 0,
                "final_version": self.version,
                "deterministic_numeric_pipeline": "fraction/fixed-integer",
                "wall_clock_fields": False,
            },
            "input": {
                "files": self.loaded.input_files,
                "hashes": self.loaded.input_hashes,
                "config_hash": self.config_hash,
                "feature_groups_hash": self.groups_hash,
                "candidates_hash": self.candidates_hash,
            },
            "config": self.config.to_dict(),
            "feature_catalog": self.catalog.to_dict(),
            "summary": summary,
            "requests": [
                self.last_eval.requests[request_id]
                for request_id in sorted(self.last_eval.requests)
            ],
            "failures": [failure.to_dict() for failure in failures],
            "pending": list(self.last_eval.pending),
            "schedule": self._schedule_report(self.last_eval),
            "commands": list(self.command_records),
            "cache": {
                "capacity_blocks": self.cache.capacity,
                "resident_blocks": len(self.cache),
                "resident": self.cache.snapshot(),
                "evictions": list(self.cache.evictions),
                "reuse_request_count": reuse_count,
                "recompute_request_count": recompute_count,
                "reuse_requests": list(self.last_eval.reuse_requests),
                "recompute_requests": list(self.last_eval.recompute_requests),
                "policies": {
                    "eviction": "least_recently_used",
                    "tie_break": "lexicographic_block_key",
                    "key_basis": "upstream_content_fingerprint+config_fingerprint+quantization",
                },
            },
            "snapshot": {
                "start_decision": self.start_decision,
                "initial_commit": initial_snapshot,
                "commits": list(self.snapshot.commits),
                "latest_manifest": latest_manifest,
                "atomic_replace": (
                    "staging directory is fully written, previous snapshot is "
                    "renamed to backup, staging is renamed into place"
                ),
            },
            "injection_hits": [
                self.injection_hits[key] for key in sorted(self.injection_hits)
            ],
            "consistency": {
                "incremental_projection_hash": incremental_hash,
                "fresh_cold_projection_hash": fresh_hash,
                "fresh_cold_match": fresh_match,
                "comparison_scope": (
                    "scores, contribution order, rerank rounds/reductions, final order, "
                    "cache keys, cache references, failures and pending queue"
                ),
                "note": (
                    "A fresh cache may legitimately report different reuse telemetry; "
                    "the canonical result projection is compared instead."
                ),
            },
            "measurement": {
                "latency_model": "deterministic_mac_units",
                "actual_wall_clock_emitted": False,
                "reason": "field-level reproducibility across repeated Windows runs",
            },
        }


def run_experiment(
    sample_dir: str | Path,
    output_dir: str | Path,
    *,
    injections_path: str | Path | None = None,
    clean: bool = True,
    enable_snapshots: bool = True,
) -> dict[str, Any]:
    loaded = load_sample_dir(sample_dir, injections_path)
    if loaded.config is None:
        return {
            "schema_version": "1.0",
            "tool": "deterministic-rerank-lab",
            "run": {"mode": "incremental", "clean": clean},
            "input": {
                "files": loaded.input_files,
                "hashes": loaded.input_hashes,
            },
            "summary": {
                "status": "failed",
                "completed_requests": 0,
                "error_codes": sorted({failure.code for failure in loaded.failures}),
            },
            "failures": [failure.to_dict() for failure in loaded.failures],
            "requests": [],
            "config": None,
            "feature_catalog": loaded.catalog.to_dict(),
        }
    engine = ExperimentEngine(
        loaded,
        output_dir,
        clean=clean,
        enable_snapshots=enable_snapshots,
    )
    return engine.run()

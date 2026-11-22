"""Pipeline orchestration: caching, concurrency and deterministic ordering.

The engine never lets thread scheduling influence output ordering. Every task
carries a stable sort key (sample_id, region, stage); results are collected in
completion order but always re-sorted before they are emitted or audited.
Content-addressed cache hits are reused verbatim (no recomputation) and both the
reused and recomputed lists are produced as first-class, auditable output.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .config import (
    PipelineConfig,
    STAGE_LOAD,
    STAGE_FEATURE,
    STAGE_FUSION,
    STAGE_LEDGER,
    STAGE_ORDER,
)
from .errors import PipelineError, FailureRecord, failure_from_error
from .features import extract_region_feature, feature_cache_key
from .fusion import fuse_region, fusion_cache_key
from .hashing import hash_json, cache_key, combine
from .models import (
    Sample,
    RegionFeature,
    RegionResult,
    LedgerEntry,
)
from .snapshot import SnapshotHandle, stage_version_hash


@dataclass
class RunAudit:
    reused: list[str] = field(default_factory=list)
    recomputed: list[str] = field(default_factory=list)
    cache_hits: dict[str, int] = field(default_factory=dict)

    def record_reuse(self, label: str) -> None:
        self.reused.append(label)

    def record_recompute(self, label: str) -> None:
        self.recomputed.append(label)

    def finalize(self) -> None:
        self.reused.sort()
        self.recomputed.sort()
        self.cache_hits = dict(sorted(self.cache_hits.items()))


@dataclass
class EngineResult:
    samples: list[Sample]
    features: list[RegionFeature]
    results: list[RegionResult]
    ledger: list[LedgerEntry]
    failures: list[FailureRecord]
    audit: RunAudit
    recovery_version: int
    final_version: int
    stale_error: Optional[PipelineError]
    relation_hash: str


def _stable_task_key(sample_id: str, region: int, stage: str) -> tuple:
    return (sample_id, region, STAGE_ORDER.index(stage))


def _run_pool(
    max_workers: int,
    tasks: list[tuple[tuple, Callable[[], Any]]],
    failures: list[FailureRecord],
) -> list[Any]:
    """Run tasks with a bounded thread pool, capturing per-task failures.

    ``tasks`` are (sort_key, callable) pairs. Exceptions classified as
    PipelineError become failure records; other exceptions propagate. Output is
    returned already sorted by the stable key.
    """
    ordered = sorted(tasks, key=lambda item: item[0])
    if not ordered:
        return []

    def guarded(call: Callable[[], Any]) -> Any:
        try:
            return ("ok", call())
        except PipelineError as exc:
            return ("error", exc)

    workers = max(1, min(max_workers, len(ordered)))
    collected: list[tuple[tuple, Any]] = []
    if workers == 1:
        for key, call in ordered:
            collected.append((key, guarded(call)))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(guarded, call): key for key, call in ordered
            }
            for future in futures:
                collected.append((futures[future], future.result()))

    successes: list[Any] = []
    for key, outcome in sorted(collected, key=lambda item: item[0]):
        status, payload = outcome
        if status == "error":
            failures.append(failure_from_error(payload))
        else:
            successes.append(payload)
    return successes


def ledger_cache_key(entries: list[LedgerEntry], config: PipelineConfig) -> str:
    payload = [entry.to_dict() for entry in entries]
    return cache_key(
        "ledger",
        combine(hash_json(payload, prefix="ledger-entries"), config.fingerprint()),
    )


def run_engine(
    samples: list[Sample],
    config: PipelineConfig,
    snapshot: SnapshotHandle,
    *,
    failures: Optional[list[FailureRecord]] = None,
    crash_env: Optional[dict[str, str]] = None,
    start_version: int = 0,
    stale_error: Optional[PipelineError] = None,
    relation_hash: str = "",
) -> EngineResult:
    failures = list(failures or [])
    audit = RunAudit()
    crash_env = crash_env or {}

    completed_samples = [s.sample_id for s in samples]
    stage_versions = dict(snapshot.manifest.get("stage_versions", {}))
    current_version = start_version

    # ------------------------------------------------------------------ load
    # Samples are already validated by the loader; persist a load-stage marker
    # per sample so the stage is visible inside the snapshot.
    load_tasks: list[tuple[tuple, Callable[[], Any]]] = []
    for sample in samples:
        key = cache_key(
            "load",
            combine(sample.sample_id, sample.image_hash, sample.assertion_hash,
                    sample.relation_hash, config.fingerprint()),
        )

        def make_load(sample=sample, key=key):
            def task() -> dict:
                if snapshot.has_artifact(STAGE_LOAD, key):
                    audit.record_reuse(f"{STAGE_LOAD}:{sample.sample_id}:{key[:12]}")
                    return snapshot.load_artifact(STAGE_LOAD, key)
                audit.record_recompute(f"{STAGE_LOAD}:{sample.sample_id}:{key[:12]}")
                payload = sample.to_dict()
                snapshot.write_artifact_atomic(STAGE_LOAD, key, payload)
                return payload
            return task

        load_tasks.append((_stable_task_key(sample.sample_id, -1, STAGE_LOAD), make_load()))
    _run_pool(config.max_workers, load_tasks, failures)
    stage_versions[STAGE_LOAD] = stage_version_hash(
        [s.to_dict() for s in samples]
    )
    current_version = snapshot.promote_stage(
        STAGE_LOAD, config.fingerprint(), relation_hash,
        completed_samples, stage_versions,
    )
    _maybe_crash(crash_env, STAGE_LOAD, snapshot)

    # --------------------------------------------------------------- feature
    feature_tasks: list[tuple[tuple, Callable[[], Any]]] = []
    for sample in samples:
        for region in range(config.region_count):
            fkey = feature_cache_key(sample, region, config)
            tkey = _stable_task_key(sample.sample_id, region, STAGE_FEATURE)

            def make_feature(sample=sample, region=region, fkey=fkey):
                def task() -> RegionFeature:
                    if snapshot.has_artifact(STAGE_FEATURE, fkey):
                        audit.record_reuse(
                            f"{STAGE_FEATURE}:{sample.sample_id}:r{region}:{fkey[:12]}"
                        )
                        raw = snapshot.load_artifact(STAGE_FEATURE, fkey)
                        return RegionFeature(**raw)
                    audit.record_recompute(
                        f"{STAGE_FEATURE}:{sample.sample_id}:r{region}:{fkey[:12]}"
                    )
                    feature = extract_region_feature(sample, region, config)
                    snapshot.write_artifact_atomic(
                        STAGE_FEATURE, fkey, feature.to_dict()
                    )
                    return feature
                return task

            feature_tasks.append((tkey, make_feature()))
    features: list[RegionFeature] = _run_pool(
        config.max_workers, feature_tasks, failures
    )
    features.sort(key=lambda item: item.sort_key)
    stage_versions[STAGE_FEATURE] = stage_version_hash(
        [f.to_dict() for f in features]
    )
    current_version = snapshot.promote_stage(
        STAGE_FEATURE, config.fingerprint(), relation_hash,
        completed_samples, stage_versions,
    )
    _maybe_crash(crash_env, STAGE_FEATURE, snapshot)

    # ---------------------------------------------------------------- fusion
    sample_by_id = {s.sample_id: s for s in samples}
    fusion_tasks: list[tuple[tuple, Callable[[], Any]]] = []
    for feature in features:
        sample = sample_by_id[feature.sample_id]
        rkey = fusion_cache_key(feature, config, sample.relation_hash)

        def make_fusion(feature=feature, sample=sample, rkey=rkey):
            def task() -> RegionResult:
                if snapshot.has_artifact(STAGE_FUSION, rkey):
                    audit.record_reuse(
                        f"{STAGE_FUSION}:{feature.sample_id}:r{feature.region}:{rkey[:12]}"
                    )
                    raw = snapshot.load_artifact(STAGE_FUSION, rkey)
                    return _region_result_from_dict(raw)
                audit.record_recompute(
                    f"{STAGE_FUSION}:{feature.sample_id}:r{feature.region}:{rkey[:12]}"
                )
                result = fuse_region(feature, sample, config)
                snapshot.write_artifact_atomic(
                    STAGE_FUSION, rkey, result.to_dict()
                )
                return result
            return task

        fusion_tasks.append(
            (_stable_task_key(feature.sample_id, feature.region, STAGE_FUSION),
             make_fusion())
        )
    results: list[RegionResult] = _run_pool(
        config.max_workers, fusion_tasks, failures
    )
    results.sort(key=lambda item: item.sort_key)
    stage_versions[STAGE_FUSION] = stage_version_hash(
        [r.to_dict() for r in results]
    )
    current_version = snapshot.promote_stage(
        STAGE_FUSION, config.fingerprint(), relation_hash,
        completed_samples, stage_versions,
    )
    _maybe_crash(crash_env, STAGE_FUSION, snapshot)

    # ---------------------------------------------------------------- ledger
    ledger_entries = [
        LedgerEntry(
            sample_id=r.sample_id,
            region=r.region,
            category=r.category,
            confidence_fixed=r.confidence_fixed,
            boundary=r.boundary,
            boundary_reasons=list(r.boundary_reasons),
            cache_key=r.cache_key,
        )
        for r in results
    ]
    ledger_entries.sort(key=lambda item: item.sort_key)
    lkey = ledger_cache_key(ledger_entries, config)
    if snapshot.has_artifact(STAGE_LEDGER, lkey):
        audit.record_reuse(f"{STAGE_LEDGER}:all:{lkey[:12]}")
    else:
        audit.record_recompute(f"{STAGE_LEDGER}:all:{lkey[:12]}")
        snapshot.write_artifact_atomic(
            STAGE_LEDGER, lkey, [e.to_dict() for e in ledger_entries]
        )
    stage_versions[STAGE_LEDGER] = stage_version_hash(
        [e.to_dict() for e in ledger_entries]
    )
    current_version = snapshot.promote_stage(
        STAGE_LEDGER, config.fingerprint(), relation_hash,
        completed_samples, stage_versions,
    )
    _maybe_crash(crash_env, STAGE_LEDGER, snapshot)

    failures.sort(key=lambda record: record.sort_key)
    audit.finalize()
    return EngineResult(
        samples=samples,
        features=features,
        results=results,
        ledger=ledger_entries,
        failures=failures,
        audit=audit,
        recovery_version=start_version,
        final_version=current_version,
        stale_error=stale_error,
        relation_hash=relation_hash,
    )


def _maybe_crash(
    crash_env: dict[str, str], stage: str, snapshot: SnapshotHandle
) -> None:
    target = crash_env.get("MM_CRASH_AFTER_STAGE")
    if target != stage:
        return
    # Hard exit simulates SIGKILL: no Python cleanup, only fully replaced
    # snapshots survive. Write a marker first for test observability.
    marker = crash_env.get("MM_CRASH_MARKER")
    if marker:
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write(str(snapshot.version))
    os._exit(137)


def _region_result_from_dict(raw: dict) -> RegionResult:
    from .models import Contribution
    payload = dict(raw)
    payload["contributions"] = [
        Contribution(**item) for item in raw["contributions"]
    ]
    return RegionResult(**payload)

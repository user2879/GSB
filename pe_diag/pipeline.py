"""Deterministic orchestration of the full diagnostic workbench.

The pipeline is a pure function of the four input files.  It plans a stable,
sorted list of (scheme, sample, stage) tasks keyed by
``scheme|sample_id|stage``, then executes them either serially or through a
thread pool.  Concurrency only changes *when* a task runs: every task reads
only its own inputs and the immutable config, results are reordered back to the
serial key order, and the snapshot is committed by a single thread, so the
concurrent and serial reports are field-for-field identical.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .aggregation import aggregate
from .cache import (CacheAudit, make_cache_key, sample_fingerprint, task_key)
from .common import content_hash
from .config import Config, Sample
from .error_tracking import track_sample
from .injection import InjectionBook
from .scoring import score_sample
from .sensitivity import run_sensitivity

SENSITIVITY_SAMPLES_PER_TIER = 12
from .snapshot import SnapshotStore
from .strategy import TierMetrics, select_strategy
from .errors import DiagError

# When the user-level thread pool is enabled, keep the underlying BLAS backend
# single-threaded to avoid thread oversubscription (nested BLAS pools otherwise
# serialize on the GIL and run much slower than the serial path).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    threadpool_limits = None

STAGES = ("encode", "score", "error")


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _sample_id_of(encode_obj: dict) -> str:
    return encode_obj.get("sample_id", "")


@dataclass
class PlannedTask:
    stage: str
    scheme: str
    sample: Sample
    precision: str
    decision: dict
    tkey: str
    cache_key: str
    tier_id: str


@dataclass
class TaskOutcome:
    tkey: str
    stage: str
    scheme: str
    sample_id: str
    tier_id: str
    cache_key: str
    payload: dict | None
    failure: dict | None


class DiagnosticPipeline:
    def __init__(self, config: Config, samples: list, injections: list,
                 scale_doc: dict, snapshot_root: str):
        self.config = config
        self.samples = samples
        self.scale_doc = scale_doc
        self.injection_book = InjectionBook(injections)
        self.input_hashes = config.fingerprint_inputs(
            samples_doc=[s.content_dict() for s in samples],
            scale_doc=scale_doc,
            injection_doc=[inj.__dict__ for inj in injections],
        )
        self.store = SnapshotStore(snapshot_root, self.input_hashes)
        self.snapshot_status = self.store.status
        self.decisions = self._plan_decisions()
        self.audit = CacheAudit(reused=[], recomputed=[], invalidated=[])
        # cache_key -> payload for entries computed in this run; snapshot
        # objects are loaded lazily so resumption stays memory-bounded
        self.memory: dict[str, dict] = {}
        self._new_count = 0
        self._sample_fps = {
            s.sample_id: sample_fingerprint(s) for s in samples}

    # -- decisions per tier ---------------------------------------------------
    def _plan_decisions(self) -> dict:
        decisions = {}
        for tier in self.scale_doc["tiers"]:
            tier_id = tier["tier_id"]
            cfg = self.config.effective(tier_id)
            metrics = TierMetrics(
                tier_id=tier_id,
                sample_count=tier["sample_count"],
                dim=cfg.dim,
                extrapolation_ratio=float(tier["extrapolation_ratio"]),
                train_len=cfg.train_len,
                target_len=cfg.target_len,
                batch_point_budget=cfg.batch_point_budget,
                memory_limit_mb=cfg.memory_limit_mb,
            )
            decisions[tier_id] = select_strategy(metrics)
        return decisions

    # -- task planning --------------------------------------------------------
    def plan_tasks(self) -> list[PlannedTask]:
        planned: list[PlannedTask] = []
        for sample in self.samples:
            cfg = self.config.effective(sample.tier_id)
            decision = self.decisions[sample.tier_id]["decision"]
            for scheme in self.config.schemes:
                scheme_decision = dict(decision)
                scheme_decision["scheme"] = scheme
                for precision in cfg.precision_tiers:
                    for stage in STAGES:
                        tkey = task_key(scheme, sample.sample_id, stage)
                        cache_key = make_cache_key(
                            stage, scheme, cfg, sample, precision,
                            scheme_decision,
                            sample_fp=self._sample_fps[sample.sample_id])
                        planned.append(PlannedTask(
                            stage=stage, scheme=scheme, sample=sample,
                            precision=precision, decision=scheme_decision,
                            tkey=tkey, cache_key=cache_key,
                            tier_id=sample.tier_id))
        planned.sort(key=lambda t: t.tkey)
        return planned

    def _sensitivity_tasks(self) -> list[tuple[str, str, str]]:
        # one per tier; cache key derived inside run via stage payload hash
        return [(t["tier_id"], "sensitivity", t["tier_id"])
                for t in self.scale_doc["tiers"]]

    # -- per-task execution ---------------------------------------------------
    def _mutated_sample(self, sample: Sample, key_overrides: dict | None):
        if not key_overrides:
            return sample
        idx = key_overrides["key_index"]
        positions = list(sample.key_positions)
        positions[idx] = key_overrides["position"]
        return Sample(
            sample_id=sample.sample_id, tier_id=sample.tier_id,
            query_pos=sample.query_pos, key_positions=tuple(positions),
            query_features=sample.query_features,
            key_features=sample.key_features, index=sample.index)

    def execute_task(self, task: PlannedTask) -> TaskOutcome:
        cfg = self.config.effective(task.tier_id)
        sample = task.sample
        injection = self.injection_book.for_task(sample.sample_id, task.stage)
        feature_overrides = None
        base_override = None
        scale_override = None
        injection_id = None
        if injection is not None:
            injection_id = injection.injection.injection_id
            feature_overrides = injection.feature_overrides
            base_override = injection.base_override
            scale_override = injection.scale_override
            sample = self._mutated_sample(sample, injection.key_overrides)

        try:
            payload = self._compute_stage(
                task, cfg, sample, feature_overrides,
                base_override, scale_override)
            payload["_injection_id"] = injection_id
            return TaskOutcome(
                tkey=task.tkey, stage=task.stage, scheme=task.scheme,
                sample_id=sample.sample_id, tier_id=task.tier_id,
                cache_key=task.cache_key, payload=payload, failure=None)
        except DiagError as exc:
            return TaskOutcome(
                tkey=task.tkey, stage=task.stage, scheme=task.scheme,
                sample_id=sample.sample_id, tier_id=task.tier_id,
                cache_key=task.cache_key, payload=None,
                failure={
                    "task_key": task.tkey,
                    "sample_id": sample.sample_id,
                    "scheme": task.scheme,
                    "stage": task.stage,
                    "tier_id": task.tier_id,
                    "code": exc.code,
                    "message": exc.message,
                    "injection_id": injection_id,
                })

    def _compute_stage(self, task, cfg, sample, feature_overrides,
                       base_override, scale_override) -> dict:
        from .encoding import PositionEncoder
        if task.stage == "encode":
            encoder = PositionEncoder(
                scheme=task.scheme,
                base=cfg.base if base_override is None else base_override,
                train_len=cfg.train_len, target_len=cfg.target_len,
                dim=cfg.dim,
                scale=cfg.scale if scale_override is None else scale_override,
                chunk_size=cfg.chunk_size, precision=task.precision,
                active_dim=task.decision["active_dim"])
            positions = [sample.query_pos, *sample.key_positions]
            result = encoder.encode_positions(positions)
            payload = {"stage": "encode", "result": result.to_dict()}
            payload["result"]["sample_id"] = sample.sample_id
            return payload

        if task.stage == "score":
            result = score_sample(
                sample, cfg, task.decision, task.precision,
                base_override=base_override, scale_override=scale_override,
                feature_overrides=feature_overrides)
            return {"stage": "score", "result": result.to_dict()}

        # error stage needs both precision score results; reuse memory or
        # compute on demand (still deterministic, same code path)
        result64 = self._score_dict(task, cfg, sample, "float64",
                                    feature_overrides, base_override,
                                    scale_override)
        result32 = None
        if "float32" in cfg.precision_tiers:
            result32 = self._score_dict(task, cfg, sample, "float32",
                                        feature_overrides, base_override,
                                        scale_override)
        tracked = track_sample(sample, cfg, task.decision,
                               result64, result32)
        return {"stage": "error", "result": tracked}

    def _score_dict(self, task, cfg, sample, precision, feature_overrides,
                    base_override, scale_override) -> dict:
        score_decision = dict(task.decision)
        score_decision["scheme"] = task.scheme
        cache_key = make_cache_key(
            "score", task.scheme, cfg, sample, precision, score_decision,
            sample_fp=self._sample_fps[sample.sample_id])
        injected = (feature_overrides is not None or
                    base_override is not None or scale_override is not None)
        if not injected and cache_key in self.memory:
            return self.memory[cache_key]["result"]
        if not injected and self.store.has_entry("score", cache_key):
            value = self.store.load_object("score", cache_key)
            self.memory[cache_key] = value
            return value["result"]
        result = score_sample(
            sample, cfg, score_decision, precision,
            base_override=base_override, scale_override=scale_override,
            feature_overrides=feature_overrides)
        return result.to_dict()

    # -- run loop -------------------------------------------------------------
    def run(self, parallel: bool) -> dict:
        planned = self.plan_tasks()
        self._planned_tasks = planned
        todo: list[PlannedTask] = []
        for task in planned:
            if self.store.has_entry(task.stage, task.cache_key):
                self.audit.reused.append(task.tkey)
            else:
                todo.append(task)
                self.audit.recomputed.append(task.tkey)

        # entries present in the old snapshot but no longer requested are
        # recorded as invalidated (they remain only in the old generation)
        requested_keys = {(task.stage, task.cache_key) for task in planned}
        present = set()
        for stage, entries in self.store.manifest.get("entries", {}).items():
            for cache_key in entries:
                present.add((stage, cache_key))
        for stage, cache_key in sorted(present - requested_keys):
            if stage in STAGES:
                self.audit.invalidated.append(
                    "{}:{}".format(stage, cache_key[:12]))

        outcomes: list[TaskOutcome] = []
        pending_payload: dict[str, dict] = {stage: {} for stage in STAGES}
        pending_tasks: dict[str, list[str]] = {stage: [] for stage in STAGES}
        max_workers = self._max_workers()

        def flush(stage: str) -> None:
            if pending_payload[stage]:
                self.store.commit(
                    stage, pending_payload[stage], pending_tasks[stage])
                pending_payload[stage] = {}
                pending_tasks[stage] = []

        if parallel and max_workers > 1 and todo:
            limiter = (threadpool_limits(limits=1)
                       if threadpool_limits is not None
                       else _NullCtx())
            with limiter:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    # map preserves input order, which is already the serial
                    # key order -> deterministic output regardless of finish
                    for outcome in pool.map(self.execute_task, todo):
                        outcomes.append(outcome)
                        self._absorb(outcome, pending_payload,
                                     pending_tasks)
                        if self._should_commit(outcome):
                            flush(outcome.stage)
        else:
            for task in todo:
                outcome = self.execute_task(task)
                outcomes.append(outcome)
                self._absorb(outcome, pending_payload, pending_tasks)
                if self._should_commit(outcome):
                    flush(outcome.stage)
        for stage in STAGES:
            flush(stage)

        # outcomes from reused snapshot entries (for report completeness)
        reused_outcomes = self._materialize_reused(planned)
        all_outcomes = sorted(outcomes + reused_outcomes,
                              key=lambda o: o.tkey)
        sensitivity = self._run_sensitivity()
        report = self._assemble_report(all_outcomes, sensitivity, parallel)
        return report

    def _max_workers(self) -> int:
        return max(self.config.effective(t["tier_id"]).workers
                   for t in self.scale_doc["tiers"])

    def _should_commit(self, outcome: TaskOutcome) -> bool:
        # snapshot_interval driven by global tier config; use the smallest
        # configured interval so every configured cadence is honored
        intervals = [self.config.effective(t["tier_id"]).snapshot_interval
                     for t in self.scale_doc["tiers"]]
        interval = min(intervals)
        return (self._new_count % interval == 0)

    def _absorb(self, outcome: TaskOutcome, pending_payload,
                pending_tasks) -> None:
        # NOTE: runs only on the orchestrating thread, so snapshot mutation is
        # never shared with worker threads (workers are pure functions)
        if outcome.payload is not None:
            self._new_count += 1
            self.memory[outcome.cache_key] = outcome.payload
            pending_payload[outcome.stage][outcome.cache_key] = \
                outcome.payload
            pending_tasks[outcome.stage].append(outcome.tkey)
        elif outcome.failure is not None:
            # durable per-task failure; excluded from statistics by design
            self.store.record_failure(outcome.failure)

    def _materialize_reused(self, planned: list) -> list[TaskOutcome]:
        out = []
        for task in planned:
            if self.store.has_entry(task.stage, task.cache_key):
                payload = self.memory.get(task.cache_key)
                if payload is None:
                    payload = self.store.load_object(
                        task.stage, task.cache_key)
                    self.memory[task.cache_key] = payload
                out.append(TaskOutcome(
                    tkey=task.tkey, stage=task.stage, scheme=task.scheme,
                    sample_id=task.sample.sample_id, tier_id=task.tier_id,
                    cache_key=task.cache_key, payload=payload, failure=None))
        return out

    # -- sensitivity (per tier) ----------------------------------------------
    def _run_sensitivity(self) -> dict:
        blocks = {}
        for tier in self.scale_doc["tiers"]:
            tier_id = tier["tier_id"]
            cfg = self.config.effective(tier_id)
            decision = dict(self.decisions[tier_id]["decision"])
            samples = [s for s in self.samples if s.tier_id == tier_id]
            ordered = sorted(samples, key=lambda smp: smp.sample_id)
            subset = ordered[:SENSITIVITY_SAMPLES_PER_TIER]
            subset_fp = content_hash(
                [self._sample_fps[smp.sample_id] for smp in subset])
            cache_key = make_cache_key(
                "sensitivity", "all", cfg, None, "float64",
                {**decision, "subset_fp": subset_fp})
            if self.store.has_entry("sensitivity", cache_key):
                if cache_key not in self.memory:
                    self.memory[cache_key] = self.store.load_object(
                        "sensitivity", cache_key)
                blocks[tier_id] = self.memory[cache_key]["result"]
                self.audit.reused.append(
                    task_key("all", tier_id, "sensitivity"))
                continue
            result = run_sensitivity(
                samples, cfg, decision, list(self.config.schemes),
                sample_limit=SENSITIVITY_SAMPLES_PER_TIER)
            payload = {"stage": "sensitivity", "result": result}
            self.memory[cache_key] = payload
            self.store.commit("sensitivity", {cache_key: payload},
                              [task_key("all", tier_id, "sensitivity")])
            blocks[tier_id] = result
            self.audit.recomputed.append(
                task_key("all", tier_id, "sensitivity"))
        return blocks

    # -- report assembly ------------------------------------------------------
    def _assemble_report(self, outcomes: list, sensitivity: dict,
                         parallel: bool) -> dict:
        failures = sorted(self.store.all_failures(),
                          key=lambda f: f["task_key"])
        # authoritative per-stage successes: load every durable object so the
        # report is identical whether results were just computed or resumed
        # restrict assembly to the *currently requested* keys so stale objects
        # left behind by precise invalidation never leak into the report
        requested = {stage: set() for stage in STAGES}
        for task in self._planned_tasks:
            requested[task.stage].add(task.cache_key)
        score_results = []
        error_results = []
        encode_objects = []
        present = self.store.manifest.get("entries", {})
        for cache_key in present.get("score", {}):
            if cache_key in requested["score"]:
                score_results.append(
                    self.store.load_object("score", cache_key)["result"])
        for cache_key in present.get("error", {}):
            if cache_key in requested["error"]:
                error_results.append(
                    self.store.load_object("error", cache_key)["result"])
        for cache_key in present.get("encode", {}):
            if cache_key in requested["encode"]:
                encode_objects.append(self.store.load_object(
                    "encode", cache_key)["result"])
        successes = [o for o in outcomes if o.payload is not None]
        encode_meta = []
        for obj in encode_objects:
            encode_meta.append({
                "scheme": obj["scheme"],
                "sample_id": _sample_id_of(obj),
                "precision": obj["precision"],
                "dim_range": obj["dim_range"],
                "base": obj["base"],
                "effective_base": obj["effective_base"],
                "scale": obj["scale"],
                "chunk_size": obj["chunk_size"],
                "injection_id": obj.get("_injection_id"),
            })
        # stable ordering key scheme|sample_id|stage (encode is constant stage)
        encode_meta.sort(key=lambda row: "{}|{}|{}|encode".format(
            row["scheme"], row["sample_id"], row["precision"]))

        tier_reports = {}
        for tier in self.scale_doc["tiers"]:
            tier_id = tier["tier_id"]
            tier_reports[tier_id] = {
                "strategy_decision": self.decisions[tier_id],
                "layered_statistics": aggregate(score_results, tier_id),
                "error_tracking": sorted(
                    [r for r in error_results if r["tier_id"] == tier_id],
                    key=lambda r: (r["scheme"], r["sample_id"])),
                "sensitivity": sensitivity[tier_id],
            }

        return {
            "input_hashes": self.input_hashes,
            "snapshot": {
                "status": self.snapshot_status,
                "global_version": self.store.manifest["global_version"],
                "generation_dir": os.path.basename(
                    str(self.store.generation_dir)),
            },
            "execution": {
                "parallel": bool(parallel),
                "max_workers": self._max_workers(),
                "task_ordering_key": "scheme|sample_id|stage",
            },
            "cache_audit": self.audit.to_dict(),
            "tier_reports": tier_reports,
            "encoding_metadata": encode_meta,
            "failures": failures,
            "failure_codes": sorted({f["code"] for f in failures}),
        }

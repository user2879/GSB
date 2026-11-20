"""Pipeline orchestration engine.

Determinism strategy
--------------------
* The point stream is fixed by input ordering.
* Within every point, feature tasks follow (topo_rank, feature name).
* Budget decisions / ledger appends / cache mutations happen on this thread;
  the thread pool only runs side-effect-free evaluation callables.
* Therefore concurrency and timing cannot change any output ordering.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .budget import BudgetScheduler, _ledger_from_dict
from .cache import BlockCache, block_key
from .common import (
    CODE_FEAT_BUDGET_EXHAUSTED,
    CODE_GLOBAL_BUDGET_EXHAUSTED,
    CODE_INJECTION_HIT,
    Failure,
    canonical_dumps,
    read_json,
    sha256_text,
    write_json,
    write_jsonl,
)
from .compute import FeatResult, FeatureComputer
from .config import PipelineConfig
from .loader import (
    load_events,
    load_feature_defs,
    load_injections,
    load_label_defs,
)
from .models import Event, FeatureDef, InjectionRule

RESULT_KEY = tuple[str, str, str, int]
from .sample import SampleGenerator
from .snapshot import SnapshotStore
from .storage import EventStore


@dataclass
class Point:
    seq: int
    kind: str           # event | query
    anchor_ts: int
    event: Event | None
    user_id: str
    item_id: str


class PipelineEngine:
    def __init__(self, cfg: PipelineConfig, snapshot: SnapshotStore) -> None:
        self.cfg = cfg
        self.snapshot = snapshot
        self.failures: list[Failure] = []
        self.feat_results: list[FeatResult] = []
        self.samples: list = []
        self.invalidated: list[dict[str, Any]] = []
        self.reused: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ load
    def load_inputs(self, event_path: str, feature_path: str,
                    label_path: str, injection_path: str | None
                    ) -> tuple[list[Event], list[FeatureDef], Any, list[InjectionRule]]:
        events, event_failures = load_events(event_path, self.cfg.behaviors)
        self.failures.extend(event_failures)

        defs, def_failures = load_feature_defs(feature_path)
        self.failures.extend(def_failures)

        valid_names = {d.name for d in defs if d.valid}
        labels, label_failures = load_label_defs(label_path, valid_names, self.cfg.behaviors)
        self.failures.extend(label_failures)

        injections: list[InjectionRule] = []
        if injection_path and os.path.exists(injection_path):
            injections = load_injections(injection_path)
        return events, defs, labels, injections

    def _apply_load_injections(self, events: list[Event],
                               injections: list[InjectionRule]) -> list[Event]:
        rules = [r for r in injections if r.stage == "load"]
        by_event: dict[str, list[InjectionRule]] = {}
        for rule in rules:
            by_event.setdefault(rule.event_id, []).append(rule)
        kept: list[Event] = []
        for ev in events:
            hit = False
            for rule in by_event.get(ev.event_id, []):
                self.failures.append(Failure(
                    "load", ev.event_id, CODE_INJECTION_HIT,
                    f"{rule.kind}", rule.injection_id))
                if rule.kind == "non_finite_context":
                    field = rule.context_field or (self.cfg.context_fields[0]
                                                   if self.cfg.context_fields else "x")
                    ev = ev.with_correction(
                        context={**ev.context, field: float("nan")})
                    # non-finite context makes the event invalid and isolated
                    self.failures.append(Failure(
                        "load", ev.event_id, "E_EVENT_INVALID",
                        "injected non-finite context", rule.injection_id))
                    hit = True
                elif rule.kind == "unknown_behavior":
                    self.failures.append(Failure(
                        "load", ev.event_id, "E_EVENT_INVALID",
                        "injected unknown behavior", rule.injection_id))
                    hit = True
                elif rule.kind == "backfill_order":
                    self.failures.append(Failure(
                        "load", ev.event_id, "E_BACKFILL_ORDER",
                        f"injected ts {rule.bad_ts_ms} before cursor",
                        rule.injection_id))
                    hit = True
            if not hit:
                kept.append(ev)
        return kept

    # ------------------------------------------------------------- point stream
    @staticmethod
    def build_points(events: list[Event], query_points: list[dict[str, Any]]) -> list[Point]:
        points: list[Point] = []
        seq = 0
        for ev in events:
            points.append(Point(seq, "event", ev.ts_ms, ev, ev.user_id, ev.item_id))
            seq += 1
        for q in sorted(query_points, key=lambda x: (x["ts_ms"], x.get("user_id", ""),
                                                     x.get("item_id", ""))):
            points.append(Point(seq, "query", int(q["ts_ms"]), None,
                                str(q.get("user_id", "")), str(q.get("item_id", ""))))
            seq += 1
        return points

    # ------------------------------------------------------------------- run
    def run(self, events: list[Event], defs: list[FeatureDef], labels: Any,
            query_points: list[dict[str, Any]] | None = None,
            precomputed: dict[tuple[str, str, str, int], FeatResult] | None = None,
            replay_invalid: set[tuple[str, str, str, int]] | None = None,
            resume_after_seq: int = -1) -> dict[str, Any]:
        valid_defs = [d for d in defs if d.valid]
        feature_side = {d.name: d.side for d in valid_defs}
        store = EventStore()
        store.rebuild(events)
        computer = FeatureComputer(store, self.cfg.fp, self.cfg.default_empty_value)
        cache = BlockCache(self.cfg.cache_block_limit)
        scheduler = BudgetScheduler(
            self.cfg.global_event_budget, self.cfg.feature_point_quota,
            self.cfg.thread_concurrency)
        sampler = SampleGenerator(labels, computer, self.cfg.fp.scale, self.cfg.quant_step)

        points = self.build_points(events, query_points or [])
        precomputed = precomputed or {}
        replay_invalid = replay_invalid or set()
        reuse_ledger = {"reused": [], "recomputed": []}
        self._reuse_ledger = reuse_ledger
        self._resume_rows = getattr(self, "_resume_rows", [])
        self._resume_defs = {d.name: d for d in valid_defs}
        if resume_after_seq >= 0:
            self._restore_scheduler(scheduler)
        self._scheduler = scheduler
        self._cache = cache
        self._computer = computer
        self._store = store
        self._points = points

        for point in points:
            if point.seq <= resume_after_seq:
                # Replay pre-cursor points through the deterministic plan only
                # to rebuild ledger/observation state, skipping thread work.
                self._replay_resume_point(point, valid_defs, computer, cache,
                                          scheduler, store)
                continue
            self._run_point(point, valid_defs, computer, cache, scheduler, store,
                            precomputed=precomputed, reuse_ledger=reuse_ledger)
            if point.kind == "event" and point.event is not None:
                new_samples, sample_failures = sampler.evaluate(
                    point.event, point.seq, feature_side,
                    set(scheduler.aborted), event_point_seq=point.seq)
                self.samples.extend(new_samples)
                self.failures.extend(sample_failures)
            if (point.seq + 1) % max(1, self.cfg.snapshot_commit_every) == 0:
                self._commit_snapshot(point, defs)

        # always commit the final state
        if points:
            self._commit_snapshot(points[-1], defs, final=True)

        return self._collect(valid_defs, labels, scheduler, cache)

    # ------------------------------------------------------------- one point
    def _run_point(self, point: Point, defs: list[FeatureDef],
                   computer: FeatureComputer, cache: BlockCache,
                   scheduler: BudgetScheduler, store: EventStore,
                   precomputed: dict[RESULT_KEY, FeatResult] | None = None,
                   reuse_ledger: dict[str, list[str]] | None = None) -> None:
        precomputed = precomputed or {}
        # Sliding windows are anchor-specific: residency never crosses points.
        cache.clear_residency()
        point_blocks: dict[str, FeatResult] = {}
        by_name = {d.name: d for d in defs}
        max_level = max((d.level for d in defs), default=-1)
        aborted_upstream: set[str] = set()

        for level in range(max_level + 1):
            level_defs = [d for d in defs if d.level == level]
            level_defs.sort(key=lambda d: (d.topo_rank, d.name))
            tasks: list[tuple[FeatureDef, Any]] = []
            for feat in level_defs:
                entity_id = point.user_id if feat.side == "user" else point.item_id
                key = block_key(feat.name, feat.side, entity_id)
                result_key = (feat.name, feat.side, entity_id, point.seq)

                upstream_aborted = any(u in aborted_upstream for u in feat.upstream)
                # Planning always runs (even on reuse/cache hit) so the auditable
                # budget ledger evolves identically in serial, concurrent and
                # incremental executions.
                decision, _ = scheduler.plan(point.seq, feat, upstream_aborted)
                if decision != "charged":
                    aborted_upstream.add(feat.name)
                    continue

                # 1) Historical block reused after an edit: never recomputed.
                if result_key in precomputed:
                    reused = precomputed[result_key]
                    cache.put(key, reused, point.seq)
                    point_blocks[key] = reused
                    self._adopt_result(reused, computer, point)
                    if reuse_ledger is not None:
                        reuse_ledger["reused"].append("|".join(map(str, result_key)))
                    continue

                # 2) Downstream cache gate: every upstream block must be
                #    referenced; an evicted block is recomputed and verified.
                if feat.is_derived and cache.enabled:
                    for up_name in feat.upstream:
                        up_feat = by_name[up_name]
                        up_entity = point.user_id if up_feat.side == "user" else point.item_id
                        up_key = block_key(up_name, up_feat.side, up_entity)
                        referenced = cache.reference(up_key, point.seq,
                                                      set(point_blocks))
                        if referenced is None:
                            recomputed = self._recompute_block(
                                up_feat, up_entity, point, computer, store)
                            original = point_blocks.get(up_key)
                            if original is not None and getattr(
                                    self, "_strict_recompute", True):
                                # element-wise identity guarantee within one
                                # run/config; relaxed for incremental replays
                                assert recomputed.value == original.value
                                assert recomputed.empty == original.empty
                            cache.put(up_key, recomputed, point.seq)
                            point_blocks[up_key] = recomputed

                def make_callable(f=feat, eid=entity_id, p=point, c=computer, s=store):
                    if f.is_derived:
                        return lambda: c.evaluate_derived(f, eid, p.seq, p.anchor_ts)
                    return lambda: c.evaluate_base(f, eid, p.seq, p.anchor_ts)

                tasks.append((feat, make_callable()))
                if reuse_ledger is not None:
                    reuse_ledger["recomputed"].append("|".join(map(str, result_key)))

            results = scheduler.run_level(tasks)
            for (feat, _), result in zip(tasks, results):
                entity_id = point.user_id if feat.side == "user" else point.item_id
                key = block_key(feat.name, feat.side, entity_id)
                cache.put(key, result, point.seq)
                point_blocks[key] = result
                self._adopt_result(result, computer, point)

    def _restore_scheduler(self, scheduler: BudgetScheduler) -> None:
        all_ledger = getattr(self, "_resume_ledger", [])
        cursor = getattr(self, "_resume_cursor_seq", -1)
        # Only pre-cursor entries form the restored prefix; post-cursor points
        # are re-executed and append fresh, identically ordered entries.
        ledger = [e for e in all_ledger if e["point_seq"] <= cursor]
        scheduler.ledger = []
        max_seq = 0
        for entry in ledger:
            max_seq = max(max_seq, entry["seq"])
            name = entry["feature"]
            if entry["decision"] == "charged":
                scheduler.feature_used[name] = entry["feature_used"]
                if scheduler.global_budget >= 0:
                    scheduler.global_remaining = entry["global_after"]
            elif entry["decision"] in ("skip_aborted", "skip_global"):
                # The first aborting decision for a feature is its canonical
                # failure; later skips merely repeat the same abort state.
                if name not in scheduler.aborted:
                    code = (CODE_GLOBAL_BUDGET_EXHAUSTED
                            if entry["decision"] == "skip_global"
                            else CODE_FEAT_BUDGET_EXHAUSTED)
                    scheduler.aborted[name] = Failure(
                        "feature", name, code,
                        self._abort_detail(code, scheduler))
            elif entry["decision"] == "skip_upstream" and name not in scheduler.aborted:
                scheduler.aborted[name] = Failure(
                    "feature", name, CODE_FEAT_BUDGET_EXHAUSTED,
                    "upstream feature aborted")
            # Keep the original snapshot sequence numbers verbatim so the
            # restored ledger prefix is byte identical to the fresh run.
            scheduler.ledger.append(_ledger_from_dict(entry))
        scheduler._seq = max_seq

    @staticmethod
    def _abort_detail(code: str, scheduler: BudgetScheduler) -> str:
        if code == CODE_GLOBAL_BUDGET_EXHAUSTED:
            return f"global budget {scheduler.global_budget} exhausted"
        return f"per-feature quota {scheduler.feature_quota} exhausted"

    def _replay_resume_point(self, point: Point, defs: list[FeatureDef],
                             computer: FeatureComputer, cache: BlockCache,
                             scheduler: BudgetScheduler, store: EventStore) -> None:
        """Rebuild pre-cursor state from persisted rows without re-planning.

        Budget usage, abort flags and ledger entries are restored verbatim from
        the snapshot, so the resumed run continues with identical remaining
        capacity and identical failure set.
        """
        rows = [r for r in self._resume_rows if r["point_seq"] == point.seq]
        # Rebuild residency exactly as a fresh point would: clear at the point
        # boundary then re-insert blocks in canonical (topo_rank, name) order.
        cache.clear_residency()
        for row in rows:
            result = FeatResult(
                row["feature"], row["side"], row["entity_id"], row["point_seq"],
                row["anchor_ts"], row["value"], row["empty"],
                row["empty_source"], row["n_inputs"])
            computer.observations.setdefault(
                result.feature, {}).setdefault(result.entity_id, [])
            computer.observations[result.feature][result.entity_id].append(
                (result.anchor_ts, point.seq, result.value, result.empty))
            key = block_key(result.feature, result.side, result.entity_id)
            cache.put(key, result, point.seq)
            self.feat_results.append(result)
        # Replay downstream references in topological order so hits, evictions
        # and recomputes reproduce the fresh run's deterministic cache trace.
        by_name = self._resume_defs
        for row in rows:
            feat = by_name[row["feature"]]
            if not feat.is_derived:
                continue
            user_id = point.user_id
            item_id = point.item_id
            for up_name in feat.upstream:
                up_feat = by_name[up_name]
                up_entity = user_id if up_feat.side == "user" else item_id
                up_key = block_key(up_name, up_feat.side, up_entity)
                point_created = {block_key(r["feature"], r["side"], r["entity_id"])
                                 for r in rows}
                cache.reference(up_key, point.seq, point_created)

    def _recompute_block(self, feat: FeatureDef, entity_id: str, point: Point,
                         computer: FeatureComputer, store: EventStore) -> FeatResult:
        if feat.is_derived:
            return computer.evaluate_derived(feat, entity_id, point.seq, point.anchor_ts)
        return computer.evaluate_base(feat, entity_id, point.seq, point.anchor_ts)
    def _adopt_result(self, result: FeatResult, computer: FeatureComputer,
                      point: Point) -> None:
        self.feat_results.append(result)
        computer.record_observation(result.feature, result.entity_id,
                                    result.anchor_ts, point.seq, result.value,
                                    result.empty)

    # -------------------------------------------------------------- snapshots
    def _definition_hash(self, defs: list[FeatureDef]) -> str:
        payload = [
            {k: getattr(d, k) for k in
             ("name", "side", "window_ms", "agg", "source_field", "upstream",
              "empty_value", "valid")}
            for d in defs
        ]
        return sha256_text(canonical_dumps(payload))

    def _config_hash(self) -> str:
        payload = {
            "thread_concurrency": self.cfg.thread_concurrency,
            "global_event_budget": self.cfg.global_event_budget,
            "feature_point_quota": self.cfg.feature_point_quota,
            "cache_block_limit": self.cfg.cache_block_limit,
            "snapshot_commit_every": self.cfg.snapshot_commit_every,
            "fixed_point_scale": self.cfg.fixed_point_scale,
            "quant_step": self.cfg.quant_step,
        }
        return sha256_text(canonical_dumps(payload))

    def _commit_snapshot(self, point: Point, defs: list[FeatureDef],
                         final: bool = False) -> None:
        # Stage artifacts are written to temp files and atomically replaced so
        # an interrupted process never leaves a partially readable snapshot.
        rank_by_name = {d.name: d.topo_rank for d in defs}
        ordered_results = sorted(
            self.feat_results,
            key=lambda r: (r.point_seq, rank_by_name.get(r.feature, 0),
                           r.feature, r.side, r.entity_id))
        feature_rows = [
            {
                "point_seq": r.point_seq,
                "anchor_ts": r.anchor_ts,
                "feature": r.feature,
                "side": r.side,
                "entity_id": r.entity_id,
                "value": r.value,
                "empty": r.empty,
                "empty_source": r.empty_source,
                "n_inputs": r.n_inputs,
            }
            for r in ordered_results
        ]
        write_jsonl(self.snapshot.stage_path("feature", "features.jsonl"),
                    feature_rows)
        if hasattr(self, "_scheduler"):
            write_jsonl(self.snapshot.stage_path("feature", "budget_ledger.jsonl"),
                        [e.__dict__ for e in self._scheduler.ledger])
            write_jsonl(self.snapshot.stage_path("sample", "samples.jsonl"),
                        [s.to_dict() for s in self.samples])
        manifest = {
            "global_version": point.seq + 1,
            "definition_hash": self._definition_hash(defs),
            "config_hash": self._config_hash(),
            "cursor_event_id": point.event.event_id if point.event else f"query:{point.seq}",
            "cursor_ts_ms": point.anchor_ts,
            "stages": {
                "load": {"version": 1, "hash": self.snapshot.hash_stage("load")},
                "feature": {"version": point.seq + 1,
                            "hash": self.snapshot.hash_stage("feature")},
                "sample": {"version": point.seq + 1,
                           "hash": self.snapshot.hash_stage("sample")},
            },
            "final": final,
        }
        self.snapshot.save_manifest(manifest)

    # --------------------------------------------------------------- collect
    def _collect(self, valid_defs: list[FeatureDef], labels: Any,
                 scheduler: BudgetScheduler, cache: BlockCache) -> dict[str, Any]:
        self.failures.extend(scheduler.aborted.values())
        self.failures.sort(key=lambda f: (f.stage, f.target, f.code, f.detail,
                                          f.injection_id))
        rank_by_name = {d.name: d.topo_rank for d in valid_defs}
        ordered_results = sorted(
            self.feat_results,
            key=lambda r: (r.point_seq, rank_by_name.get(r.feature, 0),
                           r.feature, r.side, r.entity_id))
        feature_rows = [
            {
                "point_seq": r.point_seq,
                "anchor_ts": r.anchor_ts,
                "feature": r.feature,
                "side": r.side,
                "entity_id": r.entity_id,
                "value": r.value,
                "empty": r.empty,
                "empty_source": r.empty_source,
                "n_inputs": r.n_inputs,
            }
            for r in ordered_results
        ]
        return {
            "reuse_ledger": self._reuse_ledger,
            "features": feature_rows,
            "samples": [s.to_dict() for s in self.samples],
            "ledger": [e.__dict__ for e in scheduler.ledger],
            "failures": [f.to_dict() for f in self.failures],
            "cache": {
                "resident_keys": cache.resident_keys(),
                "registry": {str(k): sorted(v)
                             for k, v in sorted(cache.registry.items())},
                "hits": cache.hits,
                "misses": cache.misses,
                "evictions": cache.evictions,
                "recomputes": cache.recomputes,
                "events": [e.__dict__ for e in cache.events],
            },
            "global_remaining": scheduler.global_remaining,
            "feature_used": dict(sorted(scheduler.feature_used.items())),
            "pending_queue": list(scheduler.pending),
        }

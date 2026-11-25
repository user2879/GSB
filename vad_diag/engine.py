"""Incremental, fingerprint-driven VAD orchestrator.

The engine maintains the accepted segment set and rebuilds a deterministic
:class:`~vad_diag.timeline.Timeline` after every command.  All stage products
are addressed by content/parameter/quantization cache keys, so a segment
append/replace/delete or a parameter change invalidates *only* the downstream
products that actually reference the changed input; everything else is served
from the persistent cache and explicitly logged as a reuse hit.

A latency-budget gate distinguishes a fresh authoritative summary from a stale
one (``E_STALE_RESULT``).  Stale summaries never advance the completed cursor and
are never written to the ledger as authoritative.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import vad as vad_mod
from . import windows as win_mod
from . import posterior as post_mod
from . import dsp
from .cache import ProductCache
from .config import Config
from .errors import (
    FailureRecord,
    VADFieldError,
    VADInjectionError,
    E_FIELD_INVALID,
    E_STALE_RESULT,
)
from .fingerprint import (
    combine,
    digest,
    fingerprint_str_set,
    make_cache_key,
)
from .injection import InjectionHarness
from .ledger import Ledger
from .loader import Command
from .segments import Segment
from .snapshot import SnapshotState, SnapshotStore, config_hash
from .timeline import Timeline


@dataclass
class PipelineResult:
    timeline: Timeline
    frames: List[vad_mod.FrameRecord]
    frame_intervals: List[Tuple[int, int]]
    regions: List[vad_mod.SpeechRegion]
    region_meta: Dict[str, Any]
    windows: List[win_mod.Window]
    features: Dict[str, win_mod.WindowFeature]
    posteriors: Dict[str, post_mod.WindowPosterior]
    decisions: List[post_mod.FrameDecision]
    endpoints: List[post_mod.EndpointInterval]
    keys: Dict[str, Any]
    failures: List[FailureRecord] = field(default_factory=list)
    affected_windows: List[str] = field(default_factory=list)
    reused_windows: List[str] = field(default_factory=list)
    recomputed_windows: List[str] = field(default_factory=list)


class VADEngine:
    def __init__(self, config: Config, ledger: Ledger, cache: ProductCache,
                 snapshot_store: SnapshotStore,
                 harness: Optional[InjectionHarness] = None,
                 recompute_budget: Optional[int] = None):
        self.config = config
        self.ledger = ledger
        self.cache = cache
        self.snapshots = snapshot_store
        self.harness = harness or InjectionHarness([])
        self.segments: Dict[str, Segment] = {}
        # Arrival order is independent of capture order.
        self.arrival_order: List[str] = []
        self.failures: List[FailureRecord] = []
        self.recompute_budget = recompute_budget
        self.recompute_budget_used = 0
        self.completed_segments: List[str] = []
        self.last_authoritative: Optional[Dict[str, Any]] = None
        self.global_version = 0
        self._last_result: Optional[PipelineResult] = None
        self._prev_regions: Dict[str, Dict[str, Any]] = {}
        # Memoized per-timeline frame fingerprints, invalidated when the
        # timeline content fingerprint changes.
        self._frame_fp_timeline: Optional[str] = None
        self._frame_fp_cache: Dict[int, str] = {}
        # When False, snapshots/cache/ledger are only flushed on explicit
        # snapshot/shutdown (keeps tight latency-budget loops fast).
        self.auto_flush = True

    # ------------------------------------------------------------------ keys
    def _quant_level(self) -> str:
        return f"Q{self.config.frac_bits}_{self.config.quant_mode}"

    def _param_fp(self) -> str:
        return digest(["params", self.config.material_params()])

    def _quant_fp(self) -> str:
        return digest(["quant", self.config.quant_params()])

    def _projection_fp(self) -> str:
        return digest([
            "projection",
            post_mod.PROJECTION_NAME,
            list(post_mod.PROJECTION_WEIGHTS_REAL),
            post_mod.PROJECTION_BIAS_REAL,
            list(post_mod._SQUASH_POINTS),
        ])

    def _window_feature_key(self, window: win_mod.Window,
                            timeline: Timeline) -> str:
        seg_fps = [timeline.get(sid).content_fingerprint()
                   for sid in window.segment_ids]
        self._ensure_frame_fp_cache(timeline)
        frame_fps = [
            self._frame_fp_cache[fi]
            for fi in window.frame_indices
        ]
        return make_cache_key(
            "window_feature",
            [fingerprint_str_set(window.segment_ids),
             combine(seg_fps), combine(frame_fps)],
            params=self._param_fp(),
            quant=self._quant_fp(),
            dim=dsp.FEATURE_DIM)

    def _ensure_frame_fp_cache(self, timeline: Timeline) -> None:
        tfp = timeline.fingerprint()
        if self._frame_fp_timeline == tfp:
            return
        flat = timeline.samples()
        cache: Dict[int, str] = {}
        for fi, (start, end) in enumerate(
                vad_mod.dsp.frame_intervals(
                    timeline.total_samples, self.config.frame_length,
                    self.config.frame_shift)):
            cache[fi] = digest(["frame", fi, flat[start:end]])
        self._frame_fp_cache = cache
        self._frame_fp_timeline = tfp

    def _frame_bounds(self, frame_index: int) -> Tuple[int, int]:
        start = frame_index * self.config.frame_shift
        return start, start + self.config.frame_length

    def _posterior_key(self, feature_key: str) -> str:
        return make_cache_key(
            "posterior", [feature_key],
            projection=self._projection_fp(),
            quant=self._quant_fp(),
            threshold=self.config.speech_threshold)

    def _endpoint_key(self, posterior_keys: Sequence[str]) -> str:
        return make_cache_key(
            "endpoint", sorted(posterior_keys),
            quant=self._quant_fp(),
            threshold=self.config.speech_threshold)

    def _region_key(self, timeline: Timeline) -> str:
        return make_cache_key(
            "vad_regions", [timeline.fingerprint()],
            params=self._param_fp())

    # ------------------------------------------------------------- pipeline
    def run_pipeline(self, changed_segments: Optional[List[str]] = None
                     ) -> PipelineResult:
        timeline = Timeline(list(self.segments.values()))
        changed_segments = changed_segments or list(self.segments.keys())
        failures: List[FailureRecord] = list(self.failures)
        affected: List[str] = []
        reused: List[str] = []
        recomputed: List[str] = []

        # ---- Stage: VAD -------------------------------------------------
        regions, frames, region_meta = vad_mod.run_vad(timeline, self.config)
        frame_intervals = [(f.start, f.end) for f in frames]
        region_key = self._region_key(timeline)
        cached_regions = self.cache.get(region_key)
        if cached_regions is None:
            self.ledger.record_recompute(
                "vad", region_key, "regions",
                quant_level=self._quant_level())
            self.cache.put(region_key, {
                "regions": [r.to_dict() for r in regions],
                "metadata": region_meta,
            })
            self._log_region_moves(regions, changed_segments)
        else:
            self.ledger.record_reuse(
                "vad", region_key, "timeline+params unchanged", "regions")
        self._prev_regions = {r.region_id: r.to_dict() for r in regions}

        # ---- Stage: windows + fixed-point features ---------------------
        windows = win_mod.cut_windows(timeline, self.config, frame_intervals)
        window_by_id = {w.window_id: w for w in windows}
        features: Dict[str, win_mod.WindowFeature] = {}
        budget = self.recompute_budget
        for window in windows:
            key = self._window_feature_key(window, timeline)
            cached = self.cache.get(key)
            touches_changed = any(sid in changed_segments
                                  for sid in window.segment_ids)
            if cached is not None:
                # Exact content+param+quant match -> reuse persisted product.
                features[window.window_id] = win_mod.WindowFeature(
                    window_id=window.window_id,
                    int_features=tuple(cached["int_features"]),
                    frac_bits=cached["frac_bits"],
                    quant_mode=cached["quant_mode"],
                    frame_count=cached["frame_count"],
                    cache_key=key)
                reused.append(window.window_id)
                self.ledger.record_reuse(
                    "window_feature", key,
                    "content/param/quant cache hit", window.window_id)
                continue

            if cached is not None:
                self.ledger.record_invalidation(
                    "window_feature", key,
                    "quantization or upstream content changed",
                    window.window_id,
                    caused_by=self._first_changed(window, changed_segments),
                    quant_level=self._quant_level())
            elif touches_changed:
                self.ledger.record_invalidation(
                    "window_feature", key,
                    "references changed segment", window.window_id,
                    caused_by=self._first_changed(window, changed_segments),
                    quant_level=self._quant_level())

            if budget is not None:
                if self.recompute_budget_used >= budget:
                    # Out of recompute budget: leave pending for stale path.
                    affected.append(window.window_id)
                    continue
                self.recompute_budget_used += 1

            feature = self._compute_feature(window, timeline, frame_intervals,
                                            key, failures)
            if feature is not None:
                features[window.window_id] = feature
                recomputed.append(window.window_id)
                affected.append(window.window_id)
                self.cache.put(key, feature.to_dict())
                self.ledger.record_recompute(
                    "window_feature", key, window.window_id,
                    quant_level=self._quant_level())

        # ---- Stage: posterior ------------------------------------------
        posteriors: Dict[str, post_mod.WindowPosterior] = {}
        for window in windows:
            if window.window_id not in features:
                continue
            feature = features[window.window_id]
            pkey = self._posterior_key(feature.cache_key)
            cached = self.cache.get(pkey)
            if cached is not None:
                posteriors[window.window_id] = self._posterior_from_cache(
                    cached, pkey)
                self.ledger.record_reuse(
                    "posterior", pkey, "feature+projection cache hit",
                    window.window_id)
                continue
            posterior = post_mod.synthesize_posterior(
                feature, self.config, cache_key=pkey)
            posteriors[window.window_id] = posterior
            self.cache.put(pkey, posterior.to_dict())
            self.ledger.record_recompute(
                "posterior", pkey, window.window_id,
                quant_level=self._quant_level())

        # ---- Stage: frame endpoint restore -----------------------------
        decisions = post_mod.restore_frame_decisions(
            windows, posteriors, frame_intervals, self.config)
        endpoints = post_mod.build_endpoint_intervals(
            decisions, frame_intervals, windows, posteriors)
        ekey = self._endpoint_key(
            [posteriors[w].cache_key for w in sorted(posteriors)])
        self.cache.put(ekey, {
            "endpoints": [e.to_dict() for e in endpoints],
            "decisions": [d.to_dict() for d in decisions],
        })

        keys = {
            "timeline": timeline.fingerprint(),
            "params": self._param_fp(),
            "quant": self._quant_fp(),
            "projection": self._projection_fp(),
            "region": region_key,
            "endpoint": ekey,
        }
        result = PipelineResult(
            timeline=timeline, frames=frames, frame_intervals=frame_intervals,
            regions=regions, region_meta=region_meta, windows=windows,
            features=features, posteriors=posteriors, decisions=decisions,
            endpoints=endpoints, keys=keys, failures=failures,
            affected_windows=sorted(set(affected)),
            reused_windows=sorted(set(reused)),
            recomputed_windows=sorted(set(recomputed)))
        self._last_result = result
        return result

    def _first_changed(self, window: win_mod.Window,
                       changed: Sequence[str]) -> Optional[str]:
        changed_set = set(changed)
        for sid in window.segment_ids:
            if sid in changed_set:
                return sid
        return None

    def _compute_feature(self, window, timeline, frame_intervals, key,
                         failures):
        try:
            feature = win_mod.extract_window_feature(
                window, timeline, self.config, frame_intervals,
                cache_key=key)
            feature = self.harness.window_hook(window.window_id, feature)
            return feature
        except VADInjectionError as exc:
            self._record_injection_failure(exc, failures)
            return None

    def _posterior_from_cache(self, cached, key):
        return post_mod.WindowPosterior(
            window_id=cached["window_id"],
            linear_q=cached["linear_q"],
            posterior_q=cached["posterior_q"],
            posterior=cached["posterior"],
            is_speech=cached["is_speech"],
            boundary_window=cached["boundary_window"],
            tie_kind=cached["tie_kind"],
            cache_key=key)

    def _log_region_moves(self, regions, changed_segments):
        current = {r.region_id: r.to_dict() for r in regions}
        for rid, new in current.items():
            old = self._prev_regions.get(rid)
            if old is not None and (
                old["frame_start"] != new["frame_start"] or
                old["frame_end"] != new["frame_end"]):
                self.ledger.record_region_move(
                    rid, old, new,
                    changed_segments[0] if changed_segments else "")

    # --------------------------------------------------------------- summary
    def _watermark(self, result: Optional[PipelineResult]) -> Dict[str, Any]:
        if result is None:
            return {"segment_ids": [], "timeline_fp": "", "version": 0}
        return {
            "segment_ids": result.timeline.segment_ids,
            "timeline_fp": result.timeline.fingerprint(),
            "version": self.global_version,
            "quant_level": self._quant_level(),
        }

    def build_summary(self, result: PipelineResult, *, stale: bool,
                      code: Optional[str], cost_ms: int,
                      budget_ms: int) -> Dict[str, Any]:
        summary = {
            "status": "stale" if stale else "fresh",
            "code": code,
            "cost_ms": cost_ms,
            "latency_budget_ms": budget_ms,
            "completed_segment_count": len(self.completed_segments),
            "accepted_segment_count": len(self.segments),
            "completed_window_count": len(result.features),
            "total_window_count": len(result.windows),
            "affected_windows": result.affected_windows,
            "reused_windows": result.reused_windows,
            "recomputed_windows": result.recomputed_windows,
            "progress": self._progress(result),
            "endpoint_intervals": [e.to_dict() for e in result.endpoints],
            "speech_regions": [r.to_dict() for r in result.regions],
            "boundary_windows": sorted(
                p.window_id for p in result.posteriors.values()
                if p.boundary_window),
            "boundary_frames": sorted(
                d.frame_index for d in result.decisions
                if d.boundary_frame),
            "watermark": self._watermark(result),
            "failures": [f.to_dict() for f in result.failures],
        }
        return summary

    def _progress(self, result: PipelineResult) -> Dict[str, Any]:
        total = max(1, len(result.windows))
        done = len(result.features)
        return {
            "windows_done": done,
            "windows_total": len(result.windows),
            "ratio": round(done / total, 6),
            "recompute_budget_used": self.recompute_budget_used,
        }

    def _estimate_cost_ms(self, changed_count: int,
                          result: PipelineResult) -> int:
        return (changed_count * self.config.segment_cost_ms
                + len(result.recomputed_windows) * self.config.window_cost_ms)

    # ------------------------------------------------------------- commands
    def handle_command(self, command: Command) -> Dict[str, Any]:
        op = command.op
        payload = command.payload
        if op == "append":
            return self._apply_segment_change(command, "append")
        if op == "replace":
            return self._apply_segment_change(command, "replace")
        if op == "delete":
            return self._apply_delete(command)
        if op == "reorder":
            return self._apply_reorder(command)
        if op == "set_params":
            return self._apply_set_params(command)
        if op == "set_quant":
            return self._apply_set_quant(command)
        if op == "query":
            return self._answer_query(command)
        if op == "snapshot":
            return self._take_snapshot(command)
        if op == "shutdown":
            return self._shutdown(command)
        raise VADFieldError(f"unknown op {op}", E_FIELD_INVALID, stage="load")

    def _admit_segment(self, segment: Segment) -> None:
        # Injection hook for load-stage segment faults.
        rule = self.harness.for_target(segment.segment_id)
        if rule is not None and rule.stage == "load":
            try:
                self.harness.corrupt_segment(segment)
            except VADInjectionError as exc:
                self._record_injection_failure(exc, self.failures)
                return
        if segment.segment_id in self.segments:
            self.failures.append(FailureRecord(
                seq=len(self.failures), code=E_FIELD_INVALID, stage="load",
                message="duplicate segment_id", item_id=segment.segment_id))
            return
        self.segments[segment.segment_id] = segment
        self.arrival_order.append(segment.segment_id)

    def _record_injection_failure(self, exc, failures) -> None:
        record = FailureRecord(
            seq=len(failures), code=exc.code, stage=exc.stage,
            message=str(exc), item_id=exc.item_id,
            injection_id=getattr(exc, "injection_id", None),
            detail=getattr(exc, "detail", {}))
        failures.append(record)
        self.ledger.record_failure(record)

    def _apply_segment_change(self, command: Command, kind: str
                              ) -> Dict[str, Any]:
        payload = command.payload
        segment: Segment = payload["segment"]
        changed = [segment.segment_id]
        if kind == "replace":
            self.segments.pop(segment.segment_id, None)
            if segment.segment_id not in self.arrival_order:
                self.arrival_order.append(segment.segment_id)
        self._admit_segment(segment)
        return self._finish_mutation(command, changed)

    def _apply_delete(self, command: Command) -> Dict[str, Any]:
        seg_id = command.payload["segment_id"]
        if seg_id in self.segments:
            del self.segments[seg_id]
        if seg_id in self.arrival_order:
            self.arrival_order.remove(seg_id)
        return self._finish_mutation(command, [seg_id])

    def _apply_reorder(self, command: Command) -> Dict[str, Any]:
        order = list(command.payload.get("arrival_order", []))
        known = set(self.segments)
        if set(order) != known:
            raise VADFieldError(
                "reorder must list exactly the accepted segments",
                E_FIELD_INVALID, stage="load")
        self.arrival_order = order
        # Arrival order never changes content; no product invalidation.
        return self._finish_mutation(command, [], content_changed=False)

    def _apply_set_params(self, command: Command) -> Dict[str, Any]:
        updates = command.payload.get("params", {})
        new_dict = self.config.to_dict()
        new_dict.update(updates)
        from .config import build_config
        self.config = build_config(new_dict)
        return self._finish_mutation(command, list(self.segments),
                                     content_changed=False,
                                     params_changed=True)

    def _apply_set_quant(self, command: Command) -> Dict[str, Any]:
        payload = command.payload
        mode = payload.get("quant_mode", self.config.quant_mode)
        frac_bits = payload.get("frac_bits", self.config.frac_bits)
        # Injection / validation of the quantization level.
        target = payload.get("target", "config")
        self.harness.validate_quant_mode(mode, target)
        if frac_bits not in (8, 15, 23):
            raise VADFieldError(
                f"frac_bits {frac_bits} invalid", E_FIELD_INVALID,
                stage="load")
        new_dict = self.config.to_dict()
        new_dict["quant_mode"] = mode
        new_dict["frac_bits"] = frac_bits
        from .config import build_config
        self.config = build_config(new_dict)
        return self._finish_mutation(command, list(self.segments),
                                     content_changed=False,
                                     params_changed=True)

    def _finish_mutation(self, command: Command, changed: List[str], *,
                         content_changed: bool = True,
                         params_changed: bool = False) -> Dict[str, Any]:
        self.global_version += 1
        result = self.run_pipeline(changed_segments=changed)
        cost = self._estimate_cost_ms(len(changed), result)
        budget = self.config.latency_budget_ms
        if cost <= budget:
            self._mark_complete(result)
            summary = self.build_summary(
                result, stale=False, code=None, cost_ms=cost,
                budget_ms=budget)
            self.last_authoritative = summary
            self.ledger.record_summary(summary)
            self._auto_snapshot()
            return {"command": command.to_dict(), "summary": summary}
        # Over budget: serve the last authoritative summary as stale.
        stale = self._stale_summary(cost, budget, result)
        return {"command": command.to_dict(), "summary": stale}

    def _mark_complete(self, result: PipelineResult) -> None:
        self.completed_segments = result.timeline.segment_ids

    def _stale_summary(self, cost: int, budget: int,
                       current: PipelineResult) -> Dict[str, Any]:
        if self.last_authoritative is None:
            base = self.build_summary(
                current, stale=True, code=E_STALE_RESULT, cost_ms=cost,
                budget_ms=budget)
            base["watermark"] = self._watermark(None)
            base["note"] = "no authoritative summary yet"
            return base
        stale = copy.deepcopy(self.last_authoritative)
        stale["status"] = "stale"
        stale["code"] = E_STALE_RESULT
        stale["cost_ms"] = cost
        stale["latency_budget_ms"] = budget
        stale["stale_watermark"] = stale.get("watermark")
        stale["current_watermark"] = self._watermark(current)
        # Explicitly not ledgered as authoritative and cursor not advanced.
        return stale

    def _answer_query(self, command: Command) -> Dict[str, Any]:
        if self._last_result is None:
            self.run_pipeline()
        result = self._last_result
        cost = self._estimate_cost_ms(0, result)
        if self.last_authoritative is not None and cost <= \
                self.config.latency_budget_ms:
            summary = copy.deepcopy(self.last_authoritative)
            return {"command": command.to_dict(), "summary": summary}
        stale = self._stale_summary(cost, self.config.latency_budget_ms, result)
        return {"command": command.to_dict(), "summary": stale}

    # ------------------------------------------------------------ snapshot
    def _snapshot_state(self) -> SnapshotState:
        result = self._last_result
        pending = [w.window_id for w in result.windows
                   if w.window_id not in result.features] if result else []
        products: Dict[str, Dict[str, Any]] = {
            "load": {
                "segments": sorted(self.segments.keys()),
                "arrival_order": list(self.arrival_order),
                "failures": [f.to_dict() for f in self.failures],
            },
            "vad": {
                "regions": [r.to_dict() for r in result.regions]
                if result else [],
                "metadata": result.region_meta if result else {},
            },
            "window": {
                wid: feat.to_dict()
                for wid, feat in result.features.items()
            } if result else {},
            "posterior": {
                wid: post.to_dict()
                for wid, post in result.posteriors.items()
            } if result else {},
        }
        return SnapshotState(
            version=self.global_version,
            config_hash=config_hash(self.config.to_dict()),
            quant_level=self._quant_level(),
            stage_versions={stage: self.global_version for stage in
                            ("load", "vad", "window", "posterior")},
            completed_segments=list(self.completed_segments),
            pending_windows=pending,
            recompute_budget_used=self.recompute_budget_used,
            products=products)

    def _auto_snapshot(self, crash_point: Optional[str] = None) -> None:
        if not self.auto_flush and crash_point is None:
            return
        state = self._snapshot_state()
        self.snapshots.save(state, crash_point=crash_point)
        self.cache.flush()
        self.ledger.flush()

    def _take_snapshot(self, command: Command,
                       crash_point: Optional[str] = None) -> Dict[str, Any]:
        if self._last_result is None:
            self.run_pipeline()
        state = self._snapshot_state()
        self.snapshots.save(state, crash_point=crash_point)
        self.cache.flush()
        self.ledger.flush()
        return {
            "command": command.to_dict(),
            "snapshot": {
                "version": self.global_version,
                "config_hash": config_hash(self.config.to_dict()),
                "quant_level": self._quant_level(),
                "pending_windows": self._snapshot_state().pending_windows,
                "completed_segments": list(self.completed_segments),
            },
        }

    def restore_from_snapshot(self) -> Optional[SnapshotState]:
        expected = config_hash(self.config.to_dict())
        state = self.snapshots.recover(expected)
        if state is None:
            return None
        self.global_version = state.version
        self.completed_segments = list(state.completed_segments)
        self.recompute_budget_used = state.recompute_budget_used
        load_product = state.products.get("load", {})
        self.arrival_order = list(load_product.get("arrival_order", []))
        # Segment *bytes* are not persisted in the snapshot by design; the
        # caller re-feeds the manifest. Recovery validates config/version and
        # resumes cursors/budgets rather than silently reusing unknown audio.
        return state

    def _shutdown(self, command: Command) -> Dict[str, Any]:
        self._auto_snapshot()
        report = self.exit_report(reason="graceful_shutdown")
        return {"command": command.to_dict(), "exit_report": report}

    def exit_report(self, reason: str = "graceful_shutdown") -> Dict[str, Any]:
        result = self._last_result
        return {
            "reason": reason,
            "global_version": self.global_version,
            "quant_level": self._quant_level(),
            "accepted_segments": sorted(self.segments.keys()),
            "completed_segments": list(self.completed_segments),
            "windows_total": len(result.windows) if result else 0,
            "windows_done": len(result.features) if result else 0,
            "endpoint_count": len(result.endpoints) if result else 0,
            "failure_count": len(self.failures),
            "injections_fired": self.harness.report.to_list(),
            "ledger_events": len(self.ledger.events),
            "snapshot_config_hash": config_hash(self.config.to_dict()),
        }

    # --------------------------------------------------------- concurrency
    def run_parallel_features(self, max_workers: Optional[int] = None
                              ) -> PipelineResult:
        """Recompute window features through a bounded thread pool.

        The active worker count never exceeds ``config.max_workers``; this is
        asserted via the pool construction in :func:`windows.extract_all`.
        """
        timeline = Timeline(list(self.segments.values()))
        regions, frames, meta = vad_mod.run_vad(timeline, self.config)
        frame_intervals = [(f.start, f.end) for f in frames]
        windows = win_mod.cut_windows(timeline, self.config, frame_intervals)

        def key_fn(window):
            return self._window_feature_key(window, timeline)

        def hook(window_id, feature):
            return self.harness.window_hook(window_id, feature)

        features = win_mod.extract_all(
            windows, timeline, self.config, frame_intervals,
            key_fn=key_fn,
            max_workers=max_workers or self.config.max_workers,
            feature_hook=hook)
        posteriors = {}
        for window in windows:
            if window.window_id not in features:
                continue
            feature = features[window.window_id]
            posteriors[window.window_id] = post_mod.synthesize_posterior(
                feature, self.config)
        decisions = post_mod.restore_frame_decisions(
            windows, posteriors, frame_intervals, self.config)
        endpoints = post_mod.build_endpoint_intervals(
            decisions, frame_intervals, windows, posteriors)
        result = PipelineResult(
            timeline=timeline, frames=frames, frame_intervals=frame_intervals,
            regions=regions, region_meta=meta, windows=windows,
            features=features, posteriors=posteriors, decisions=decisions,
            endpoints=endpoints,
            keys={"timeline": timeline.fingerprint(),
                  "params": self._param_fp(), "quant": self._quant_fp()},
            failures=list(self.failures))
        self._last_result = result
        return result

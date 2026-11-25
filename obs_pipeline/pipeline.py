"""Pipeline orchestration: run, evolve, invalidate, recompute, persist.

Ties together spec validation, record normalization, deterministic stage
execution, error attribution, drift verdicts, budgeted scheduling and the
SQLite ledger. All paths are relative to the project root.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction

from . import drift as drift_mod
from . import error_model, gen
from .ledger import Ledger
from .scheduler import Scheduler, Task
from .spec import Spec, load_spec, topo_order
from .stages import run_stage


@dataclass
class PipelineReport:
    spec_version: int
    content_digest: str
    stage_results: dict  # (stage_id, window_id) -> StageResult
    metric_stats: dict  # (metric, window_id) -> WindowMetricStats
    error_reports: dict  # metric -> MetricErrorReport
    drift_verdicts: list
    schedule: object
    batch_id: int | None = None
    invalidation: object = None


def content_digest(spec: Spec, records_by_source):
    """Content digest over spec identity-relevant content + normalized records."""
    hasher = hashlib.sha256()
    hasher.update(json.dumps(spec.raw, sort_keys=True).encode("utf-8"))
    for source in spec.sources:
        hasher.update(gen.records_digest(records_by_source[source.id]).encode("utf-8"))
    return hasher.hexdigest()


def load_records(spec: Spec, sample_dir=None):
    """Decode (or deterministically generate) normalized records per source.

    When sample_dir is given, records are read from the on-disk files in
    arrival (out-of-order) order, sorted by obs_seq, then validated; otherwise
    they are generated from the seeds. Both paths yield identical sequences.
    """
    import os

    records_by_source = {}
    for source in spec.sources:
        if sample_dir is not None:
            path = os.path.join(sample_dir, "%s.v%d.rec" % (source.id, source.record_format))
            with open(path, "r", encoding="utf-8") as handle:
                lines = [ln for ln in handle.read().splitlines() if ln.strip()]
            # Simulate out-of-order arrival, then restore canonical order.
            from .records import decode_line

            decoded = []
            for index, line in enumerate(lines):
                decoded.append(decode_line(line, index + 1, source, None))
            decoded.sort(key=lambda r: r.seq)
            from .records import decode_stream

            records = decode_stream([_re_encode(r) for r in decoded], source)
        else:
            records = gen.generate_source_records(source)
        records_by_source[source.id] = records
    return records_by_source


def _re_encode(record):
    from .records import encode_line

    return encode_line(record)


def merged_stream(spec: Spec, records_by_source):
    """Merge per-source records into one deterministic stream ordered by
    (source order in spec, seq). Null positions come from the source bitmaps."""
    merged = []
    for source in spec.sources:
        merged.extend(records_by_source[source.id])
    return merged


def metric_record_scales(records):
    scales = {}
    for record in records:
        scales.setdefault(record.metric_key, record.scale)
    return scales


def run_pipeline(spec: Spec, ledger: Ledger, records_by_source, *,
                 global_budget=10 ** 9, stage_quotas=None, workers=4,
                 affected=None, prior_results=None, prior_stats=None,
                 reference_window_id=None, observation_window_id=None):
    """Execute the pipeline over the task interval.

    affected: optional (stage_ids, window_ids) restriction; windows/stages not
    in the affected set reuse prior_results/prior_stats byte-identically.
    """
    stream = merged_stream(spec, records_by_source)
    interval = spec.task_interval
    assert len(stream) >= interval[1], "records must cover the task interval"
    stream = stream[interval[0]:interval[1]]

    ordered_stages = topo_order(spec.stages)
    quotas = stage_quotas or {s.id: 10 ** 9 for s in ordered_stages}
    windows = spec.windows

    if affected is None:
        affected_stages = {s.id for s in ordered_stages}
        affected_windows = {w.id for w in windows}
    else:
        affected_stages, affected_windows = affected

    stage_results = dict(prior_results or {})
    metric_stats = dict(prior_stats or {})

    # Execute stage-by-stage in topo order; per (stage, window) tasks are
    # scheduled under the budget/quota constraints.
    scheduler = Scheduler(global_budget, quotas, workers=workers)
    last_schedule = None
    stage_inputs = {}
    base_values = tuple(r.value_fixed for r in stream)
    for stage in ordered_stages:
        if stage.upstream:
            parent = stage.upstream[0]
            input_by_window = stage_inputs[parent]
        else:
            input_by_window = {w.id: base_values[w.start - interval[0]:w.end - interval[0]] for w in windows}
        tasks = []
        for window in windows:
            if stage.id in affected_stages and window.id in affected_windows:
                tasks.append(
                    Task(stage.id, stage.order, window.id, window.order,
                         window.end - window.start,
                         payload=input_by_window[window.id])
                )
        last_schedule = scheduler.run(tasks, lambda task: run_stage(
            stage, _window_of(spec, task.window_id), task.payload))
        outputs_by_window = {}
        for window in windows:
            key = (stage.id, window.id)
            if key in last_schedule.results:
                stage_results[key] = last_schedule.results[key]
            outputs_by_window[window.id] = stage_results[key].outputs
        stage_inputs[stage.id] = outputs_by_window

    # Per-metric window statistics over the merged stream.
    bucket_stage = next(s for s in ordered_stages if s.kind == "quantile_bucket")
    metrics = sorted(metric_record_scales(stream))
    for window in windows:
        if window.id not in affected_windows and window.id in {w for (_, w) in metric_stats}:
            continue
        slice_records = stream[window.start - interval[0]:window.end - interval[0]]
        for metric in metrics:
            metric_stats[(metric, window.id)] = drift_mod.compute_window_stats(
                slice_records, window, metric,
                bucket_stage.params["buckets"], bucket_stage.params["lo"],
                bucket_stage.params["hi"],
            )

    # Error attribution per metric.
    scales = metric_record_scales(stream)
    error_reports = error_model.derive_error_model(spec, scales)

    # Drift verdicts: reference window vs observation window (default: first
    # vs last window), per metric.
    ref_id = reference_window_id or windows[0].id
    obs_id = observation_window_id or windows[-1].id
    verdicts = []
    for metric in metrics:
        reference = metric_stats[(metric, ref_id)]
        observation = metric_stats[(metric, obs_id)]
        record_scale = scales[metric]
        # Convert the final-stage bound (final ulp) back to record-scale ulp.
        final_scale = ordered_stages[-1].scale
        bound = error_reports[metric].final_bound_ulp * Fraction(record_scale, final_scale)
        verdicts.append(
            drift_mod.judge_drift(reference, observation, spec.drift_threshold, bound)
        )

    return PipelineReport(
        spec_version=spec.spec_version,
        content_digest=content_digest(spec, records_by_source),
        stage_results=stage_results,
        metric_stats=metric_stats,
        error_reports=error_reports,
        drift_verdicts=verdicts,
        schedule=last_schedule,
    )


def _window_of(spec, window_id):
    for window in spec.windows:
        if window.id == window_id:
            return window
    raise KeyError(window_id)


def commit_report(ledger: Ledger, report: PipelineReport):
    """Persist one report as a single batch commit."""
    batch_id = ledger.begin_batch(report.spec_version, report.content_digest)
    attributions = [
        (metric, error_model.report_to_dict(rep)) for metric, rep in report.error_reports.items()
    ]
    # Progress cursor: (stages completed, windows completed) for this batch.
    cursor = (
        len({key[0] for key in report.stage_results}),
        len({key[1] for key in report.stage_results}),
    )
    ledger.commit_batch(
        batch_id,
        report.spec_version,
        report.content_digest,
        sorted(report.stage_results.values(), key=lambda r: (r.stage_id, r.window_id)),
        [report.metric_stats[k] for k in sorted(report.metric_stats)],
        attributions,
        cursor,
    )
    report.batch_id = batch_id
    return batch_id


@dataclass
class Invalidation:
    affected_stages: tuple
    affected_windows: tuple
    affected_metrics: tuple
    basis: tuple  # (stage_order, window_order, reason) sorted stable

    def as_sets(self):
        return set(self.affected_stages), set(self.affected_windows)


def compute_invalidation(old: Spec, new: Spec):
    """Derive the affected stage/window/metric sets after a spec evolution.

    Rules (propagation basis, emitted in stable (stage_order, window_order)
    order):
      * changed/added/removed stage (params, scale, kind, upstream) -> that
        stage plus its transitive dependents are affected, all windows
      * changed window boundaries -> windows whose [start, end) differ
      * changed/added/removed source -> all windows and that source's metrics
    """
    reasons = []
    old_stages = {s.id: s for s in old.stages}
    new_stages = {s.id: s for s in new.stages}
    changed_stages = set()
    for sid, stage in new_stages.items():
        old_stage = old_stages.get(sid)
        if old_stage is None or (
            old_stage.kind, old_stage.params, old_stage.scale, old_stage.upstream
        ) != (stage.kind, stage.params, stage.scale, stage.upstream):
            changed_stages.add(sid)
            reasons.append((stage.order, -1, "stage %s definition changed" % sid))
    for sid in set(old_stages) - set(new_stages):
        reasons.append((old_stages[sid].order, -1, "stage %s removed" % sid))

    # Downstream closure over the new graph.
    dependents = {s.id: [] for s in new.stages}
    for stage in new.stages:
        for up in stage.upstream:
            dependents[up].append(stage.id)
    affected_stages = set(changed_stages)
    queue = list(changed_stages)
    while queue:
        node = queue.pop(0)
        for dep in dependents.get(node, []):
            if dep not in affected_stages:
                affected_stages.add(dep)
                queue.append(dep)
                reasons.append((new_stages[dep].order, -1,
                                "stage %s downstream of changed stage" % dep))

    old_windows = {w.id: w for w in old.windows}
    new_windows = {w.id: w for w in new.windows}
    affected_windows = set()
    for wid, window in new_windows.items():
        old_window = old_windows.get(wid)
        if old_window is None or (old_window.start, old_window.end) != (window.start, window.end):
            affected_windows.add(wid)
            reasons.append((-1, window.order, "window %s boundaries changed" % wid))
    for wid in set(old_windows) - set(new_windows):
        reasons.append((-1, old_windows[wid].order, "window %s removed" % wid))

    old_sources = {s.id: s for s in old.sources}
    affected_metrics = set()
    sources_changed = False
    for source in new.sources:
        old_source = old_sources.get(source.id)
        if old_source is None or old_source != source:
            sources_changed = True
            affected_metrics.update(source.metrics)
            reasons.append((-1, -1, "source %s declaration changed" % source.id))
    if sources_changed:
        affected_windows.update(w.id for w in new.windows)
        affected_stages.update(s.id for s in new.stages)
    if not affected_metrics:
        affected_metrics.update(m for s in new.sources for m in s.metrics)

    if changed_stages and not affected_windows:
        # Stage-only edits recompute every window of the affected stages.
        affected_windows.update(w.id for w in new.windows)
    if affected_windows and not affected_stages:
        # Window-only edits recompute every stage of the affected windows.
        affected_stages.update(s.id for s in new.stages)

    basis = tuple(sorted(set(reasons)))
    return Invalidation(
        affected_stages=tuple(sorted(affected_stages, key=lambda s: new_stages[s].order)),
        affected_windows=tuple(sorted(affected_windows, key=lambda w: new_windows[w].order)),
        affected_metrics=tuple(sorted(affected_metrics)),
        basis=basis,
    )

"""Command line entry point for the feature/sample pipeline."""
from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Any

from .common import (
    CODE_SNAPSHOT_STALE,
    Failure,
    read_json,
    read_jsonl,
    sha256_text,
    write_json,
    write_jsonl,
    canonical_dumps,
)
from .config import PipelineConfig
from .engine import PipelineEngine
from .snapshot import SnapshotStore
from .verifier import IncrementalVerifier, descendants


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def apply_edits(events, defs, labels, edits, cfg: PipelineConfig):
    """Return (final_events, final_defs, final_labels, final_cfg, touched, changed)."""
    events = {e.event_id: copy.deepcopy(e) for e in events}
    defs = {d.name: copy.deepcopy(d) for d in defs}
    labels = {lb.name: copy.deepcopy(lb) for lb in labels}
    cfg = copy.deepcopy(cfg)
    touched_events: set[str] = set()
    changed_features: set[str] = set()
    changed_fields: dict[str, set[str]] = {}
    backfill_failures: list[Failure] = []
    verifier = IncrementalVerifier(cfg)

    cursor_ts = max((e.ts_ms for e in events.values()), default=-1)
    for edit in sorted(edits["edits"], key=lambda e: e["edit_id"]):
        kind = edit["type"]
        if kind == "correct_event":
            ev = events[edit["event_id"]]
            cursor_event_id = max(events, key=lambda k: events[k].ts_ms)
            if "bad_ts_ms" in edit:
                # Rejected backfill: corrected timestamp is before the cursor.
                attempted = ev.with_correction(ts_ms=int(edit["bad_ts_ms"]))
                failure = verifier.check_backfill_order(
                    list(events.values()), attempted, cursor_event_id)
                if failure is not None:
                    failure = Failure(failure.stage, failure.target,
                                      failure.code,
                                      f"{edit['edit_id']}: {failure.detail}")
                    backfill_failures.append(failure)
                continue  # the bad write is never accepted
            # valid backfill: field-level merge so untouched fields keep values
            merged_context = dict(ev.context)
            actual_changes = {
                k for k, v in edit["context"].items()
                if k not in ev.context or float(ev.context[k]) != float(v)}
            merged_context.update(
                {k: float(v) for k, v in edit["context"].items()})
            corrected = ev.with_correction(
                ts_ms=max(ev.ts_ms, cursor_ts), context=merged_context)
            events[edit["event_id"]] = corrected
            touched_events.add(edit["event_id"])
            changed_fields[edit["event_id"]] = actual_changes
        elif kind == "window_change":
            defs[edit["feature"]].window_ms = int(edit["window_ms"])
            changed_features.add(edit["feature"])
        elif kind == "agg_change":
            defs[edit["feature"]].agg = edit["agg"]
            changed_features.add(edit["feature"])
        elif kind == "label_delay_change":
            labels[edit["label"]].delay_ms = int(edit["delay_ms"])
        elif kind == "cache_budget_change":
            cfg.cache_block_limit = int(edit["cache_block_limit"])
    return (sorted(events.values(), key=lambda e: e.ts_ms),
            list(defs.values()), list(labels.values()), cfg,
            touched_events, changed_features, backfill_failures,
            changed_fields)


def _write_outputs(out_dir: str, tag: str, result: dict[str, Any]) -> None:
    tag_dir = os.path.join(out_dir, tag)
    os.makedirs(tag_dir, exist_ok=True)
    write_jsonl(os.path.join(tag_dir, "features.jsonl"), result["features"])
    write_jsonl(os.path.join(tag_dir, "samples.jsonl"), result["samples"])
    write_jsonl(os.path.join(tag_dir, "budget_ledger.jsonl"), result["ledger"])
    write_jsonl(os.path.join(tag_dir, "failures.jsonl"), result["failures"])
    write_json(os.path.join(tag_dir, "cache.json"), result["cache"])
    if "invalidation_ledger" in result:
        write_json(os.path.join(tag_dir, "invalidation_ledger.json"),
                   result["invalidation_ledger"])
    if "reuse_ledger" in result:
        write_json(os.path.join(tag_dir, "reuse_ledger.json"),
                   result["reuse_ledger"])


def _diagnostics(result: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    cache = result["cache"]
    codes: dict[str, int] = {}
    for fail in result["failures"]:
        codes[fail["code"]] = codes.get(fail["code"], 0) + 1
    injections_hit = sorted({f["injection_id"] for f in result["failures"]
                             if f["injection_id"]})
    return {
        "meta": meta,
        "counts": {
            "feature_rows": len(result["features"]),
            "samples": len(result["samples"]),
            "ready_samples": sum(1 for s in result["samples"] if s["status"] == "ready"),
            "pending_samples": sum(1 for s in result["samples"]
                                   if s["status"] == "pending"),
            "ledger_entries": len(result["ledger"]),
            "failures": len(result["failures"]),
        },
        "failure_codes": codes,
        "injections_hit": injections_hit,
        "cache_telemetry": {
            "hits": cache["hits"],
            "misses": cache["misses"],
            "evictions": cache["evictions"],
            "recomputes": cache["recomputes"],
            "resident_key_count": len(cache["resident_keys"]),
            "registry_key_count": len(cache["registry"]),
        },
        "budget": {
            "global_remaining": result["global_remaining"],
            "feature_used": result["feature_used"],
            "pending_queue_len": len(result["pending_queue"]),
        },
    }


def run_pipeline(data_dir: str, out_dir: str, use_injections: bool,
                 concurrency_override: int | None = None,
                 tag: str = "full") -> dict[str, Any]:
    cfg = PipelineConfig.load(os.path.join(data_dir, "pipeline.yaml"))
    if concurrency_override is not None:
        cfg.thread_concurrency = concurrency_override
    snap = SnapshotStore(os.path.join(out_dir, "snapshot", tag))
    snap.reset()
    engine = PipelineEngine(cfg, snap)
    events, defs, labels, injections = engine.load_inputs(
        os.path.join(data_dir, "events.jsonl"),
        os.path.join(data_dir, "feature_defs.json"),
        os.path.join(data_dir, "label_defs.json"),
        os.path.join(data_dir, "injections.json") if use_injections else None)
    if use_injections:
        events = engine._apply_load_injections(events, injections)
    query_points = read_json(os.path.join(data_dir, "query_points.json"))
    result = engine.run(events, defs, labels, query_points=query_points)
    result["load_failures"] = [f.to_dict() for f in engine.failures
                                if f.stage == "load"]
    # persist the staged load inputs in the snapshot area
    write_jsonl(snap.stage_path("load", "valid_events.jsonl"),
                [e.__dict__ for e in events])
    write_json(snap.stage_path("load", "feature_defs.valid.json"),
               [d.__dict__ for d in defs])
    _write_outputs(out_dir, tag, result)
    write_json(os.path.join(out_dir, tag, "diagnostics.json"),
               _diagnostics(result, {"tag": tag, "concurrency": cfg.thread_concurrency,
                                     "injections": use_injections}))
    return {"cfg": cfg, "events": events, "defs": defs, "labels": labels,
            "result": result, "snapshot": snap, "query_points": query_points}


def snapshot_stale_self_test(out_dir: str, bundle: dict[str, Any]) -> Failure:
    snap: SnapshotStore = bundle["snapshot"]
    expected = snap.load_manifest() or {}
    drifted = dict(expected)
    drifted["definition_hash"] = sha256_text("drifted")
    failure = snap.check_stale(expected, drifted)
    assert failure is not None and failure.code == CODE_SNAPSHOT_STALE
    write_json(os.path.join(out_dir, "snapshot_stale_check.json"),
               {"status": "detected", "failure": failure.to_dict()})
    return failure


def run_resume(data_dir: str, out_dir: str) -> dict[str, Any]:
    """Resume the 'full' snapshot and verify equivalence with a fresh run."""
    from .recovery import verify_manifest, resume_state, RecoveryError
    from .loader import load_feature_defs

    cfg = PipelineConfig.load(os.path.join(data_dir, "pipeline.yaml"))
    snap = SnapshotStore(os.path.join(out_dir, "snapshot", "full"))
    engine = PipelineEngine(cfg, snap)
    events, defs, labels, injections = engine.load_inputs(
        os.path.join(data_dir, "events.jsonl"),
        os.path.join(data_dir, "feature_defs.json"),
        os.path.join(data_dir, "label_defs.json"),
        os.path.join(data_dir, "injections.json"))
    events = engine._apply_load_injections(events, injections)

    def_hash = engine._definition_hash(defs)
    cfg_hash = engine._config_hash()
    try:
        manifest = verify_manifest(snap, def_hash, cfg_hash)
    except RecoveryError as exc:
        write_json(os.path.join(out_dir, "resume_report.json"),
                   {"status": "stale", "failure": exc.failure.to_dict()})
        return {"status": "stale", "failure": exc.failure.to_dict()}

    state = resume_state(snap)
    engine._resume_rows = read_jsonl(os.path.join(
        snap.stage_dir, "feature", "features.jsonl"))
    ledger_path = os.path.join(snap.stage_dir, "feature", "budget_ledger.jsonl")
    engine._resume_ledger = (read_jsonl(ledger_path)
                             if os.path.exists(ledger_path) else [])
    engine._resume_cursor_seq = state["cursor_seq"]
    query_points = read_json(os.path.join(data_dir, "query_points.json"))
    resumed = engine.run(events, defs, labels, query_points=query_points,
                         resume_after_seq=state["cursor_seq"])
    # Rebuild the full ordered sample set from the now-complete observation
    # history and the restored abort set.
    from .sample import SampleGenerator
    feature_side = {d.name: d.side for d in defs if d.valid}
    sampler = SampleGenerator(labels, engine._computer, cfg.fp.scale,
                              cfg.quant_step)
    aborted = set(engine._scheduler.aborted)
    rebuilt_samples: list = []
    sample_failures: list = []
    event_point_seq = {ev.event_id: idx for idx, ev in enumerate(events)}
    for ev in events:
        new_samples, fails = sampler.evaluate(
            ev, event_point_seq[ev.event_id], feature_side, aborted,
            event_point_seq=event_point_seq[ev.event_id])
        rebuilt_samples.extend(new_samples)
        sample_failures.extend(fails)
    engine.samples = rebuilt_samples
    resumed["samples"] = [s.to_dict() for s in rebuilt_samples]
    # Rebuild the canonical failure list from three deterministic sources:
    # load-stage validation/injection faults, feature budget aborts and
    # sample-stage pending faults.
    load_failures = [f for f in engine.failures if f.stage == "load"]
    all_failures = list(load_failures)
    # scheduler.aborted is keyed by feature, so each abort appears once,
    # matching the fresh run's de-duplicated failure collection.
    all_failures.extend(engine._scheduler.aborted.values())
    all_failures.extend(sample_failures)
    seen: set = set()
    deduped: list = []
    for fail in all_failures:
        key = (fail.stage, fail.target, fail.code, fail.detail, fail.injection_id)
        if key not in seen:
            seen.add(key)
            deduped.append(fail)
    all_failures = deduped
    all_failures.sort(key=lambda f: (f.stage, f.target, f.code,
                                     f.detail, f.injection_id))
    engine.failures = all_failures
    resumed["failures"] = [f.to_dict() for f in all_failures]
    _write_outputs(out_dir, "resumed", resumed)
    # Re-read the canonical, sorted artifact so the in-memory comparison uses
    # exactly the same ordering/content as the persisted output.
    resumed["failures"] = read_jsonl(os.path.join(
        out_dir, "resumed", "failures.jsonl"))

    fresh_snap = SnapshotStore(os.path.join(out_dir, "snapshot", "resume_fresh"))
    fresh_snap.reset()
    fresh_engine = PipelineEngine(cfg, fresh_snap)
    # Mirror the same injection-isolated load-stage failures so both runs
    # share an identical starting state.
    fresh_engine.failures.extend(f for f in engine.failures if f.stage == "load")
    fresh = fresh_engine.run(
        events, defs, labels, query_points=query_points)
    verifier = IncrementalVerifier(cfg)
    ok, diff, _ = verifier.compare_runs(
        fresh, resumed, os.path.join(out_dir, "resume_equivalence.json"))
    report = {"status": "resumed" if ok else "mismatch",
              "resumed_from_seq": state["cursor_seq"],
              "manifest_version": manifest["global_version"],
              "first_diff": diff}
    write_json(os.path.join(out_dir, "resume_report.json"), report)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="featflow pipeline runner")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-injections", action="store_true")
    ap.add_argument("--serial", action="store_true",
                    help="additional concurrency=1 consistency run")
    ap.add_argument("--mode", choices=["run", "resume"], default="run")
    args = ap.parse_args()
    if args.mode == "resume":
        report = run_resume(args.data, args.out)
        print(canonical_dumps(report))
        return 0 if report["status"] == "resumed" else 2

    os.makedirs(args.out, exist_ok=True)
    bundle = run_pipeline(args.data, args.out,
                          use_injections=not args.no_injections, tag="full")
    cfg = bundle["cfg"]

    # ---- incremental edit sequence vs full rebuild on final inputs ----
    # Both branches consume the *same* injection-isolated event set, so the
    # only differences they may exhibit come from the edit sequence itself.
    edits = read_json(os.path.join(args.data, "edits.json"))
    (final_events, final_defs, final_labels, final_cfg, touched, changed,
     backfill_failures, changed_fields) = apply_edits(
        bundle["events"], bundle["defs"], bundle["labels"], edits, cfg)

    # Recompute topological ranks after definition edits (same algorithm as loader).
    from .loader import load_feature_defs

    tmp_defs = os.path.join(args.out, "_final_defs.json")
    write_json(tmp_defs, [_def_to_row(d) for d in final_defs])
    final_defs, def_failures = load_feature_defs(tmp_defs)

    verifier = IncrementalVerifier(final_cfg)
    verifier._changed_fields = changed_fields
    invalid = verifier.invalidation_set(final_defs, final_events, touched, changed)

    snap_full = SnapshotStore(os.path.join(args.out, "snapshot", "final_full"))
    snap_full.reset()
    full_engine = PipelineEngine(final_cfg, snap_full)
    full_result = full_engine.run(final_events, final_defs, final_labels,
                                  query_points=bundle["query_points"])
    full_result["load_failures"] = bundle["result"]["load_failures"]
    full_engine.failures.extend(
        Failure(**row) for row in bundle["result"]["load_failures"])
    full_engine.failures.sort(key=lambda f: (f.stage, f.target, f.code,
                                             f.detail, f.injection_id))
    full_result["failures"] = [f.to_dict() for f in full_engine.failures]
    _write_outputs(args.out, "final_full", full_result)

    snap_inc = SnapshotStore(os.path.join(args.out, "snapshot", "incremental"))
    snap_inc.reset()
    inc_result = verifier.replay_incremental(
        bundle["result"], final_defs, final_events, final_labels,
        bundle["query_points"], invalid, snap_inc)
    _write_outputs(args.out, "incremental", inc_result)

    equivalent, diff, verify_failures = verifier.compare_runs(
        full_result, inc_result,
        os.path.join(args.out, "increment_equivalence.json"))

    # ---- serial vs concurrent consistency ----
    serial_info = None
    if args.serial:
        serial_bundle = run_pipeline(args.data, args.out,
                                     use_injections=not args.no_injections,
                                     concurrency_override=1, tag="serial")
        concurrent_bundle = run_pipeline(args.data, args.out,
                                         use_injections=not args.no_injections,
                                         concurrency_override=max(2, cfg.thread_concurrency),
                                         tag="concurrent")
        serial_info = {
            "features_equal": serial_bundle["result"]["features"] ==
            concurrent_bundle["result"]["features"],
            "samples_equal": serial_bundle["result"]["samples"] ==
            concurrent_bundle["result"]["samples"],
            "ledger_equal": serial_bundle["result"]["ledger"] ==
            concurrent_bundle["result"]["ledger"],
            "failures_equal": serial_bundle["result"]["failures"] ==
            concurrent_bundle["result"]["failures"],
        }
        write_json(os.path.join(args.out, "concurrency_consistency.json"), serial_info)

    stale_failure = snapshot_stale_self_test(args.out, bundle)

    report = {
        "equivalent": equivalent,
        "first_diff": diff,
        "invalidated_features": sorted(invalid),
        "backfill_order_rejected": [f.to_dict() for f in backfill_failures],
        "snapshot_stale": stale_failure.to_dict(),
        "serial_vs_concurrent": serial_info,
        "full_diagnostics": _diagnostics(
            bundle["result"], {"tag": "full", "concurrency": cfg.thread_concurrency}),
        "incremental_diagnostics": _diagnostics(
            inc_result, {"tag": "incremental"}),
    }
    write_json(os.path.join(args.out, "run_report.json"), report)
    print(canonical_dumps({"equivalent": equivalent, "first_diff": diff}))
    return 0 if equivalent else 2


def _def_to_row(d) -> dict[str, Any]:
    return {"name": d.name, "side": d.side, "window_ms": d.window_ms,
            "agg": d.agg, "source_field": d.source_field, "upstream": d.upstream,
            "empty_value": d.empty_value}


if __name__ == "__main__":
    raise SystemExit(main())

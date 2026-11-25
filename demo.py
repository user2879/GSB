"""End-to-end demo: deterministic samples, three-format normalization,
layered fixed-point attribution, invalidation recompute and kill-recovery
replay. Run from the project root:  python3 demo.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from obs_pipeline import gen
from obs_pipeline.error_model import compare_versions, report_to_dict
from obs_pipeline.ledger import Ledger
from obs_pipeline.pipeline import (
    commit_report,
    compute_invalidation,
    content_digest,
    load_records,
    run_pipeline,
)
from obs_pipeline.spec import load_spec, validate_spec

SAMPLES = "samples"
LEDGER_PATH = os.path.join(SAMPLES, "ledger.sqlite")


def banner(title):
    print("\n=== %s ===" % title)


def main():
    banner("1. generate deterministic samples (seed=42) and validate spec")
    spec_dict = gen.build_sample_spec(seed=42)
    gen.write_sample_tree(spec_dict, SAMPLES)
    spec = load_spec(os.path.join(SAMPLES, "spec.json"))
    print("spec_version=%d stages=%s windows=%s" % (
        spec.spec_version,
        [s.id for s in spec.stages],
        [(w.id, w.start, w.end) for w in spec.windows],
    ))

    banner("2. three-format normalization (v1/v2/v3) with fill provenance")
    records_by_source = load_records(spec, sample_dir=SAMPLES)
    for source in spec.sources:
        first = records_by_source[source.id][0]
        print("source=%s format=v%d filled=%s rationale=%s" % (
            source.id, first.version, list(first.filled_fields),
            list(first.fill_rationale)[:1],
        ))
        print("  digest=%s" % gen.records_digest(records_by_source[source.id])[:16])

    banner("3. run pipeline: fixed-point stages, null rules, scheduling")
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)
    ledger = Ledger(LEDGER_PATH)
    digest = content_digest(spec, records_by_source)
    ledger.open_snapshot(spec.spec_version, digest)
    quotas = {s.id: 40 for s in spec.stages}
    report = run_pipeline(spec, ledger, records_by_source,
                          global_budget=80, stage_quotas=quotas, workers=4)
    for stage in spec.stages:
        for window in spec.windows:
            result = report.stage_results[(stage.id, window.id)]
            print("stage=%-8s window=%s nulls=%2d summary=%s" % (
                stage.id, window.id, result.null_count, result.summary[:12]))
        print("  semantics=%s" % json.dumps(result.semantics, ensure_ascii=False))
        print("  null_rule=%s" % result.null_rule)
    sched = report.schedule
    print("schedule: budget=%d quotas=%s peak_global=%d peak_stage=%s total_queue_wait_ms=%d" % (
        sched.global_budget, sched.stage_quotas, sched.peak_global_points,
        sched.peak_stage_points, sched.total_queue_wait_ms))

    banner("4. error attribution per metric (dominant stage, explicit threshold)")
    for metric, rep in sorted(report.error_reports.items()):
        print("metric=%s bound=%s ulp dominant=%s (threshold=%d ppm)" % (
            metric, rep.final_bound_ulp, rep.dominant_stage,
            rep.dominance_threshold_ppm))
        for sid, share in rep.shares:
            print("  share %-8s %s" % (sid, share))

    banner("5. drift verdicts (reference w0 vs observation w4)")
    for verdict in report.drift_verdicts:
        print("metric=%s drift=%s threshold=%s bound=%s cause=%s exceeded=%s error_affected=%s" % (
            verdict.metric_key, verdict.drift_amount, verdict.threshold,
            verdict.error_bound, verdict.cause, verdict.exceeded,
            verdict.error_affected))

    commit_report(ledger, report)
    print("committed batch=%d" % report.batch_id)

    banner("6. evolve spec (window_avg window 4->6, scale change) and recompute only the affected subset")
    evolved = json.loads(json.dumps(spec_dict))
    evolved["spec_version"] = 2
    evolved["stages"][1]["params"]["window"] = 6
    evolved["stages"][1]["scale"] = 1000
    new_spec = validate_spec(evolved)
    invalidation = compute_invalidation(spec, new_spec)
    print("affected_stages=%s" % (invalidation.affected_stages,))
    print("affected_windows=%s" % (invalidation.affected_windows,))
    for reason in invalidation.basis:
        print("  basis stage_order=%s window_order=%s %s" % reason)
    new_records = load_records(new_spec, sample_dir=SAMPLES)
    new_digest = content_digest(new_spec, new_records)
    ledger.open_snapshot(new_spec.spec_version, new_digest)
    rerun = run_pipeline(new_spec, ledger, new_records,
                         global_budget=80, stage_quotas={s.id: 40 for s in new_spec.stages},
                         affected=invalidation.as_sets(),
                         prior_results=report.stage_results,
                         prior_stats=report.metric_stats)
    unchanged = [
        key for key in rerun.stage_results
        if key in report.stage_results
        and rerun.stage_results[key].summary == report.stage_results[key].summary
    ]
    print("unaffected (byte-identical) stage summaries: %s" % sorted(unchanged))
    old_attr = ledger.load_error_attributions(spec_version=1)
    commit_report(ledger, rerun)
    print("old-version attributions kept for comparison: %s" % sorted(old_attr))
    comparison = compare_versions(report.error_reports, rerun.error_reports)
    for row in comparison:
        print("  compare %s" % row)

    banner("7. forced-kill recovery: uncommitted batch is unknown, replay matches byte-for-byte")
    proc = subprocess.run([sys.executable, "tools/crash_writer.py", LEDGER_PATH,
                           str(new_spec.spec_version), new_digest])
    assert proc.returncode == 137, "crash writer must be killed (no exit report)"
    ledger.close()
    ledger = Ledger(LEDGER_PATH)
    status = ledger.recovery_status()
    print("recovery status: %s" % status)
    replay = run_pipeline(new_spec, ledger, new_records,
                          global_budget=80, stage_quotas={s.id: 40 for s in new_spec.stages},
                          affected=invalidation.as_sets(),
                          prior_results=report.stage_results,
                          prior_stats=report.metric_stats)
    mismatch = ledger.verify_stage_summaries(replay.stage_results)
    print("ledger replay mismatch: %s" % (mismatch,))
    ledger.close()
    print("\nDEMO OK")


if __name__ == "__main__":
    main()

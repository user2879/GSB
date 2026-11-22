"""End-to-end acceptance demo (local, deterministic, no network).

Run with::

    python run_demo.py

It regenerates the fixed-seed samples and exercises three acceptance
scenarios, writing JSON artefacts under ``out/`` and a human readable summary
to stdout:

1. incremental replay vs full rebuild field-level equivalence across an
   append / replace / delete edit sequence;
2. global-budget exhaustion, retained todos and deterministic resume;
3. staged snapshot recovery, including stale-version detection.
"""

import json
import shutil
from pathlib import Path

from dialog_state_pipeline.config import load_config, load_fsm
from dialog_state_pipeline.injection import load_injection
from dialog_state_pipeline.pipeline import Pipeline
from dialog_state_pipeline.report import comparison_view, write_report
from dialog_state_pipeline.sample_data import write_sample_dir
from dialog_state_pipeline.snapshot import SnapshotStore

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "out"


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _pipeline(sample_dir, work_dir, config_name, *, refill=False,
              injection=False, tag="run"):
    config = load_config(sample_dir / config_name)
    config.snapshot_dir = str(work_dir / f"snapshot_{tag}")
    fsm = load_fsm(sample_dir / "fsm.yaml")
    injections = (load_injection(sample_dir / "injection_config.json")
                  if injection else {"injections": []})
    return Pipeline(config, fsm, injections,
                    budget_refill_on_resume=refill), config, fsm


def scenario_equivalence(sample_dir, work_dir):
    print("\n[1] incremental vs full replay equivalence")
    base = _read_jsonl(sample_dir / "events_base.jsonl")
    edited = _read_jsonl(sample_dir / "events.jsonl")
    stream = work_dir / "stream.jsonl"
    _write_jsonl(stream, base)

    incremental, _cfg, _fsm = _pipeline(sample_dir, work_dir,
                                        "pipeline_config.json", tag="inc")
    incremental.store.reset()
    incremental.run(stream, None)
    _write_jsonl(stream, edited)
    report_inc = incremental.run(stream, None)

    full, _cfg, _fsm = _pipeline(sample_dir, work_dir,
                                 "full_replay_config.json", tag="full")
    full.store.reset()
    report_full = full.run(stream, work_dir / "report_full.json", mode="full")
    write_report(report_inc, work_dir / "report_incremental.json")

    equal = comparison_view(report_inc) == comparison_view(report_full)
    reused = {sid: s["reused_turns"] for sid, s in
              report_inc["sessions"].items() if s["reused_turns"]}
    print(f"    field-level identical: {equal}")
    print(f"    reused cached turns  : {reused}")
    print(f"    fingerprints: inc={report_inc['report_fingerprint'][:16]} "
          f"full={report_full['report_fingerprint'][:16]}")
    assert equal
    return equal


def scenario_budget(sample_dir, work_dir):
    print("\n[2] budget exhaustion, todo retention and resume")
    pipeline, _cfg, _fsm = _pipeline(sample_dir, work_dir,
                                     "budget_config.json", refill=True,
                                     injection=True, tag="budget")
    pipeline.store.reset()
    events = sample_dir / "events.jsonl"
    first = pipeline.run(events, None)
    print(f"    run #1: consumed={first['budget']['consumed']} "
          f"remaining={first['budget']['remaining']} "
          f"pending={first['budget']['pending_sessions']}")
    report = first
    runs = 1
    while report["budget"]["pending_sessions"] and runs < 20:
        report = pipeline.run(events, None)
        runs += 1
        print(f"    run #{runs}: consumed={report['budget']['consumed']} "
              f"pending={report['budget']['pending_sessions']}")
    write_report(report, work_dir / "report_budget_resume.json")
    seqs = [row["seq"] for row in report["budget"]["deduction_sequence"]]
    auditable = seqs == list(range(1, len(seqs) + 1))
    print(f"    ledger gap-free and auditable: {auditable}")
    print(f"    injection hits: "
          f"{sorted(report['injections']['hits'].keys())}")
    assert not report["budget"]["pending_sessions"] and auditable
    return report


def scenario_snapshot(sample_dir, work_dir):
    print("\n[3] staged snapshot recovery and staleness detection")
    pipeline, config, fsm = _pipeline(sample_dir, work_dir,
                                      "pipeline_config.json", tag="snap")
    pipeline.store.reset()
    events = sample_dir / "events.jsonl"
    pipeline.run(events, None, stop_after_stage="load")
    store = SnapshotStore(config.snapshot_dir)
    after_load = store.load()
    print(f"    after load stage   : stage={after_load.stage} "
          f"symlink={store.committed.is_symlink()}")
    pipeline.run(events, None, stop_after_stage="advance")
    after_advance = store.load()
    print(f"    after advance stage: checkpoints="
          f"{len(after_advance.checkpoints)} pending="
          f"{after_advance.pending_sessions}")
    report = pipeline.run(events, work_dir / "report_snapshot.json")
    settled = store.load()
    print(f"    after settle stage : completed_batches="
          f"{settled.completed_batches} global_version="
          f"{settled.global_version}")

    changed = load_config(sample_dir / "pipeline_config.json")
    changed.snapshot_dir = config.snapshot_dir
    changed.max_workers += 1
    stale_pipeline = Pipeline(changed, fsm, {"injections": []})
    stale_report = stale_pipeline.run(events, None)
    event = stale_report.get("snapshot_event", {})
    print(f"    config change -> {event.get('code')}: {event.get('message')}")
    assert event.get("code") == "E_SNAPSHOT_STALE"
    return report


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    sample_dir = SAMPLES if (SAMPLES / "fsm.yaml").exists() else write_sample_dir(SAMPLES)
    write_sample_dir(SAMPLES)
    work_dir = OUT / "demo_work"
    work_dir.mkdir(parents=True)
    scenario_equivalence(sample_dir, work_dir)
    scenario_budget(sample_dir, work_dir)
    scenario_snapshot(sample_dir, work_dir)
    print("\nAll acceptance scenarios completed. Artefacts in out/")


if __name__ == "__main__":
    main()


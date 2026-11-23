"""Reproducible end-to-end demonstration on local deterministic samples.

Scenarios:
  1. generate samples and run the full pipeline (dedup + correlation + stats);
  2. forced-kill injection -> UNKNOWN batch -> restart -> replay verification;
  3. spec evolution (version bump) -> staleness + whole-view replacement diff;
  4. three fault injections: graph cycle (E_SPEC), negative silence window
     (E_FIELD), ledger tampering (E_SNAPSHOT).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from alert_pipeline import samplegen
from alert_pipeline.errors import (FieldError, SnapshotError, SpecError)
from alert_pipeline.ledger import Ledger
from alert_pipeline.pipeline import Pipeline, PipelineConfig
from alert_pipeline.util import write_json

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "artifacts" / "demo_work"
SAMPLES = ROOT / "samples"


def fresh_dirs(tag: str) -> tuple[Path, Path]:
    d = WORK / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d, d / "logs"


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    paths = samplegen.generate_all(SAMPLES, seed=20261124)
    samplegen.generate_injected(SAMPLES)
    report: dict = {}

    # Scenario 1: normal full run ------------------------------------------------
    run_dir, log_dir = fresh_dirs("normal")
    cfg = PipelineConfig(
        graph_path=paths["graph"], rules_path=paths["rules"],
        stream_path=paths["stream"],
        ledger_path=str(run_dir / "ledger.sqlite"),
        log_dir=str(log_dir), batch_size=12, global_limit=4,
        per_service_limit=2, task_timeout_ms=2000)
    pipe = Pipeline(cfg)
    report["normal_run"] = pipe.run()
    pipe.write_run_report(report["normal_run"], run_dir / "run_report.json")
    pipe.close()

    # Scenario 2: forced kill then recovery -------------------------------------
    crash_dir, crash_logs = fresh_dirs("crash")
    crash_cfg = PipelineConfig(
        graph_path=paths["graph"], rules_path=paths["rules"],
        stream_path=paths["stream"],
        ledger_path=str(crash_dir / "ledger.sqlite"),
        log_dir=str(crash_logs), batch_size=12)
    pipe = Pipeline(crash_cfg)
    crashed = pipe.run(crash_at_batch=2)
    pipe.close()
    report["crash"] = {"crashed_run": crashed}

    pipe = Pipeline(crash_cfg)  # restart: only committed rows are trusted
    recovered = pipe.resume()
    pipe.write_run_report(recovered, crash_dir / "recovery_report.json")
    pipe.close()
    report["crash"]["recovery"] = recovered

    # Scenario 3: evolution with version bump -----------------------------------
    evolved_dir, evolved_logs = fresh_dirs("evolution")
    evo_cfg = PipelineConfig(
        graph_path=paths["graph"], rules_path=paths["rules"],
        stream_path=paths["stream"],
        ledger_path=str(evolved_dir / "ledger.sqlite"),
        log_dir=str(evolved_logs), batch_size=12)
    pipe = Pipeline(evo_cfg)
    pipe.run()
    evolved_rules = json.loads(Path(paths["rules"]).read_text(encoding="utf-8"))
    evolved_rules["declared_version"] = 2
    evolved_rules["window_version"] = 2
    evolved_rules["default_silence_window_ms"] = 2500
    new_rules_path = evolved_dir / "rules_v2.json"
    write_json(new_rules_path, evolved_rules)
    evolution = pipe.evolve(paths["graph"], str(new_rules_path),
                            paths["stream"])
    pipe.write_run_report(evolution, evolved_dir / "evolution_report.json")
    pipe.close()
    report["evolution"] = {
        "view_diff": evolution["view_diff"],
        "stale_batches_marked": evolution["stale_batches_marked"],
    }

    # Scenario 4: fault injections ----------------------------------------------
    injections: dict = {}
    # 4a. cyclic graph -> E_SPEC
    try:
        bad_cfg = PipelineConfig(
            graph_path=str(SAMPLES / "injected" / "graph_cycle.json"),
            rules_path=paths["rules"], stream_path=paths["stream"],
            ledger_path=str(evolved_dir / "unused_cycle.sqlite"))
        Pipeline(bad_cfg)
    except SpecError as exc:
        injections["graph_cycle"] = {"code": exc.code, "message": str(exc)}

    # 4b. negative silence window -> E_FIELD
    try:
        bad_cfg = PipelineConfig(
            graph_path=paths["graph"], rules_path=paths["rules"],
            stream_path=str(SAMPLES / "injected" /
                            "alerts_negative_window.ndjson"),
            ledger_path=str(evolved_dir / "unused_field.sqlite"))
        Pipeline(bad_cfg)
    except FieldError as exc:
        injections["negative_silence_window"] = {
            "code": exc.code, "message": str(exc)}

    # 4c. ledger tampering -> E_SNAPSHOT
    snap_dir, snap_logs = fresh_dirs("snapshot")
    snap_cfg = PipelineConfig(
        graph_path=paths["graph"], rules_path=paths["rules"],
        stream_path=paths["stream"],
        ledger_path=str(snap_dir / "ledger.sqlite"),
        log_dir=str(snap_logs), batch_size=12)
    pipe = Pipeline(snap_cfg)
    pipe.run()
    pipe.close()
    ledger = Ledger(snap_cfg.ledger_path)
    first_batch = ledger.conn.execute(
        "SELECT MIN(batch_id) FROM aggregates").fetchone()[0]
    ledger.corrupt_for_injection("aggregates", first_batch)
    ledger.close()
    try:
        pipe = Pipeline(snap_cfg)
        pipe.run()
    except SnapshotError as exc:
        injections["ledger_tamper"] = {"code": exc.code,
                                       "message": str(exc)}
    report["injections"] = injections

    write_json(WORK / "demo_summary.json", report)
    print(json.dumps({
        "normal_aggregates": report["normal_run"]["total_aggregates"],
        "normal_suppressed": report["normal_run"]["total_suppressed"],
        "recovery_status": report["crash"]["recovery"]["status"],
        "unknown_reran": report["crash"]["recovery"]["reran_unknown_batches"],
        "view_diff": report["evolution"]["view_diff"],
        "injection_codes": {k: v["code"] for k, v in injections.items()},
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

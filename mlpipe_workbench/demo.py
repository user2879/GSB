"""Repeatable three-scenario demonstration with a stable line ordering."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from .demo_data import write_demo_inputs
from .engine import Engine
from .ledger import Ledger
from .loader import load_workspace
from .logs import JsonlLogger
from .recovery import RecoveryManager

ROOT = Path(__file__).resolve().parents[1]


def _fresh_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_demo(out_dir: Path) -> str:
    out_dir = out_dir.resolve()
    if out_dir in (ROOT, ROOT.parent, Path.home(), Path(out_dir.anchor)):
        raise ValueError("demo output directory is too broad; choose a dedicated subdirectory")
    _fresh_dir(out_dir)
    runtime_dir = out_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    evolved_dir = out_dir / "evolved"
    evolved_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "demo.jsonl"
    logger = JsonlLogger(log_path)
    lines: list[str] = []

    data_path = ROOT / "data" / "sample.json"
    pipeline_path = ROOT / "data" / "pipeline.json"
    if not data_path.exists() or not pipeline_path.exists():
        write_demo_inputs(ROOT / "data", evolved=False)

    # Scenario 1: cold start.
    ledger_path = runtime_dir / "demo.db"
    prepared = load_workspace(data_path, pipeline_path)
    with Ledger(ledger_path, logger) as ledger:
        cold = Engine(ledger).run(
            prepared,
            run_id="demo-cold",
            batch_size=32,
            command={"command": "demo", "scenario": "cold_start"},
        )
        view = ledger.current_conclusion()["view"]
    lines.extend(
        [
            "SCENARIO 1 COLD_START",
            f"status={cold.status}",
            f"run_id={cold.run_id}",
            f"spec_version={cold.spec_version}",
            f"fingerprint={cold.fingerprint}",
            f"rows={view['row_count']}",
            f"stages={','.join(stage['stage_id'] + ':' + stage['status'] for stage in view['stages'])}",
            f"shards={len(cold.shards)}",
            "shard_ranges=" + ",".join(f"{s['start_row']}-{s['end_row']}" for s in cold.shards),
            f"final_data_digest={cold.final_data_digest}",
            f"cold_start_elapsed_ms={cold.elapsed_ms}",
            "END SCENARIO 1",
            "",
        ]
    )

    # Scenario 2: specification and sample evolution, whole-view replacement.
    evolved_data, evolved_pipeline = write_demo_inputs(evolved_dir, evolved=True)
    evolved_prepared = load_workspace(evolved_data, evolved_pipeline)
    with Ledger(ledger_path, logger) as ledger:
        evolved = Engine(ledger).run(
            evolved_prepared,
            run_id="demo-evolved",
            batch_size=32,
            command={"command": "demo", "scenario": "evolution"},
        )
        conclusion = ledger.current_conclusion()
        current_fp = conclusion["fingerprint"]
    report = evolved.evolution or {}
    config_changed = [stage["stage_id"] for stage in report.get("config_changed_stages", [])]
    affected = [stage["stage_id"] for stage in report.get("affected_stages", [])]
    lines.extend(
        [
            "SCENARIO 2 SPEC_EVOLUTION",
            f"status={evolved.status}",
            f"old_spec_version={report.get('old_spec_version')}",
            f"new_spec_version={evolved.spec_version}",
            f"old_fingerprint={report.get('old_fingerprint')}",
            f"new_fingerprint={evolved.fingerprint}",
            f"dataset_changed={str(report.get('dataset_changed')).lower()}",
            f"config_changed_stages={','.join(config_changed)}",
            f"affected_stages={','.join(affected)}",
            f"old_shards={report.get('old_shard_count')}",
            f"new_shards={report.get('new_shard_count')}",
            f"invalidated_shards={report.get('invalidated_shard_count')}",
            f"preserved_shards={report.get('preserved_shard_count')}",
            "preserved_shard_indices="
            + ",".join(str(index) for index in report.get("preserved_shard_indices", [])),
            f"unaffected_outputs_stable={str(report.get('unaffected_outputs_stable')).lower()}",
            f"view_replaced=true current_fingerprint={current_fp}",
            "END SCENARIO 2",
            "",
        ]
    )

    # Scenario 3: abrupt termination after a committed stage, then recover and
    # replay the same run.
    crash_db = runtime_dir / "crash.db"
    command = [
        sys.executable,
        "-m",
        "mlpipe_workbench",
        "run",
        "--data",
        str(data_path),
        "--pipeline",
        str(pipeline_path),
        "--db",
        str(crash_db),
        "--log",
        str(out_dir / "crash.jsonl"),
        "--run-id",
        "demo-crash",
        "--batch-size",
        "32",
        "--crash-after-stage",
        "fill_missing",
    ]
    crashed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    with Ledger(crash_db, logger) as ledger:
        run_before = ledger.get_run("demo-crash")
        recovery = RecoveryManager(ledger).recover("demo-crash")
        run_after = ledger.get_run("demo-crash")
    inspection = recovery["inspection"]
    replay = recovery["replay"]
    lines.extend(
        [
            "SCENARIO 3 CRASH_RECOVERY",
            f"crash_exit_code={crashed.returncode}",
            f"status_before_recovery={run_before['status'] if run_before else 'MISSING'}",
            f"exit_report_available={str(inspection.get('exit_report_available')).lower()}",
            "recovered_stages="
            + ",".join(stage["stage_id"] for stage in inspection.get("recovered_stages", [])),
            "unknown_stages="
            + ",".join(stage["stage_id"] for stage in inspection.get("unknown_stages", [])),
            "rerun_plan=" + ",".join(inspection.get("rerun_plan", [])),
            f"replay_status={replay.get('status')}",
            f"stages_verified={replay.get('stages_verified')}",
            f"shards_verified={replay.get('shards_verified')}",
            f"final_data_digest={replay.get('final_data_digest')}",
            f"matches_cold_final={str(replay.get('final_data_digest') == cold.final_data_digest).lower()}",
            f"status_after_recovery={run_after['status'] if run_after else 'MISSING'}",
            "END SCENARIO 3",
            "",
        ]
    )

    summary = "\n".join(lines).rstrip() + "\n"
    (out_dir / "demo_summary.txt").write_text(summary, encoding="utf-8", newline="\n")
    (out_dir / "demo_summary.json").write_text(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "summary": summary.splitlines(),
                "ledger": str(ledger_path),
                "crash_ledger": str(crash_db),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary

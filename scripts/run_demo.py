from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "var" / "demo"
DATA = ROOT / "data"
PARAMS = DATA / "params"


def reset_demo_directory() -> None:
    allowed_parent = (ROOT / "var").resolve()
    if DEMO.exists():
        resolved = DEMO.resolve()
        if resolved.parent != allowed_parent or resolved.name != "demo":
            raise RuntimeError(f"refusing to remove unexpected path: {resolved}")
        shutil.rmtree(DEMO)
    DEMO.mkdir(parents=True, exist_ok=False)


def run_cli(arguments: list[str], *, expected_exit: int = 0) -> tuple[dict[str, Any], int]:
    command = [sys.executable, "-m", "columnar_pipeline", *arguments]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != expected_exit:
        raise RuntimeError(
            f"command returned {completed.returncode}, expected {expected_exit}: "
            f"{' '.join(command)}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"command produced no JSON output: {' '.join(command)}")
    return json.loads(lines[-1]), completed.returncode


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    reset_demo_directory()
    ledger = DEMO / "ledger.sqlite"
    snapshots = DEMO / "snapshots"
    log_file = DEMO / "pipeline.jsonl"
    scenarios = [
        ("small_no_skew", DATA / "small.json", PARAMS / "small.json", "01_small_no_skew"),
        ("skew_amplified", DATA / "skew_amplified.json", PARAMS / "skew.json", "02_skew_amplified"),
        ("tight_budget", DATA / "skew_amplified.json", PARAMS / "tight_budget.json", "03_tight_budget"),
    ]
    records: list[dict[str, Any]] = []
    result_bytes: dict[str, bytes] = {}
    for name, data_path, config_path, directory_name in scenarios:
        out_dir = DEMO / directory_name
        summary, exit_code = run_cli(
            [
                "run",
                "--data",
                str(data_path),
                "--config",
                str(config_path),
                "--out-dir",
                str(out_dir),
                "--ledger",
                str(ledger),
                "--snapshot-dir",
                str(snapshots),
                "--log-file",
                str(log_file),
            ]
        )
        result_file = Path(summary["result_file"])
        result_bytes[name] = result_file.read_bytes()
        replay, replay_exit = run_cli(
            [
                "replay",
                "--snapshot",
                summary["snapshot_path"],
                "--data",
                str(data_path),
                "--ledger",
                str(ledger),
                "--out-dir",
                str(out_dir / "replay"),
                "--log-file",
                str(log_file),
            ]
        )
        trace = json.loads((out_dir / "trace.json").read_text(encoding="utf-8"))
        queue_observed = any(task["queue_count"] > 0 for task in trace["tasks"])
        records.append(
            {
                "name": name,
                "order": len(records) + 1,
                "run_code": summary["code"],
                "run_exit_code": exit_code,
                "query_id": summary["query_id"],
                "result_digest": summary["result_digest"],
                "result_file_sha256": sha256_file(result_file),
                "task_count": summary["task_count"],
                "split_count": summary["split_count"],
                "split_children_count": summary["split_children_count"],
                "replay_code": replay["code"],
                "replay_exit_code": replay_exit,
                "replay_tasks": replay["replayed_tasks"],
                "replay_result_digest": replay["result_digest"],
                "budget_queue_observed": queue_observed,
            }
        )
    identical = result_bytes["skew_amplified"] == result_bytes["tight_budget"]
    all_replays_ok = all(
        item["replay_code"] == "OK"
        and item["replay_result_digest"] == item["result_digest"]
        for item in records
    )
    summary_document = {
        "format_version": 1,
        "scenario_order": [name for name, _, _, _ in scenarios],
        "scenarios": records,
        "checks": {
            "all_runs_ok": all(item["run_code"] == "OK" for item in records),
            "all_replays_ok": all_replays_ok,
            "skew_split_observed": records[1]["split_count"] > 0,
            "tight_budget_queue_observed": records[2]["budget_queue_observed"],
            "skew_and_tight_budget_result_bytes_identical": identical,
            "ports_used": [],
            "paths_recreated_under_var_demo": True,
        },
    }
    summary_path = DEMO / "demo_summary.json"
    summary_path.write_text(
        json.dumps(
            summary_document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_document, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if not all(
        [
            summary_document["checks"]["all_runs_ok"],
            all_replays_ok,
            summary_document["checks"]["skew_split_observed"],
            summary_document["checks"]["tight_budget_queue_observed"],
            identical,
        ]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

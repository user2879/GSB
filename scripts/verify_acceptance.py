from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy_workbench.canonical import canonical_json
from deploy_workbench.engine import resolve_all
from deploy_workbench.errors import E_REPLAY, WorkbenchError
from deploy_workbench.fixed import FixedQuantizer
from deploy_workbench.service import WorkbenchService
from deploy_workbench.snapshot import read_snapshot, write_snapshot
from deploy_workbench.spec import load_spec


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    sample = ROOT / "data" / "sample_spec.json"
    report: dict = {"checks": []}
    with tempfile.TemporaryDirectory(prefix="workbench-acceptance-", dir=ROOT / "artifacts") as temporary:
        work = Path(temporary)
        state = work / "state"
        spec_path = work / "spec.json"
        raw = json.loads(sample.read_text(encoding="utf-8"))
        spec_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

        spec = load_spec(spec_path)
        quantizer = FixedQuantizer(spec.scale_power, spec.overflow_policy)
        first_resolution = resolve_all(spec, quantizer)
        second_resolution = resolve_all(spec, quantizer)
        check(canonical_json(first_resolution) == canonical_json(second_resolution), "resolution is not deterministic")
        check(
            sorted(first_resolution) == ["dev", "prod", "staging"],
            "unexpected environment set",
        )
        key_count = sum(len(env["keys"]) for env in first_resolution.values())
        uncertain_count = sum(
            sum(1 for item in env["comparisons"] if item["affected_by_error_bound"])
            for env in first_resolution.values()
        )
        for env in first_resolution.values():
            for task in env["tasks"].values():
                if task["status"] == "ok":
                    check(
                        Decimal(task["actual_error"]) <= Decimal(task["error_bound"]),
                        f"actual error exceeds bound for {env['environment_id']}/{task['task_id']}",
                    )
        report["checks"].append(
            {
                "name": "deterministic source-chain resolution",
                "status": "pass",
                "environment_count": len(first_resolution),
                "key_count": key_count,
                "uncertain_comparisons": uncertain_count,
                "dev_headroom_chain": [
                    (step["source_id"], step["selected"])
                    for step in first_resolution["dev"]["keys"]["headroom"]["chain"]
                ],
            }
        )

        service = WorkbenchService(spec_path, state)
        first_run = service.run_all()
        check(first_run["status"] == "complete", "initial run did not complete")
        replay = service.replay_snapshot()
        check(replay["status"] == "verified", "snapshot replay did not verify")
        task_samples = {
            env_id: {
                task_id: {
                    "fixed_result": task["fixed_result"],
                    "error_bound": task["error_bound"],
                    "actual_error": task["actual_error"],
                    "dominant": task["attribution"]["dominant_segment"],
                }
                for task_id, task in sorted(env["tasks"].items())
                if task["status"] == "ok"
            }
            for env_id, env in sorted(first_resolution.items())
        }
        report["checks"].append(
            {
                "name": "initial ledger and snapshot replay",
                "status": "pass",
                "run_id": first_run["run_id"],
                "versions": first_run["versions"],
                "snapshot_replay": replay["status"],
                "task_samples": task_samples,
            }
        )
        service.close()

        raw["environments"][0]["entries"]["file_dev"]["headroom"] = 0.5
        spec_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        service = WorkbenchService(spec_path, state)
        stale = service.get_current_view()
        check(stale["stale"] and stale["environments"] == [], "changed declaration did not invalidate current view")
        second_run = service.run_all()
        check(second_run["status"] == "complete", "replacement run did not complete")
        versions = service.list_versions()
        old = next(item for item in versions if item["run_id"] == first_run["run_id"])
        check(old["stale"] and old["read_only"], "old run was not retained as stale/read-only")
        report["checks"].append(
            {
                "name": "version bump and whole-view replacement",
                "status": "pass",
                "old_versions": {
                    "environment": first_run["versions"]["environment_version"],
                    "priority": first_run["versions"]["priority_version"],
                    "scale": first_run["versions"]["scale_version"],
                },
                "new_versions": {
                    "environment": second_run["versions"]["environment_version"],
                    "priority": second_run["versions"]["priority_version"],
                    "scale": second_run["versions"]["scale_version"],
                },
                "affected_environment_count": second_run["stats"]["affected_environment_count"],
                "affected_key_count": second_run["stats"]["affected_key_count"],
                "old_run_read_only": old["read_only"],
            }
        )

        snapshot = read_snapshot(service.latest_snapshot_path)
        snapshot["environments"]["dev"]["keys"]["cpu_cores"]["effective_fixed"] += 1
        tampered_path = work / "tampered_snapshot.json"
        write_snapshot(tampered_path, snapshot)
        try:
            service.replay_snapshot(tampered_path)
        except WorkbenchError as error:
            check(error.code == E_REPLAY, "tampered replay did not emit E_REPLAY")
            report["checks"].append(
                {
                    "name": "first replay mismatch location",
                    "status": "pass",
                    "code": error.code,
                    "environment": error.environment,
                    "key": error.details.get("key"),
                    "field": error.details.get("field"),
                }
            )
        else:
            raise AssertionError("tampered snapshot was accepted")
        service.close()

        injection_results = {}
        for flag, expected_code in (("--inject-env", "E_ENV"), ("--inject-bind", "E_BIND"), ("--inject-grid", "E_GRID")):
            injection_state = work / f"inject-{expected_code.lower()}"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "deploy_workbench",
                    "run",
                    "--spec",
                    str(sample),
                    "--state-root",
                    str(injection_state),
                    flag,
                    "--max-json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            payload = json.loads(completed.stdout)
            check(payload["code"] == expected_code, f"unexpected injection code for {flag}")
            injection_results[expected_code] = {
                "exit_code": completed.returncode,
                "path": payload["path"],
                "structured_log": (injection_state / "logs" / "workbench.jsonl").exists(),
            }
        report["checks"].append({"name": "exception injection chain", "status": "pass", "results": injection_results})

    report["overall"] = "pass"
    output = ROOT / "artifacts" / "acceptance_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

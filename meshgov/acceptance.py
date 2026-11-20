from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .util import atomic_write_json, utc_now_iso
from .viewmodel import build_view_model


class Acceptance:
    def __init__(self, repo: Path, report_path: Path, *, run_tests: bool = True) -> None:
        self.repo = repo.resolve()
        self.report_path = report_path
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
        self.out = self.repo / "artifacts" / "acceptance" / run_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.run_tests = run_tests
        self.checks: list[dict[str, Any]] = []
        self.evidence: dict[str, Any] = {"work_dir": str(self.out)}

    def check(self, name: str, passed: bool, detail: Any) -> None:
        self.checks.append(
            {
                "name": name,
                "status": "pass" if passed else "fail",
                "detail": detail,
            }
        )

    def command(
        self,
        name: str,
        args: list[str],
        *,
        timeout: int = 120,
    ) -> tuple[int, dict[str, Any] | None, str, str]:
        completed = subprocess.run(
            [sys.executable, *args],
            cwd=self.repo,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
        )
        parsed = None
        stdout = completed.stdout.strip()
        if stdout:
            for line in reversed(stdout.splitlines()):
                try:
                    parsed = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        self.evidence[name] = {
            "command": [sys.executable, *args],
            "returncode": completed.returncode,
            "stdout_tail": "\n".join(completed.stdout.strip().splitlines()[-4:]),
            "stderr_tail": "\n".join(completed.stderr.strip().splitlines()[-8:]),
        }
        return completed.returncode, parsed, completed.stdout, completed.stderr

    def run(self) -> dict[str, Any]:
        env = {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "cwd": str(self.repo),
        }
        if self.run_tests:
            code, _, stdout, stderr = self.command(
                "unit_tests",
                ["-m", "unittest", "discover", "-v"],
                timeout=180,
            )
            self.check(
                "unit_tests",
                code == 0,
                {
                    "returncode": code,
                    "summary": (stderr.strip().splitlines()[-1] if stderr.strip() else stdout.strip().splitlines()[-1] if stdout.strip() else "no output"),
                },
            )

        doctor_code, doctor, _, _ = self.command(
            "doctor",
            [
                "-m",
                "meshgov",
                "doctor",
                "--workspace",
                str(self.repo),
                "--report",
                str(self.out / "doctor.json"),
            ],
        )
        self.check(
            "environment_doctor",
            doctor_code == 0 and doctor is not None,
            doctor,
        )

        run_dir = self.out / "main"
        run_dir.mkdir(parents=True, exist_ok=True)
        common = [
            "--requests",
            "samples/requests.json",
            "--start-upstreams",
        ]
        code_one, run_one, _, _ = self.command(
            "run_v1_first",
            [
                "-m",
                "meshgov",
                "run",
                "--rules",
                "samples/rules.json",
                *common,
                "--db",
                str(run_dir / "first.db"),
                "--log",
                str(run_dir / "first.jsonl"),
                "--snapshots",
                str(run_dir / "snapshots-first"),
                "--report",
                str(run_dir / "first-report.json"),
                "--round-id",
                "accept-v1-first",
            ],
        )
        self.check(
            "run_v1_verified",
            code_one == 0
            and run_one is not None
            and run_one.get("verification", {}).get("status") == "pass",
            run_one,
        )

        code_two, run_two, _, _ = self.command(
            "run_v1_second",
            [
                "-m",
                "meshgov",
                "run",
                "--rules",
                "samples/rules.json",
                *common,
                "--db",
                str(run_dir / "second.db"),
                "--log",
                str(run_dir / "second.jsonl"),
                "--snapshots",
                str(run_dir / "snapshots-second"),
                "--report",
                str(run_dir / "second-report.json"),
                "--round-id",
                "accept-v1-second",
            ],
        )
        self.check(
            "run_v1_repeat_verified",
            code_two == 0
            and run_two is not None
            and run_two.get("verification", {}).get("status") == "pass",
            run_two,
        )
        deterministic = False
        deterministic_detail: dict[str, Any] = {}
        if run_one and run_two:
            first_snapshot = json.loads(
                (run_dir / "snapshots-first" / "latest.json").read_text(encoding="utf-8")
            )
            second_snapshot = json.loads(
                (run_dir / "snapshots-second" / "latest.json").read_text(encoding="utf-8")
            )
            first_sequence = [
                [row["request_id"], row["attempt_no"], row["target_instance"]]
                for row in first_snapshot["attempts"]
            ]
            second_sequence = [
                [row["request_id"], row["attempt_no"], row["target_instance"]]
                for row in second_snapshot["attempts"]
            ]
            deterministic = first_sequence == second_sequence
            deterministic_detail = {
                "first_length": len(first_sequence),
                "second_length": len(second_sequence),
                "first_target_counts": _target_counts(first_sequence),
                "second_target_counts": _target_counts(second_sequence),
            }
        self.check("deterministic_instance_sequence", deterministic, deterministic_detail)

        old_snapshot_path = Path(run_one["snapshot_path"]) if run_one else None
        old_hash = (
            hashlib.sha256(old_snapshot_path.read_bytes()).hexdigest()
            if old_snapshot_path and old_snapshot_path.exists()
            else None
        )
        v2_code, v2_run, _, _ = self.command(
            "run_v2",
            [
                "-m",
                "meshgov",
                "run",
                "--rules",
                "samples/rules_v2.json",
                *common,
                "--db",
                str(run_dir / "first.db"),
                "--log",
                str(run_dir / "first.jsonl"),
                "--snapshots",
                str(run_dir / "snapshots-first"),
                "--report",
                str(run_dir / "v2-report.json"),
                "--round-id",
                "accept-v2",
            ],
        )
        latest_path = run_dir / "snapshots-first" / "latest.json"
        latest = (
            json.loads(latest_path.read_text(encoding="utf-8"))
            if latest_path.exists()
            else {}
        )
        old_unchanged = bool(
            old_snapshot_path
            and old_hash
            and old_snapshot_path.exists()
            and hashlib.sha256(old_snapshot_path.read_bytes()).hexdigest() == old_hash
        )
        self.check(
            "snapshot_whole_replacement",
            v2_code == 0
            and v2_run is not None
            and v2_run.get("verification", {}).get("status") == "pass"
            and latest.get("round_id") == "accept-v2"
            and old_unchanged,
            {
                "latest_round": latest.get("round_id"),
                "latest_rule_version": latest.get("rule_version"),
                "old_snapshot_path": str(old_snapshot_path) if old_snapshot_path else None,
                "old_snapshot_hash_unchanged": old_unchanged,
                "all_latest_items_current_version": all(
                    row.get("rule_version") == latest.get("rule_version")
                    for collection in ("requests", "attempts", "services")
                    for row in latest.get(collection, [])
                ),
            },
        )
        if old_snapshot_path:
            diff_code, diff_report, _, _ = self.command(
                "snapshot_diff",
                [
                    "-m",
                    "meshgov",
                    "diff",
                    "--old",
                    str(old_snapshot_path),
                    "--new",
                    str(latest_path),
                    "--report",
                    str(run_dir / "snapshot-diff.json"),
                ],
            )
            self.check(
                "snapshot_diff_identifies_routes",
                diff_code == 0
                and diff_report is not None
                and diff_report.get("version_changed") is True
                and bool(
                    diff_report.get("route_diff", {}).get("changed_prefixes")
                    or diff_report.get("route_diff", {}).get("added_prefixes")
                ),
                diff_report,
            )

        mid_dir = self.out / "mid-round"
        mid_dir.mkdir(parents=True, exist_ok=True)
        live_rules = mid_dir / "rules-live.json"
        shutil.copyfile(self.repo / "samples" / "rules.json", live_rules)
        mid_code, mid_run, _, _ = self.command(
            "mid_round_rule_change",
            [
                "-m",
                "meshgov",
                "run",
                "--rules",
                str(live_rules),
                *common,
                "--db",
                str(mid_dir / "mesh.db"),
                "--log",
                str(mid_dir / "events.jsonl"),
                "--snapshots",
                str(mid_dir / "snapshots"),
                "--report",
                str(mid_dir / "run.json"),
                "--round-id",
                "accept-mid-change",
                "--mutate-rules-to",
                "samples/rules_v2.json",
                "--mutate-after-ms",
                "80",
            ],
        )
        self.check(
            "rules_changed_during_round_detected",
            mid_code == 0
            and mid_run is not None
            and mid_run.get("rules_changed_during_round") is True
            and mid_run.get("rule_version") != mid_run.get("observed_rule_version"),
            mid_run,
        )

        injections = self._run_injections()
        crash = self._run_crash_recovery()
        view_ok, view_detail = self._view_model_check(latest)
        self.check("tkinter_headless_view_model", view_ok, view_detail)

        all_passed = all(item["status"] == "pass" for item in self.checks)
        report = {
            "schema_version": 1,
            "command": "acceptance",
            "status": "pass" if all_passed else "fail",
            "generated_at_utc": utc_now_iso(),
            "environment": env,
            "summary": {
                "checks_total": len(self.checks),
                "checks_passed": sum(item["status"] == "pass" for item in self.checks),
                "checks_failed": sum(item["status"] == "fail" for item in self.checks),
            },
            "checks": self.checks,
            "injections": injections,
            "crash_recovery": crash,
            "evidence": self.evidence,
        }
        markdown_path = self.report_path.with_suffix(".md")
        report["report_path"] = str(self.report_path)
        report["markdown_report_path"] = str(markdown_path)
        atomic_write_json(self.report_path, report)
        markdown_path.write_text(_render_markdown(report), encoding="utf-8")
        return report

    def _run_injections(self) -> dict[str, Any]:
        root = self.out / "injections"
        root.mkdir(parents=True, exist_ok=True)
        specifications = [
            ("E_RULE", "rule_corruption", 2, []),
            ("E_UPSTREAM", "upstream_unreachable", 3, []),
            ("E_SNAPSHOT", "snapshot_write_failure", 4, ["--start-upstreams"]),
        ]
        output: dict[str, Any] = {}
        for expected_code, alias, expected_exit, extra in specifications:
            name = expected_code.lower()
            log_path = root / f"{name}.jsonl"
            report_path = root / f"{name}.json"
            code, parsed, _, _ = self.command(
                f"inject_{name}",
                [
                    "-m",
                    "meshgov",
                    "run",
                    "--rules",
                    "samples/rules.json",
                    "--requests",
                    "samples/requests.json",
                    "--db",
                    str(root / f"{name}.db"),
                    "--log",
                    str(log_path),
                    "--snapshots",
                    str(root / f"snapshots-{name}"),
                    "--report",
                    str(report_path),
                    "--round-id",
                    f"accept-inject-{name}",
                    "--inject",
                    alias,
                    *extra,
                ],
            )
            log_codes = []
            if log_path.exists():
                for line in log_path.read_text(encoding="utf-8").splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("code") == expected_code:
                        log_codes.append(event.get("event"))
            passed = (
                code == expected_exit
                and parsed is not None
                and parsed.get("code") == expected_code
                and bool(log_codes)
            )
            self.check(
                f"injection_{expected_code}",
                passed,
                {
                    "returncode": code,
                    "structured_log_codes": log_codes,
                    "report": parsed,
                },
            )
            output[expected_code] = {
                "returncode": code,
                "expected_exit": expected_exit,
                "structured_log_codes": log_codes,
            }
        return output

    def _run_crash_recovery(self) -> dict[str, Any]:
        root = self.out / "crash"
        root.mkdir(parents=True, exist_ok=True)
        crash_code, _, _, _ = self.command(
            "forced_termination",
            [
                "-m",
                "meshgov",
                "run",
                "--rules",
                "samples/rules_crash.json",
                "--requests",
                "samples/requests_crash.json",
                "--upstream-config",
                "samples/upstreams_crash.json",
                "--start-upstreams",
                "--crash-after-ms",
                "450",
                "--db",
                str(root / "mesh.db"),
                "--log",
                str(root / "events.jsonl"),
                "--snapshots",
                str(root / "snapshots"),
                "--report",
                str(root / "run.json"),
                "--round-id",
                "accept-crash",
            ],
        )
        recover_code, recovery, _, _ = self.command(
            "recovery",
            [
                "-m",
                "meshgov",
                "recover",
                "--db",
                str(root / "mesh.db"),
                "--log",
                str(root / "events.jsonl"),
                "--report",
                str(root / "recovery.json"),
            ],
        )
        passed = (
            crash_code == 137
            and recover_code == 0
            and recovery is not None
            and recovery.get("exit_report_supported") is False
            and len(recovery.get("unknown_items", [])) == 6
            and len(recovery.get("unknown_attempts", [])) >= 1
            and bool(recovery.get("persisted_quota_states"))
        )
        self.check(
            "forced_termination_recovery",
            passed,
            {
                "crash_exit": crash_code,
                "recover_exit": recover_code,
                "exit_report_supported": recovery.get("exit_report_supported") if recovery else None,
                "unknown_items": len(recovery.get("unknown_items", [])) if recovery else None,
                "unknown_attempts": len(recovery.get("unknown_attempts", [])) if recovery else None,
                "persisted_quota_states": len(recovery.get("persisted_quota_states", [])) if recovery else None,
            },
        )
        return {
            "crash_exit": crash_code,
            "recover_exit": recover_code,
            "exit_report_supported": recovery.get("exit_report_supported") if recovery else None,
            "unknown_items": recovery.get("unknown_items", []) if recovery else [],
            "unknown_attempts": recovery.get("unknown_attempts", []) if recovery else [],
        }

    def _view_model_check(self, snapshot: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if not snapshot:
            return False, {"reason": "latest snapshot unavailable"}
        view = build_view_model(snapshot)
        instances = view["instances"]
        routes = view["routes"]
        instance_order_ok = instances == sorted(
            instances, key=lambda row: (row["service"], row["port"])
        )
        route_order_ok = routes == sorted(
            routes, key=lambda row: (row["prefix"], row["service"])
        )
        version_ok = all(
            route.get("rule_version") == view["meta"]["rule_version"] for route in routes
        )
        return instance_order_ok and route_order_ok and version_ok, {
            "progress": view["progress"],
            "instance_count": len(instances),
            "route_count": len(routes),
            "stable_instance_order": instance_order_ok,
            "stable_route_order": route_order_ok,
            "single_rule_version": version_ok,
            "conclusion": view["summary"].get("conclusion"),
        }


def _target_counts(sequence: list[list[Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, _, target in sequence:
        counts[str(target)] = counts.get(str(target), 0) + 1
    return dict(sorted(counts.items()))


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Service Mesh Workbench Acceptance Report",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Checks: {report['summary']['checks_passed']}/{report['summary']['checks_total']} passed",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for check in report["checks"]:
        lines.append(f"| {check['name']} | {check['status'].upper()} |")
    lines.extend(
        [
            "",
            "## Injection evidence",
            "",
            "| Code | Exit | Structured log events |",
            "|---|---:|---|",
        ]
    )
    for code, item in report["injections"].items():
        lines.append(
            f"| {code} | {item['returncode']} | {', '.join(item['structured_log_codes'])} |"
        )
    crash = report["crash_recovery"]
    lines.extend(
        [
            "",
            "## Forced termination",
            "",
            f"- Run exit: `{crash['crash_exit']}`",
            f"- Recovery exit: `{crash['recover_exit']}`",
            f"- Exit report supported: `{crash['exit_report_supported']}`",
            f"- Unknown requests requiring rerun: `{len(crash['unknown_items'])}`",
            f"- Unknown incomplete attempts: `{len(crash['unknown_attempts'])}`",
            "",
            "The interrupted round is intentionally not converted into an exit report.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the complete local acceptance suite")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--report", default="artifacts/reports/acceptance.json")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    acceptance = Acceptance(
        Path(args.workspace),
        Path(args.report),
        run_tests=not args.skip_tests,
    )
    report = acceptance.run()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

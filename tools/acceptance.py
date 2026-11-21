"""Run all reproducible acceptance checks and emit a validation report."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from txn_mw.canonical import canonical_json
from txn_mw.scenario import run_scenario


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    output_root = ROOT / "artifacts" / "validation"
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "ok": False,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "checks": {},
    }
    unit = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=120,
    )
    (output_root / "unittest.log").write_text(unit.stdout + unit.stderr, encoding="utf-8")
    match = re.search(r"Ran (\d+) tests", unit.stderr)
    summary["checks"]["unit_tests"] = {
        "ok": unit.returncode == 0,
        "tests": int(match.group(1)) if match else None,
        "returncode": unit.returncode,
    }

    scenario = run_scenario(
        ROOT / "samples" / "scenario.json",
        artifacts_dir=output_root / "scenario",
        log_path=output_root / "scenario-structured.jsonl",
    )
    summary["checks"]["scenario"] = {
        "ok": scenario.get("ok"),
        "errors": scenario.get("errors", []),
        "transactions": len(scenario.get("transactions", {})),
        "pulls": len(scenario.get("pulls", {}).get("concurrent", []))
        + len(scenario.get("pulls", {}).get("sequential", [])),
        "pull_stats": scenario.get("latency", {}).get("overall", {}).get("pull"),
        "query_stats": scenario.get("latency", {}).get("overall", {}).get("query"),
        "inventory": scenario.get("inventory", {}).get("counts"),
        "verify_ok": scenario.get("verify", {}).get("ok"),
        "ledger_verify_ok": scenario.get("ledger_verify", {}).get("ok"),
        "structured_error_codes": sorted(
            {
                json.loads(line)["code"]
                for line in (output_root / "scenario-structured.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("level") == "error"
            }
        ),
    }

    crash = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "crash_harness.py")],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=120,
    )
    crash_path = ROOT / "artifacts" / "crash" / "crash-harness-result.json"
    crash_result = json.loads(crash_path.read_text(encoding="utf-8")) if crash_path.exists() else {}
    summary["checks"]["crash_recovery"] = {
        "ok": crash_result.get("ok"),
        "error": crash_result.get("error"),
        "forced_exit_report_created": crash_result.get("forced_kill", {}).get("forced_exit_report_created"),
        "incomplete_queries_requeued": crash_result.get("recovery", {}).get(
            "incomplete_queries_marked_unknown_and_requeued"
        ),
        "tail_incomplete_count": len(crash_result.get("tail_recovery", {}).get("tail_incomplete_records", [])),
        "latency_after_recovery": crash_result.get("latency_after_recovery", {}).get("overall"),
        "error_codes_observed": crash_result.get("error_codes_observed"),
        "ledger_verify_ok": crash_result.get("ledger_verify", {}).get("ok"),
    }

    summary["source_sha256"] = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted((ROOT / "txn_mw").glob("*.py"))
    }
    summary["ok"] = all(check.get("ok") for check in summary["checks"].values())
    (output_root / "acceptance-report.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")

    lines = [
        "# Validation Result",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        f"Python: `{platform.python_version()}` on `{platform.platform()}`",
        "",
        "## Checks",
        "",
        "| Check | Result | Evidence |",
        "|---|---|---|",
        f"| Unit/integration tests | {'PASS' if summary['checks']['unit_tests']['ok'] else 'FAIL'} | `{summary['checks']['unit_tests']['tests']}` tests |",
        f"| Deterministic loopback scenario | {'PASS' if summary['checks']['scenario']['ok'] else 'FAIL'} | `{summary['checks']['scenario']['transactions']}` transactions, `{summary['checks']['scenario']['pulls']}` pulls |",
        f"| Forced-kill recovery | {'PASS' if summary['checks']['crash_recovery']['ok'] else 'FAIL'} | no forced exit report claimed; incomplete query requeued |",
        "",
        "## Scenario Ledger",
        "",
        f"- Pull samples: `{summary['checks']['scenario']['pull_stats']['sample_count']}`; "
        f"mean `{summary['checks']['scenario']['pull_stats']['mean_ms_x1000']}` / "
        f"P95 `{summary['checks']['scenario']['pull_stats']['p95_ms_x1000']}` / "
        f"max `{summary['checks']['scenario']['pull_stats']['max_ms_x1000']}` ms x1000.",
        f"- Query samples: `{summary['checks']['scenario']['query_stats']['sample_count']}`; "
        f"mean `{summary['checks']['scenario']['query_stats']['mean_ms_x1000']}` / "
        f"P95 `{summary['checks']['scenario']['query_stats']['p95_ms_x1000']}` / "
        f"max `{summary['checks']['scenario']['query_stats']['max_ms_x1000']}` ms x1000.",
        f"- Structured error codes observed: `{', '.join(summary['checks']['scenario']['structured_error_codes'])}`.",
        f"- Independent replay: `{summary['checks']['scenario']['verify_ok']}`; ledger-file verification: `{summary['checks']['scenario']['ledger_verify_ok']}`.",
        "",
        "## Recovery Ledger",
        "",
        f"- Forced exit report created: `{summary['checks']['crash_recovery']['forced_exit_report_created']}`.",
        f"- Incomplete queries marked unknown and requeued: `{summary['checks']['crash_recovery']['incomplete_queries_requeued']}`.",
        f"- Tail incomplete records quarantined: `{summary['checks']['crash_recovery']['tail_incomplete_count']}`.",
        f"- Recovery error codes: `{', '.join(summary['checks']['crash_recovery']['error_codes_observed'] or [])}`.",
        "",
        "The machine-readable truth source is `artifacts/validation/acceptance-report.json`; per-interaction latency rows are in the generated SQLite ledgers.",
        "",
    ]
    (ROOT / "VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(canonical_json(summary))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


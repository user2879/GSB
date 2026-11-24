#!/usr/bin/env python
"""Reproduce the acceptance checks and write machine-readable evidence."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmledger.demo import generate_workspace  # noqa: E402
from mmledger.jsonutil import atomic_write_json, atomic_write_text, hash_json  # noqa: E402
from mmledger.pipeline import run_pipeline  # noqa: E402


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def check(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "details": details}


def copy_sample_workspace(source: Path, target: Path) -> Path:
    shutil.copytree(source / "samples", target / "samples")
    (target / "snapshots").mkdir(parents=True, exist_ok=True)
    (target / "reports").mkdir(parents=True, exist_ok=True)
    return target / "samples" / "config.json"


def verify_parameter_version_boundary(samples: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp)
        config_path = copy_sample_workspace(samples, target)
        before = run_pipeline(config_path, jobs=4, full_rebuild=True)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["workers"][0]["parameter_version"] = 5
        config["workers"][0]["parameter_digest"] = "sha256:pv5-alpha"
        config["parameters"].append({"parameter_version": 5, "digest": "sha256:pv5-alpha"})
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        after = run_pipeline(config_path, jobs=4)
        actual = after.report["actual_invalidation"]
        expected_invalidated = ["batch-001", "batch-003", "batch-005"]
        expected_reused = ["batch-002", "batch-004"]
        return check(
            "parameter_version_invalidation_boundary",
            actual["normalization_computed_batches"] == expected_invalidated
            and actual["normalization_reused_batches"] == expected_reused
            and actual["resource_computed_batches"] == expected_invalidated
            and actual["resource_reused_batches"] == expected_reused,
            {
                "before_digest": before.ledger["ledger_digest"],
                "after_digest": after.ledger["ledger_digest"],
                "changed_workers": actual["changed_workers"],
                "normalization_computed": actual["normalization_computed_batches"],
                "normalization_reused": actual["normalization_reused_batches"],
                "resource_computed": actual["resource_computed_batches"],
                "resource_reused": actual["resource_reused_batches"],
            },
        )


def verify_budget_and_quota(samples: Path) -> list[dict[str, Any]]:
    checks = []
    with tempfile.TemporaryDirectory() as temp:
        config_path = copy_sample_workspace(samples, Path(temp))
        limited = run_pipeline(config_path, jobs=2, full_rebuild=True, normalization_budget=9)
        first_pending = limited.ledger["scheduler"]["pending_queue"]
        first_codes = [item["code"] for item in limited.ledger["failures"]]
        checks.append(
            check(
                "normalization_budget_pending",
                first_pending == ["batch-004", "batch-005"]
                and first_codes.count("E_NORMALIZE_BUDGET_PENDING") == 2,
                {
                    "budget": 9,
                    "consumed": limited.ledger["scheduler"]["consumed_budget"],
                    "pending": first_pending,
                    "failure_codes": first_codes,
                },
            )
        )
        resumed = run_pipeline(config_path, jobs=2, normalization_budget=4)
        tickets = resumed.ledger["scheduler"]["tickets"]
        checks.append(
            check(
                "previous_pending_priority",
                [ticket["batch_id"] for ticket in tickets] == ["batch-004", "batch-005"]
                and all(ticket["priority_pending"] for ticket in tickets),
                {
                    "budget": 4,
                    "tickets": tickets,
                    "consumed": resumed.ledger["scheduler"]["consumed_budget"],
                },
            )
        )
    with tempfile.TemporaryDirectory() as temp:
        config_path = copy_sample_workspace(samples, Path(temp))
        quota = run_pipeline(config_path, jobs=2, full_rebuild=True, worker_quota_points=5)
        blocked = quota.ledger["scheduler"]["quota_blocked"]
        codes = [item["code"] for item in quota.ledger["failures"]]
        checks.append(
            check(
                "worker_quota_exceeded",
                blocked == ["batch-003"] and "E_WORKER_QUOTA_EXCEEDED" in codes,
                {"quota": 5, "blocked": blocked, "failure_codes": codes},
            )
        )
    return checks


def run_unit_tests() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout
    match = re.search(r"Ran (\d+) tests?", output)
    return check(
        "unit_tests",
        completed.returncode == 0 and "OK" in output,
        {
            "command": "python -m unittest discover -s tests -v",
            "returncode": completed.returncode,
            "tests": int(match.group(1)) if match else None,
            "tail": "\n".join(output.strip().splitlines()[-12:]),
        },
    )


def main() -> int:
    generation = generate_workspace(ROOT, use_subprocess_workers=True, clean_outputs=True, jobs=4)
    config_path = ROOT / "samples" / "config.json"

    full_serial = run_pipeline(config_path, jobs=1, full_rebuild=True)
    serial_payload = canonical(full_serial.ledger)
    full_concurrent = run_pipeline(config_path, jobs=8, full_rebuild=True)
    concurrent_payload = canonical(full_concurrent.ledger)
    incremental = run_pipeline(config_path, jobs=4)
    incremental_payload = canonical(incremental.ledger)
    checks = [
        check(
            "serial_concurrent_ledger_equality",
            serial_payload == concurrent_payload,
            {
                "serial_digest": full_serial.ledger["ledger_digest"],
                "concurrent_digest": full_concurrent.ledger["ledger_digest"],
            },
        ),
        check(
            "incremental_full_ledger_equality",
            serial_payload == incremental_payload,
            {
                "full_digest": full_serial.ledger["ledger_digest"],
                "incremental_digest": incremental.ledger["ledger_digest"],
                "actual_normalization_computed": incremental.report["actual_invalidation"][
                    "normalization_computed_batches"
                ],
                "actual_resource_computed": incremental.report["actual_invalidation"][
                    "resource_computed_batches"
                ],
            },
        ),
        check(
            "three_version_normalized_ledger",
            sorted(incremental.ledger["receipt_versions"].values()) == ["r1", "r2", "r3", "r3", "r3", "r3"],
            {
                "receipt_versions": incremental.ledger["receipt_versions"],
                "counts": incremental.ledger["counts"],
                "ledger_digest": incremental.ledger["ledger_digest"],
            },
        ),
    ]

    conversion = json.loads((ROOT / "samples" / "conversion" / "roundtrip_check.json").read_text(encoding="utf-8"))
    checks.append(
        check(
            "conversion_roundtrip_restores_all_fields",
            conversion["completeness"] == "complete" and all(conversion["fields_equal"].values()),
            conversion,
        )
    )

    fault = run_pipeline(config_path, jobs=4, fault_drill=True, commit_snapshot=False)
    injection_hits = fault.report["injection_hits"]
    checks.append(
        check(
            "fault_injection_isolation",
            len(injection_hits) == 4 and all(item["hit"] for item in injection_hits),
            {
                "injection_hits": injection_hits,
                "failure_codes": [item["code"] for item in fault.report["failures"]],
            },
        )
    )

    checks.append(verify_parameter_version_boundary(ROOT))
    checks.extend(verify_budget_and_quota(ROOT))
    checks.append(run_unit_tests())

    snapshot = json.loads((ROOT / "snapshots" / "CURRENT.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / "snapshots" / "generations" / snapshot["generation"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks.append(
        check(
            "snapshot_manifest_complete",
            manifest["completed_stages"] == ["parse", "ownership", "resources", "ledger"]
            and manifest["config_hash"]
            and manifest["receipt_fingerprints"]
            and manifest["completed_workers"] == ["w-alpha", "w-beta"],
            {
                "global_version": manifest["global_version"],
                "config_hash": manifest["config_hash"],
                "receipt_fingerprint_count": len(manifest["receipt_fingerprints"]),
                "completed_workers": manifest["completed_workers"],
                "consumed_budget": manifest["consumed_budget"],
            },
        )
    )

    # Leave the workspace in an identical-input continued-run state. This keeps
    # the final ledger unchanged while proving the cache was actually reused.
    final = run_pipeline(config_path, jobs=4)
    checks.append(
        check(
            "final_continued_run_reuses_cache",
            canonical(final.ledger) == serial_payload
            and not final.report["actual_invalidation"]["normalization_computed_batches"]
            and not final.report["actual_invalidation"]["resource_computed_batches"],
            {
                "ledger_digest": final.ledger["ledger_digest"],
                "actual_invalidation": final.report["actual_invalidation"],
            },
        )
    )

    passed = all(item["passed"] for item in checks)
    report = {
        "schema_version": "mmledger-verification-v1",
        "status": "passed" if passed else "failed",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": sys.platform,
        },
        "workspace": str(ROOT),
        "generation": generation,
        "ledger_digest": final.ledger["ledger_digest"],
        "checks": checks,
    }
    atomic_write_json(ROOT / "reports" / "verification.json", report)
    lines = [
        "# Acceptance Verification",
        "",
        f"- Status: **{report['status']}**",
        f"- Python: `{report['environment']['python']}`",
        f"- NumPy: `{report['environment']['numpy']}`",
        f"- Ledger digest: `{report['ledger_digest']}`",
        f"- Worker ports: `{generation['ports']}`",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for item in checks:
        lines.append(f"| `{item['check']}` | {'PASS' if item['passed'] else 'FAIL'} |")
    lines.extend(["", "## Details", "", "```json", json.dumps(checks, ensure_ascii=False, indent=2), "```", ""])
    atomic_write_text(ROOT / "reports" / "verification.md", "\n".join(lines))
    print(json.dumps({"status": report["status"], "ledger_digest": report["ledger_digest"], "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

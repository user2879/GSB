#!/usr/bin/env python3
"""Summarise the acceptance artefacts produced by run_acceptance.sh."""

from __future__ import annotations

import json
from pathlib import Path


OUT = Path("results/acceptance")


def read(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def code(name: str) -> int:
    return int((OUT / f"{name}.exit_code").read_text().strip())


def main() -> int:
    expected_validation_codes = {
        "validate-valid": 0,
        "validate-duplicate_row": 1,
        "validate-nonfinite_row": 1,
        "validate-timestamp_reversal": 1,
        "validate-bad_dag": 1,
        "validate-missing_upstream": 1,
        "validate-missing_raw_field": 1,
        "validate-bad_config_quota": 1,
        "validate-bad_config_lease_ttl": 1,
        "validate-bad_plan": 1,
    }
    validation = {}
    for name, expected_code in expected_validation_codes.items():
        payload = read(f"{name}.stdout.json")
        validation[name] = {
            "exit_code": code(name),
            "expected_exit_code": expected_code,
            "passed": code(name) == expected_code,
            "failures": [
                {"code": item.get("code"), "stage": item.get("stage")}
                for item in payload.get("failures", [])
            ],
        }

    quota = read("quota-report.json")
    incremental = read("incremental-report.json")
    serial = read("serial-report.json")
    resume = read("resume-report.json")
    lease = read("lease-demo.stdout.json")
    incremental_vs_full = read("compare-incremental-final.stdout.json")
    parallel_vs_serial = read("compare-parallel-serial.stdout.json")
    resume_vs_incremental = read("compare-resume-final.stdout.json")

    expected_injections = {
        "nonfinite": ("INJ-001-NONFINITE", "E_ROW_INVALID"),
        "timestamp_reversal": ("INJ-002-TIMESTAMP-REVERSAL", "E_ROW_INVALID"),
        "lease_ttl_zero": ("INJ-003-LEASE-TTL-ZERO", "E_CONFIG_INVALID"),
        "cache_budget_zero": ("INJ-004-CACHE-BUDGET-ZERO", "E_CONFIG_INVALID"),
    }
    injections = {}
    for name, (expected_id, expected_code) in expected_injections.items():
        report = read(f"injection-{name}-report.json")
        hits = report["injection_hits"]
        failure_codes = sorted({item["code"] for item in report["failures"]})
        injections[name] = {
            "hit_count": len(hits),
            "hit": hits[0] if hits else None,
            "expected_injection_id": expected_id,
            "expected_code": expected_code,
            "failure_codes": failure_codes,
            "pending": report["pending"],
            "passed": len(hits) == 1
            and hits[0]["injection_id"] == expected_id
            and hits[0]["observed_code"] == expected_code
            and failure_codes == [expected_code]
            and not report["pending"],
        }

    checks = {
        "validation": all(item["passed"] for item in validation.values()),
        "lease_demo": code("lease-demo") == 0
        and lease["status"] == "reclaimed"
        and lease["conflict_observed"],
        "incremental_vs_full": code("compare-incremental-final") == 0
        and incremental_vs_full["equal"],
        "parallel_vs_serial": code("compare-parallel-serial") == 0
        and parallel_vs_serial["equal"]
        and parallel_vs_serial["execution_equal"],
        "quota": code("run-quota") == 3
        and quota["status"] == "partial"
        and bool(quota["pending"])
        and all(
            value <= 60 for value in quota["consumed_points_by_batch"].values()
        ),
        "crash": code("crash") == 137,
        "resume": code("run-resume") == 0
        and resume["run"]["resumed"]
        and code("compare-resume-final") == 0
        and resume_vs_incremental["equal"],
        "injections": all(item["passed"] for item in injections.values()),
        "unit_tests": code("unit-tests") == 0,
        "concurrency_limit": incremental["run"]["max_observed_concurrency"]
        <= incremental["run"]["max_workers"]
        and serial["run"]["max_observed_concurrency"] == 1,
    }
    summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "validation": validation,
        "lease_demo": {
            "status": lease["status"],
            "elapsed_ms": lease["elapsed_ms"],
            "conflict_observed": lease["conflict_observed"],
            "events": [event["event"] for event in lease["events"]],
            "reclaim": next(
                event
                for event in lease["events"]
                if event["event"] == "lease_reclaimed"
            ),
        },
        "incremental_vs_full": incremental_vs_full,
        "parallel_vs_serial": parallel_vs_serial,
        "resume_vs_incremental": resume_vs_incremental,
        "quota": {
            "status": quota["status"],
            "pending": quota["pending"],
            "failure_codes": sorted({item["code"] for item in quota["failures"]}),
            "consumed_points_by_batch": quota["consumed_points_by_batch"],
        },
        "concurrency": {
            "parallel": {
                "max_workers": incremental["run"]["max_workers"],
                "max_observed_concurrency": incremental["run"]["max_observed_concurrency"],
            },
            "serial": {
                "max_workers": serial["run"]["max_workers"],
                "max_observed_concurrency": serial["run"]["max_observed_concurrency"],
            },
        },
        "resume": {
            "resumed": resume["run"]["resumed"],
            "resumed_from_generation": resume["run"]["resumed_from_generation"],
            "uncommitted_temporaries": len(
                resume["run"]["uncommitted_temporary_generations"]
            ),
            "status": resume["status"],
        },
        "injections": injections,
        "incremental_counts": {
            "results": len(incremental["results"]),
            "summaries": len(incremental["summaries"]),
            "skips": len(incremental["skips"]),
            "entities": len(incremental["entities"]),
            "snapshot_generation": incremental["run"]["snapshot_generation"],
        },
        "unit_tests_exit_code": code("unit-tests"),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

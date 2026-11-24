#!/usr/bin/env python3
"""Run the local acceptance matrix and write a reproducible JSON report."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / ".deps", ROOT / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402

from gsb_ledger.attribution import attribute_receipts  # noqa: E402
from gsb_ledger.config import load_config, load_requests  # noqa: E402
from gsb_ledger.conversions import roundtrip_via_sidecar  # noqa: E402
from gsb_ledger.engine import LedgerEngine  # noqa: E402
from gsb_ledger.receipts import load_receipts  # noqa: E402
from gsb_ledger.resources import compute_resource  # noqa: E402
from gsb_ledger.scenario import run_scenario  # noqa: E402
from gsb_ledger.util import atomic_write_json  # noqa: E402
from gsb_ledger.views import compare_views, render_view  # noqa: E402


SAMPLE = ROOT / "sample"
RUNTIME = ROOT / "artifacts" / "runtime" / "acceptance"
OUTPUT = ROOT / "artifacts" / "verification"


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    entries = [str(ROOT / "src")]
    if (ROOT / ".deps").is_dir():
        entries.insert(0, str(ROOT / ".deps"))
    environment["PYTHONPATH"] = os.pathsep.join(
        entries + [environment.get("PYTHONPATH", "")]
    )
    environment["NO_COLOR"] = "1"
    return environment


def _free_ports(count: int) -> list[int]:
    result: list[int] = []
    for port in range(18300, 18400):
        if len(result) == count:
            break
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            try:
                candidate.bind(("127.0.0.1", port))
            except OSError:
                continue
            result.append(port)
    if len(result) != count:
        raise RuntimeError("not enough free ports in the configured acceptance range")
    return result


def _prepare_worker_project() -> Path:
    project = RUNTIME / "worker-project"
    if project.exists():
        shutil.rmtree(project)
    shutil.copytree(SAMPLE, project, ignore=shutil.ignore_patterns(".state"))
    raw = json.loads((project / "config.json").read_text(encoding="utf-8"))
    for worker, port in zip(raw["workers"], _free_ports(len(raw["workers"])), strict=True):
        worker["port"] = port
    (project / "config.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return project


def main() -> int:
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    RUNTIME.mkdir(parents=True)
    OUTPUT.mkdir(parents=True)
    environment = _environment()
    report: dict[str, object] = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    checks: dict[str, bool] = {}

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts="],
        cwd=str(ROOT),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    match = re.search(r"(\d+) passed", tests.stdout)
    report["tests"] = {
        "returncode": tests.returncode,
        "passed": int(match.group(1)) if match else 0,
        "tail": tests.stdout.strip().splitlines()[-5:],
    }
    checks["tests"] = tests.returncode == 0

    engine = LedgerEngine(SAMPLE, state_dir=RUNTIME / "sample-state")
    sample_result = engine.build(resume=False)
    atomic_write_json(OUTPUT / "sample-ledger.json", sample_result.ledger)
    report["sample_ledger"] = {
        "ledger_version": sample_result.ledger_version,
        "summary": sample_result.ledger["summary"],
        "authority_rules": {
            entry["request_id"]: {
                "batch_id": entry["authority"]["batch_id"],
                "rule": entry["authority_rule"],
            }
            for entry in sample_result.ledger["entries"]
        },
    }
    checks["sample_ledger"] = (
        sample_result.ledger["summary"]["entries_authoritative"] == 6
        and not sample_result.ledger["failures"]
    )

    scenario = run_scenario(
        source_project=SAMPLE,
        work_dir=RUNTIME / "scenario-work",
        commands_path=SAMPLE / "commands" / "incremental_sequence.json",
        out_dir=OUTPUT / "scenario",
    )
    report["scenario"] = {
        "commands": scenario["commands"],
        "identical": scenario["identical"],
        "differences": scenario["differences"],
        "incremental_ledger_version": scenario["incremental_ledger_version"],
        "full_ledger_version": scenario["full_ledger_version"],
        "summary": scenario["summary"],
    }
    checks["incremental_equals_full"] = bool(scenario["identical"])

    fault_engine = LedgerEngine(
        SAMPLE,
        state_dir=RUNTIME / "fault-state",
        fault_config="faults/isolated_faults.json",
    )
    fault_result = fault_engine.build(resume=False)
    atomic_write_json(OUTPUT / "fault-ledger.json", fault_result.ledger)
    diagnostics = fault_result.ledger["diagnostics"]
    atomic_write_json(OUTPUT / "fault-diagnostics.json", diagnostics)
    report["faults"] = {
        "all_hit": all(item["hit"] for item in diagnostics),
        "items": [
            {
                "injection_id": item["injection_id"],
                "hit": item["hit"],
                "expected_code": item["expected_code"],
                "observed_codes": item["observed_codes"],
                "observed_details": item["observed_details"],
            }
            for item in diagnostics
        ],
        "failure_codes": sorted({item["code"] for item in fault_result.ledger["failures"]}),
    }
    checks["fault_injections"] = bool(report["faults"]["all_hit"])

    fast = render_view(sample_result.ledger, mode="fast", budget_ms=1000)
    authority = render_view(sample_result.ledger, mode="authoritative", budget_ms=None)
    exhausted = render_view(sample_result.ledger, mode="fast", budget_ms=0)
    view_issues = compare_views(fast, authority)
    report["views"] = {
        "complete_fast_equals_authority": fast["entries"] == authority["entries"],
        "divergence_issues": [issue.to_dict() for issue in view_issues],
        "budget_zero": {
            "status": exhausted["status"],
            "error_codes": exhausted["error_codes"],
            "progress": exhausted["progress"],
        },
    }
    checks["fast_authority"] = (
        not view_issues
        and exhausted["status"] == "partial"
        and exhausted["error_codes"] == ["E_RESPONSE_BUDGET_EXCEEDED"]
    )

    config = load_config(SAMPLE / "config.json")
    requests = load_requests(SAMPLE / "requests.ndjson", batch_size=config.batch_size)
    receipt_load = load_receipts(config, project_root=SAMPLE)
    original = next(item for item in receipt_load.receipts if item.key == "b-alpha-001/r-001")
    down, up, sidecar = roundtrip_via_sidecar(original, "r1")
    report["conversion"] = {
        "preserved_fields": sorted(down.preserved_fields),
        "down_lost_fields": list(down.lost_fields),
        "up_lost_fields": list(up.lost_fields),
        "restored": all(
            getattr(up.receipt, field) == getattr(original, field)
            for field in (
                "batch_id",
                "request_id",
                "output_digest",
                "param_version",
                "duration_ms",
                "worker_id",
                "rank",
                "covered_refs",
                "param_digest",
            )
        ),
        "sidecar_records": len(sidecar.records),
    }
    checks["conversion_roundtrip"] = bool(
        report["conversion"]["restored"] and not up.lost_fields
    )

    attribution = attribute_receipts(receipt_load.receipts, requests, config)
    batch = next(item for item in attribution.batches if item.batch_id == "b-alpha-001")
    fixed_points = []
    for bits in (16, 24, 32, 48, 64):
        metric = compute_resource(batch, replace(config, fixed_point_bits=bits))
        fixed_points.append(
            {
                "bits": bits,
                "fixed_macs": metric.fixed_macs,
                "utilization_units": metric.utilization_units,
                "utilization_scale": metric.utilization_scale,
                "retained_decimal_digits": metric.retained_decimal_digits,
            }
        )
    report["fixed_point"] = fixed_points
    checks["fixed_point"] = len({item["fixed_macs"] for item in fixed_points}) == len(
        fixed_points
    ) and len({item["retained_decimal_digits"] for item in fixed_points}) == len(fixed_points)

    worker_project = _prepare_worker_project()
    start = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "start_workers.py"), str(worker_project)],
        cwd=str(ROOT),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    smoke = None
    stop = None
    try:
        if start.returncode == 0:
            smoke = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "smoke_worker.py"),
                    str(worker_project),
                    "w-alpha",
                    "b-acceptance-smoke",
                ],
                cwd=str(ROOT),
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
    finally:
        stop = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "stop_workers.py")],
            cwd=str(ROOT),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    smoke_payload = None
    if smoke is not None and smoke.returncode == 0:
        try:
            smoke_payload = json.loads(smoke.stdout)
        except json.JSONDecodeError:
            smoke_payload = None
    report["worker"] = {
        "start_returncode": start.returncode,
        "start_output_tail": start.stdout.strip().splitlines()[-3:],
        "smoke_returncode": None if smoke is None else smoke.returncode,
        "smoke_ok": bool(smoke_payload and smoke_payload.get("ok")),
        "receipt_rows": 0 if not smoke_payload else len(smoke_payload.get("receipts", [])),
        "stop_returncode": stop.returncode if stop is not None else None,
    }
    checks["worker_loopback"] = bool(report["worker"]["smoke_ok"])

    report["checks"] = checks
    report["ok"] = all(checks.values())
    atomic_write_json(OUTPUT / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

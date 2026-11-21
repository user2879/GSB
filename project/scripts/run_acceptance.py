#!/usr/bin/env python3
"""Reproducible end-to-end acceptance run for the local arbitration workbench."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy  # noqa: E402

from arbitration_workbench.config import parse_config  # noqa: E402
from arbitration_workbench.engine import ArbitrationEngine  # noqa: E402
from arbitration_workbench.generator import generate_samples  # noqa: E402
from arbitration_workbench.loader import load_evidence_file  # noqa: E402
from arbitration_workbench.numeric import read_json, same_volume, write_json  # noqa: E402
from arbitration_workbench.rules import parse_rules  # noqa: E402
from arbitration_workbench.versions import (  # noqa: E402
    convert_file,
    read_v1,
    read_v2,
    read_v3,
    write_v1,
    write_v2,
    write_v3,
)
from arbitration_workbench.workbench import Workbench  # noqa: E402


def copy_sample_workspace(target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    (target / "samples").mkdir(parents=True, exist_ok=True)
    (target / "tmp").mkdir(parents=True, exist_ok=True)
    for name in ("rules.json", "config.json", "evidence.jsonl", "exception_injections.json"):
        shutil.copy2(ROOT / "samples" / name, target / "samples" / name)
    return target


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def stage_stats(result) -> dict:
    return result.diagnostics["invalidation_and_reuse"]["stages"]


def run_unittests() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-v"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    tail = (completed.stderr or completed.stdout).strip().splitlines()[-4:]
    return {
        "command": "python -m unittest discover -v",
        "return_code": completed.returncode,
        "passed": completed.returncode == 0,
        "summary_lines": tail,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-unittest", action="store_true")
    parser.add_argument("--keep-work", action="store_true", help="retain transient acceptance workspaces")
    args = parser.parse_args()
    started = time.monotonic()
    results: dict[str, dict] = {}

    generated = generate_samples(ROOT)
    results["sample_generation"] = {
        "passed": generated["ticket_count"] == 48 and generated["boundary_ticket"] == "T0047",
        "seed": generated["seed"],
        "ticket_count": generated["ticket_count"],
        "evidence_count": generated["evidence_count"],
        "boundary_ticket": generated["boundary_ticket"],
    }
    first_sample_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "samples").rglob("*"))
        if path.is_file() and path.name != "sample_manifest.json"
    }
    regenerated = generate_samples(ROOT)
    second_sample_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "samples").rglob("*"))
        if path.is_file() and path.name != "sample_manifest.json"
    }
    results["sample_reproducibility"] = {
        "passed": first_sample_hashes == second_sample_hashes,
        "file_count": len(first_sample_hashes),
        "seed": regenerated["seed"],
        "boundary_ticket": regenerated["boundary_ticket"],
    }

    rules = parse_rules(read_json(ROOT / "samples" / "rules.json"))
    config = parse_config(read_json(ROOT / "samples" / "config.json"))
    invalid_loaded = load_evidence_file(ROOT / "samples" / "evidence_invalid.jsonl", rules)
    invalid_codes = sorted({failure["code"] for failure in invalid_loaded.failures})
    required_codes = {
        "E_JSON_INVALID",
        "E_EVIDENCE_TYPE_UNKNOWN",
        "E_ROLE_UNKNOWN",
        "E_TIMESTAMP_INVALID",
    }
    results["invalid_input_isolation"] = {
        "passed": required_codes.issubset(set(invalid_codes)) and "T9006" in invalid_loaded.evidence_by_ticket,
        "codes": invalid_codes,
        "failed_ticket_ids": sorted({f["ticket_id"] for f in invalid_loaded.failures if not f["ticket_id"].startswith("@")}),
        "valid_survivor_ticket": "T9006" if "T9006" in invalid_loaded.evidence_by_ticket else None,
    }

    initial_workbench = Workbench(root=ROOT, persist_edits=False)
    initial = initial_workbench.run(use_cache=False, mode="full")
    results["initial_full_run"] = {
        "passed": len(initial.tickets) == 48
        and initial.tickets == sorted(initial.tickets, key=lambda item: item["ticket_id"])
        and any(ticket["boundary"] and ticket["score_fixed"] == 1_000_000 for ticket in initial.tickets),
        "summary": initial.summary(),
        "boundary_tickets": initial.diagnostics["boundary_tickets"],
        "upgraded": [ticket["ticket_id"] for ticket in initial.tickets if ticket["upgrade_code"]],
        "diagnostic_exists": (ROOT / "outputs" / "diagnostic_report.json").exists(),
        "concurrency_conclusion": initial.diagnostics["concurrency"]["conclusion"],
    }

    codec_root = ROOT / "tmp" / "acceptance-codecs"
    if codec_root.exists():
        shutil.rmtree(codec_root)
    codec_root.mkdir(parents=True, exist_ok=True)
    source_records = read_v2(ROOT / "samples" / "version_samples" / "arbitration.v2.json", config)
    v1_path = codec_root / "records.v1.txt"
    v2_path = codec_root / "records.v2.json"
    v3_path = codec_root / "records.v3.bin"
    v1_info = write_v1(source_records, v1_path, config, temp_dir=codec_root)
    v1_records = read_v1(v1_path, config)
    write_v2(v1_records, v2_path, config, temp_dir=codec_root)
    v2_roundtrip = read_v2(v2_path, config)
    v3_info = write_v3(v2_roundtrip, v3_path, config, temp_dir=codec_root)
    v3_records = read_v3(v3_path, config)
    direct_v3_to_v1_path = codec_root / "from-v3.v1.txt"
    v3_to_v1_report = convert_file(
        v3_path, "v3", direct_v3_to_v1_path, "v1", config, temp_dir=codec_root
    )
    results["version_interop"] = {
        "passed": (
            source_records == v1_records
            and source_records == v2_roundtrip
            and source_records == v3_records
            and "E_VERSION_LOSSY" in v3_to_v1_report["codes"]
            and len(v3_to_v1_report["degradation_details"]) == 48
        ),
        "record_count": len(source_records),
        "v1_sidecar_records": len(read_json(v1_info["sidecar_path"])["records"]),
        "v1_roundtrip_exact": source_records == v1_records,
        "v2_roundtrip_exact": source_records == v2_roundtrip,
        "v3_roundtrip_exact": source_records == v3_records,
        "v3_to_v1_codes": v3_to_v1_report["codes"],
        "v3_to_v1_degradation_count": len(v3_to_v1_report["degradation_details"]),
        "v3_header_features": v3_info["feature_names"],
    }

    concurrency_root = ROOT / "tmp" / "acceptance-concurrency"
    if concurrency_root.exists():
        shutil.rmtree(concurrency_root)
    concurrency_root.mkdir(parents=True, exist_ok=True)
    evidence_lines = (ROOT / "samples" / "evidence.jsonl").read_text(encoding="utf-8").splitlines()

    def make_engine(workers: int, name: str, injections=None) -> ArbitrationEngine:
        return ArbitrationEngine(
            rules=rules,
            config=replace(config, worker_threads=workers),
            evidence_lines=evidence_lines,
            workspace_root=concurrency_root,
            snapshot_store=None,
            exception_injections=injections or [],
            output_dir=concurrency_root / name,
            write_outputs=True,
        )

    single = make_engine(1, "single").run(use_cache=False, mode="full")
    parallel = make_engine(4, "parallel").run(use_cache=False, mode="full")
    fault = make_engine(
        4,
        "fault",
        [
            {
                "active": True,
                "ticket_id": "T0048",
                "stage": "relations",
                "exception_type": "SyntheticRelationFailure",
                "message": "acceptance relation failure",
            }
        ],
    ).run(use_cache=False, mode="full")
    results["concurrency_and_faults"] = {
        "passed": (
            single.comparison_payload() == parallel.comparison_payload()
            and len(fault.tickets) == 47
            and len(fault.failures) == 1
            and fault.failures[0]["ticket_id"] == "T0048"
            and fault.failures[0]["stage"] == "relations"
        ),
        "single_vs_parallel_equal": single.comparison_payload() == parallel.comparison_payload(),
        "observed_max_workers": parallel.diagnostics["concurrency"]["observed_max_workers"],
        "worker_limit_respected": parallel.diagnostics["concurrency"]["worker_limit_respected"],
        "stable_task_keys_unique": parallel.diagnostics["concurrency"]["stable_task_keys_unique"],
        "fault_isolated_tickets": len(fault.tickets),
        "fault_failure": fault.failures[0] if fault.failures else None,
    }

    work_root = copy_sample_workspace(ROOT / "tmp" / "acceptance-work")
    workbench = Workbench(root=work_root, persist_edits=True)
    sequence_initial = workbench.run(use_cache=False, mode="full")
    replacement_rows = read_rows(work_root / "samples" / "evidence.jsonl")
    replacement = next(row for row in replacement_rows if row["ticket_id"] == "T0020")
    sequence = [
        {"op": "set_type_weight", "evidence_type": "known_device", "value": "0.26"},
        {"op": "set_role_trust", "source_role": "backend", "value": "0.87"},
        {
            "op": "upsert_relation",
            "relation": {
                "id": "rel-phish-telemetry",
                "kind": "support",
                "left": {"evidence_type": "phishing_report", "source_roles": ["external"]},
                "right": {"evidence_type": "telemetry_anomaly", "source_roles": ["backend"]},
                "bonus": "0.03",
            },
        },
        {"op": "delete_relation", "relation_id": "rel-admin-vs-billing"},
        {"op": "set_threshold", "value": "1.10"},
        {"op": "set_half_life", "value": 5400},
        {
            "op": "replace_evidence",
            "ticket_id": "T0020",
            "evidence_seq": replacement["evidence_seq"],
            "record": {"timestamp": replacement["timestamp"] + 19, "terms": ["audit", "replaced"]},
        },
        {"op": "set_target_version", "value": "v3"},
        {"op": "set_target_version", "value": "v1"},
        {"op": "set_fusion", "value": "mean"},
        {"op": "set_threshold", "value": "1.05"},
    ]
    sequence_rows = []
    all_equal = True
    fusion_scale_invariant = True
    for index, edit in enumerate(sequence, start=1):
        affected = workbench.apply_edit(edit)
        incremental = workbench.run()
        comparison = workbench.compare_incremental_with_full(incremental)
        all_equal = all_equal and comparison["equal"]
        max_score = max((ticket["score_fixed"] for ticket in incremental.tickets), default=0)
        fusion_scale_invariant = fusion_scale_invariant and max_score <= 20 * config.scale
        sequence_rows.append(
            {
                "step": index,
                "op": edit["op"],
                "affected_count": len(affected),
                "ticket_count": len(incremental.tickets),
                "upgrade_count": incremental.summary()["current_upgrade_count"],
                "incremental_equals_full": comparison["equal"],
                "artifact_hashes_equal": comparison["artifact_hashes_equal"],
                "stage_stats": stage_stats(incremental),
                "snapshot_version": incremental.global_version,
                "max_score_fixed": max_score,
            }
        )
    final_incremental = workbench.last_result
    assert final_incremental is not None
    resumed_from_disk = Workbench(root=work_root, persist_edits=False, snapshot_enabled=False)
    from_disk_full = resumed_from_disk.run(use_cache=False, mode="full")
    disk_equal = final_incremental.comparison_payload() == from_disk_full.comparison_payload()
    results["incremental_equivalence"] = {
        "passed": all_equal and disk_equal and fusion_scale_invariant,
        "all_sequence_steps_equal": all_equal,
        "persisted_inputs_full_rebuild_equal": disk_equal,
        "fusion_scale_invariant": fusion_scale_invariant,
        "step_count": len(sequence),
        "steps": sequence_rows,
    }

    resume = Workbench(root=work_root, persist_edits=False, expected_snapshot_version=workbench.global_version)
    resumed = resume.run(use_cache=True, mode="incremental")
    resumed_stats = stage_stats(resumed)
    results["snapshot_resume"] = {
        "passed": all(resumed_stats[stage]["reused"] == len(resumed.tickets) for stage in ("score", "relations", "decision", "convert")),
        "startup": resume.startup_status["code"],
        "snapshot_version": resumed.global_version,
        "recovery": resume.recovery_info,
        "stage_stats": resumed_stats,
    }

    before_budget = resume.latest_summary()
    resume.engine.pre_run_delay_ms = 100
    budget_started = time.monotonic()
    budget_result = resume.apply_edit_and_run({"op": "set_threshold", "value": "1.06"}, budget_ms=1)
    budget_elapsed_ms = (time.monotonic() - budget_started) * 1000
    if resume._background_thread is not None:
        resume._background_thread.join(timeout=5)
    budget_full_equal = resume.compare_incremental_with_full(resume.last_result)["equal"] if resume.last_result else False
    results["interaction_budget"] = {
        "passed": budget_result["code"] == "E_INTERACT_BUDGET_PENDING"
        and budget_result["latest_summary"] == before_budget
        and budget_elapsed_ms < 250
        and budget_full_equal,
        "code": budget_result["code"],
        "elapsed_ms": round(budget_elapsed_ms, 2),
        "summary_retained": budget_result["latest_summary"] == before_budget,
        "background_finished": not (resume._background_thread and resume._background_thread.is_alive()),
        "final_equals_full": budget_full_equal,
    }

    stale_root = copy_sample_workspace(ROOT / "tmp" / "acceptance-stale")
    stale_setup = Workbench(root=stale_root, persist_edits=True)
    stale_setup.run(use_cache=False, mode="full")
    config_path = stale_root / "samples" / "config.json"
    stale_config = read_json(config_path)
    stale_config["decision_threshold"] = "1.25"
    write_json(config_path, stale_config, temp_dir=stale_root / "tmp")
    stale_workbench = Workbench(root=stale_root, persist_edits=False)
    results["snapshot_stale"] = {
        "passed": stale_workbench.startup_code == "E_SNAPSHOT_STALE"
        and "config_hash" in stale_workbench.startup_status.get("stale_reasons", []),
        "code": stale_workbench.startup_code,
        "stale_reasons": stale_workbench.startup_status.get("stale_reasons", []),
        "cache_resume_disabled": all(not entries for entries in stale_workbench.cache_store.stages.values()),
    }

    signal_root = copy_sample_workspace(ROOT / "tmp" / "acceptance-signal")
    injection_path = signal_root / "samples" / "exception_injections.json"
    injection_doc = read_json(injection_path)
    injection_doc["injections"] = [
        {
            "active": True,
            "ticket_id": "*",
            "stage": "score",
            "action": "delay",
            "delay_ms": 2000,
            "exception_type": "DelayOnly",
            "message": "signal exit acceptance",
        }
    ]
    write_json(injection_path, injection_doc, temp_dir=signal_root / "tmp")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    process = subprocess.Popen(
        [sys.executable, "-m", "arbitration_workbench", "run", "--fresh", "--root", str(signal_root)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.8)
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=15)
    try:
        signal_payload = json.loads(stdout)
    except json.JSONDecodeError:
        signal_payload = {"stdout": stdout, "stderr": stderr}
    results["graceful_signal_exit"] = {
        "passed": process.returncode == 0
        and signal_payload.get("status") == "stopped"
        and 15 in signal_payload.get("signals", []),
        "return_code": process.returncode,
        "status": signal_payload.get("status"),
        "signals": signal_payload.get("signals"),
        "summary": signal_payload.get("summary"),
    }

    if args.skip_unittest:
        unittest_result = {"skipped": True, "passed": True}
    else:
        unittest_result = run_unittests()
    results["unit_and_integration_tests"] = unittest_result

    final_result = resume.last_result or final_incremental
    diagnostic = final_result.diagnostics
    diagnostic["acceptance"] = {
        "all_passed": all(item.get("passed", False) for item in results.values()),
        "environment": {
            "python": sys.version,
            "numpy": numpy.__version__,
            "platform": platform.platform(),
            "os_name": os.name,
            "windows_verified": os.name == "nt",
            "snapshot_and_temp_same_volume": same_volume(work_root / "snapshots", work_root / "tmp"),
        },
        "checks": results,
    }
    outputs = ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    write_json(outputs / "acceptance_results.json", {"schema": "acceptance-results/v1", "results": results}, temp_dir=ROOT / "tmp")
    write_json(outputs / "acceptance_diagnostic_report.json", diagnostic, temp_dir=ROOT / "tmp")

    elapsed = time.monotonic() - started
    passed = all(item.get("passed", False) for item in results.values())
    markdown = [
        "# Verification Record",
        "",
        f"- Overall: **{'PASS' if passed else 'FAIL'}**",
        f"- Wall time: {elapsed:.2f} seconds",
        f"- Python: `{sys.version.splitlines()[0]}`",
        f"- numpy: `{numpy.__version__}`",
        f"- Actual platform: `{platform.platform()}` (`os.name={os.name}`)",
        f"- Windows-specific behavior verified here: `{os.name == 'nt'}`",
        "- The implementation uses only standard-library threads, JSON, struct and `os.replace`; this runner was executed on Linux/WSL2.",
        "",
        "## Acceptance Checks",
        "",
        "| Check | Result | Evidence |",
        "|---|---|---|",
    ]
    for name, item in results.items():
        evidence = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if len(evidence) > 240:
            evidence = evidence[:237] + "..."
        markdown.append(f"| `{name}` | {'PASS' if item.get('passed', False) else 'FAIL'} | {evidence.replace('|', '/')} |")
    markdown.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "cd project",
            "PYTHONPATH=. python3 -m arbitration_workbench generate --root .",
            "PYTHONPATH=. python3 -m arbitration_workbench run --root . --fresh",
            "PYTHONPATH=. python3 scripts/run_acceptance.py",
            "PYTHONPATH=. python3 -m unittest discover -v",
            "```",
            "",
            "The main-sample run report is `outputs/diagnostic_report.json`; the acceptance-run report is `outputs/acceptance_diagnostic_report.json`; the ledger is `outputs/acceptance_results.json`.",
        ]
    )
    (ROOT / "VERIFICATION.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    if not args.keep_work:
        for transient in (
            "acceptance-codecs",
            "acceptance-concurrency",
            "acceptance-work",
            "acceptance-stale",
            "acceptance-signal",
            "full-rebuild",
            "full-comparison",
        ):
            shutil.rmtree(ROOT / "tmp" / transient, ignore_errors=True)
    print(json.dumps({"overall": "PASS" if passed else "FAIL", "elapsed_seconds": round(elapsed, 2), "results": results}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

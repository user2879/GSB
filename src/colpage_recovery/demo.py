from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .engine import ReplayEngine
from .errors import DiagnosticError
from .ledger import Ledger
from .loader import build_bundle, inject_payload_fault, load_json_file
from .service import DiagnosticService
from .viewmodel import build_state_view_model


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]


def determinism_probe(sample_dir: str | Path) -> dict[str, Any]:
    pages_payload = load_json_file(Path(sample_dir) / "pages.json")
    wal_payload = load_json_file(Path(sample_dir) / "wal.json")
    budget_path = Path(sample_dir) / "budget.json"
    budget_payload = load_json_file(budget_path) if budget_path.exists() else None
    bundle = build_bundle(pages_payload, wal_payload, budget_payload)
    reference = ReplayEngine(bundle).replay_all(batch_size=bundle.budget.ledger_batch_entries)
    expected = {page_id: digest for page_id, digest in reference.digest_map().items()}
    variants: list[dict[str, Any]] = []
    for batch_size, global_slots, tx_quota in (
        (1, 1, 1),
        (2, 4, 2),
        (3, 3, 1),
        (len(bundle.entries), max(1, bundle.budget.global_slots + 1), bundle.budget.per_tx_quota),
    ):
        variant_budget = replace(
            bundle.budget,
            ledger_batch_entries=max(1, batch_size),
            global_slots=global_slots,
            per_tx_quota=tx_quota,
            work_delay_ms=0,
        )
        variant_bundle = replace(bundle, budget=variant_budget)
        result = ReplayEngine(variant_bundle).replay_all()
        actual = result.digest_map()
        if actual != expected:
            raise AssertionError("nondeterministic page digests across batch/scheduler variants")
        variants.append(
            {
                "batch_size": batch_size,
                "global_slots": global_slots,
                "per_tx_quota": tx_quota,
                "digests_match": True,
                "slot_peak_max": max((batch.slot_peak for batch in result.batches), default=0),
            }
        )
    return {
        "status": "matched",
        "page_version_count": len(reference.records),
        "final_versions": reference.final_versions,
        "variants": variants,
    }


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + existing if existing else "")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_end_to_end_demo(
    sample_dir: str | Path = "samples",
    state_dir: str | Path = "state/demo",
    *,
    report_path: str | Path | None = "artifacts/demo_report.json",
) -> dict[str, Any]:
    sample_dir = Path(sample_dir).resolve()
    state_dir = Path(state_dir).resolve()
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True)

    determinism = determinism_probe(sample_dir)
    command = [
        sys.executable,
        "-m",
        "colpage_recovery.cli",
        "run",
        "--samples",
        str(sample_dir),
        "--state-dir",
        str(state_dir),
        "--reset",
        "--crash-before-commit",
        "2",
    ]
    completed = subprocess.run(
        command,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    db_path = state_dir / "recovery.db"
    killed_batches: list[dict[str, Any]] = []
    if db_path.exists():
        with Ledger(db_path) as ledger:
            ledger.migrate()
            generation = ledger.current_generation()
            killed_batches = ledger.query_dicts(
                "SELECT round_no,attempt_no,status,start_lsn,end_lsn "
                "FROM replay_batch WHERE generation=? ORDER BY round_no,attempt_no",
                (generation,),
            )

    with DiagnosticService(state_dir=state_dir, sample_dir=sample_dir) as service:
        service.initialize(reset=False)
        recovery = service.recover_and_verify()

    evolution: dict[str, Any] = {}
    with DiagnosticService(state_dir=state_dir, sample_dir=sample_dir) as service:
        original = service.bundle
        evolved_budget = replace(
            original.budget,
            global_slots=original.budget.global_slots + 1,
            checkpoint_interval=original.budget.checkpoint_interval,
        )
        service.bundle = replace(original, budget=evolved_budget)
        service.engine = ReplayEngine(service.bundle)
        evolved_generation = service.initialize(reset=False)
        evolved_run = service.replay_and_commit()
        evolution = {
            "from_generation": 1,
            "to_generation": evolved_generation,
            "monotonic_increment": evolved_generation == 2,
            "evolved_global_slots": evolved_budget.global_slots,
            "run": evolved_run,
        }

    state_view = build_state_view_model(state_dir)
    report = {
        "ok": True,
        "sample_dir": str(sample_dir),
        "state_dir": str(state_dir),
        "determinism": determinism,
        "force_kill": {
            "command": command,
            "returncode": completed.returncode,
            "nonzero_exit_as_expected": completed.returncode != 0,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
            "batches_immediately_after_kill": killed_batches,
        },
        "recovery": recovery,
        "spec_evolution": evolution,
        "state_view": state_view,
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    return report


def validation_injection_report(
    sample_dir: str | Path, injection: str
) -> dict[str, Any]:
    pages_payload = load_json_file(Path(sample_dir) / "pages.json")
    wal_payload = load_json_file(Path(sample_dir) / "wal.json")
    budget_path = Path(sample_dir) / "budget.json"
    budget_payload = load_json_file(budget_path) if budget_path.exists() else None
    try:
        mutated_pages, mutated_wal = inject_payload_fault(
            pages_payload, wal_payload, injection
        )
        build_bundle(mutated_pages, mutated_wal, budget_payload)
    except DiagnosticError as error:
        return {"ok": False, "injection": injection, "error": error.to_dict()}
    return {
        "ok": True,
        "injection": injection,
        "error": {"code": "E_INJECTION_NOT_DETECTED"},
    }


def snapshot_injection_report(
    sample_dir: str | Path = "samples",
    state_dir: str | Path = "state/injection-snapshot",
) -> dict[str, Any]:
    state_dir = Path(state_dir)
    if state_dir.exists():
        shutil.rmtree(state_dir)
    try:
        with DiagnosticService(state_dir=state_dir, sample_dir=sample_dir) as service:
            service.initialize(reset=True)
            service.replay_and_commit()
            ledger = service.require_ledger()
            tampered = ledger.tamper_content_digest(service.generation or 1)
            service.recover_and_verify()
    except DiagnosticError as error:
        return {
            "ok": False,
            "injection": "snapshot-mismatch",
            "tampered": tampered if "tampered" in locals() else None,
            "error": error.to_dict(),
        }
    return {
        "ok": True,
        "injection": "snapshot-mismatch",
        "error": {"code": "E_INJECTION_NOT_DETECTED"},
    }


def injection_report(
    injection: str,
    sample_dir: str | Path = "samples",
    state_dir: str | Path = "state/injection",
) -> dict[str, Any]:
    if injection == "snapshot-mismatch":
        return snapshot_injection_report(sample_dir, state_dir)
    return validation_injection_report(sample_dir, injection)

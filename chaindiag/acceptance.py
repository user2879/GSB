from __future__ import annotations

import copy
import json
import sys
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np

from .exchange import e1_to_e2, e2_to_e1
from .ledger import CacheStore
from .model import validate_chain
from .pipeline import compare_reports, engine_from_data, state_to_engine
from .samplegen import default_bundle, write_sample_bundle
from .runtime import verify_write_permission
from .snapshot import SnapshotStore
from .state import apply_commands


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fresh_state(bundle: dict[str, Any]) -> dict[str, Any]:
    chain = copy.deepcopy(bundle["chain_e2"])
    return {
        "config": copy.deepcopy(bundle["config"]),
        "chain": chain,
        "operators": chain["operators"],
        "tokens": copy.deepcopy(bundle["tokens"]),
        "boundaries": copy.deepcopy(bundle["segments"]["boundaries"]),
    }


def run_roundtrip(bundle: dict[str, Any]) -> dict[str, Any]:
    original_chain = bundle["chain_e2"]
    e1_text, sidecar = e2_to_e1(original_chain)
    restored_chain = e1_to_e2(e1_text, sidecar)
    validate_chain(restored_chain)

    original = engine_from_data(
        config=bundle["config"], chain=original_chain, tokens=bundle["tokens"],
        segments=bundle["segments"]["boundaries"], store=CacheStore(), serial=True,
    ).run(with_width_reference=False)
    restored = engine_from_data(
        config=bundle["config"], chain=restored_chain, tokens=bundle["tokens"],
        segments=bundle["segments"]["boundaries"], store=CacheStore(), serial=True,
    ).run(with_width_reference=False)
    comparison = compare_reports(original, restored)
    return {
        "e1_text": e1_text,
        "sidecar": sidecar,
        "restored_chain": restored_chain,
        "comparison": comparison,
        "restored_operator_count": len(restored_chain["operators"]),
    }


def run_edit_sequence(bundle: dict[str, Any]) -> dict[str, Any]:
    state = _fresh_state(bundle)
    store = CacheStore()
    state_to_engine(state, store=store, serial=True).run(with_width_reference=False)
    steps: list[dict[str, Any]] = []
    engine = None
    for command in bundle["edits"]["commands"]:
        summary = apply_commands(state, [command])[0]
        engine = state_to_engine(state, store=store, serial=True)
        report = engine.run(with_width_reference=False)
        steps.append(
            {
                "command": command["op"],
                "summary": summary,
                "status": report["status"],
                "computed": sum(e["action"] == "computed" for e in report["reuse_ledger"]),
                "reused": sum(e["action"] == "reused" for e in report["reuse_ledger"]),
                "pending": len(report["scheduler"]["pending"]),
            }
        )
    final_rerun = engine.run(with_width_reference=False)
    all_reused = all(entry["action"] == "reused" for entry in final_rerun["reuse_ledger"])

    clean_state = _fresh_state(bundle)
    apply_commands(clean_state, bundle["edits"]["commands"])
    full_report = state_to_engine(clean_state, store=CacheStore(), serial=True).run(with_width_reference=False)
    comparison = compare_reports(final_rerun, full_report)
    return {
        "steps": steps,
        "second_run_all_reused": all_reused,
        "final_vs_full_rebuild": comparison,
        "incremental_report": final_rerun,
        "full_report": full_report,
    }


def run_budget_cases(bundle: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, overrides in {
        "pending": {"point_budget": 11},
        "quota": {"operator_quota": 3},
        "healthy": {},
    }.items():
        serial = engine_from_data(
            config=bundle["config"], chain=bundle["chain_e2"], tokens=bundle["tokens"],
            segments=bundle["segments"]["boundaries"], store=CacheStore(), serial=True, overrides=overrides,
        ).run(with_width_reference=False)
        concurrent = engine_from_data(
            config=bundle["config"], chain=bundle["chain_e2"], tokens=bundle["tokens"],
            segments=bundle["segments"]["boundaries"], store=CacheStore(), serial=False, overrides=overrides,
        ).run(with_width_reference=False)
        output[name] = {
            "overrides": overrides,
            "comparison": compare_reports(serial, concurrent, scheduler=True),
            "serial_status": serial["scheduler"]["status"],
            "consumed_budget": serial["scheduler"]["consumed_budget"],
            "pending_count": len(serial["scheduler"]["pending"]),
            "diagnostics": serial["scheduler"]["diagnostics"][:4],
            "max_observed_workers": concurrent["scheduler"]["max_observed_workers"],
        }
    return output


def run_acceptance(output_dir: str | Path = "artifacts/verification") -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_check = verify_write_permission(output)
    sample_paths = write_sample_bundle("samples")
    bundle = default_bundle()

    healthy_store = CacheStore()
    healthy = engine_from_data(
        config=bundle["config"], chain=bundle["chain_e2"], tokens=bundle["tokens"],
        segments=bundle["segments"]["boundaries"], store=healthy_store, serial=False,
    ).run(with_width_reference=True)
    _write_json(output / "healthy.json", healthy)
    _write_json(output / "reuse_ledger.json", healthy["reuse_ledger"])
    healthy_store.save(output / "cache_ledger.json")

    serial = engine_from_data(
        config=bundle["config"], chain=bundle["chain_e2"], tokens=bundle["tokens"],
        segments=bundle["segments"]["boundaries"], store=CacheStore(), serial=True,
    ).run(with_width_reference=False)
    concurrency = compare_reports(serial, healthy, scheduler=False)

    faults = engine_from_data(
        config=bundle["config"], chain=bundle["chain_e2"], tokens=bundle["tokens"],
        segments=bundle["segments"]["boundaries"], injections=bundle["injections"]["injections"],
        store=healthy_store, serial=True,
    ).run(with_width_reference=False)
    _write_json(output / "faults.json", faults)
    _write_json(output / "fault-report.json", faults["fault_report"])
    healthy_by_uid = {token["token_uid"]: token for token in healthy["token_results"]}
    fault_by_uid = {token["token_uid"]: token for token in faults["token_results"]}
    targeted = {uid for report in faults["fault_report"] for uid in report["token_uids"]}
    unaffected_equal = all(
        fault_by_uid[uid]["vector"] == healthy_by_uid[uid]["vector"]
        and fault_by_uid[uid]["cache_keys"] == healthy_by_uid[uid]["cache_keys"]
        for uid in healthy_by_uid
        if uid not in targeted
    )

    roundtrip = run_roundtrip(bundle)
    _write_json(output / "roundtrip.json", {k: v for k, v in roundtrip.items() if k not in ("sidecar", "restored_chain")})
    (output / "chain.e1.roundtrip.txt").write_text(roundtrip["e1_text"], encoding="utf-8")
    _write_json(output / "chain.e1.sidecar.json", roundtrip["sidecar"])
    _write_json(output / "chain.e2.restored.json", roundtrip["restored_chain"])

    edits = run_edit_sequence(bundle)
    _write_json(output / "incremental-final.json", edits["incremental_report"])
    _write_json(output / "full-rebuild.json", edits["full_report"])
    _write_json(output / "edit-steps.json", edits["steps"])

    budget_cases = run_budget_cases(bundle)
    _write_json(output / "scheduler-cases.json", budget_cases)

    snapshot_root = output / "snapshot"
    snapshot = SnapshotStore(snapshot_root)
    manifest = snapshot.commit_report(
        healthy, config_hash=healthy["config_hash"], exchange_version=healthy["exchange_version"]
    )
    recovery = snapshot.recover(config_hash=healthy["config_hash"], exchange_version=healthy["exchange_version"]).to_dict()

    checks = {
        "write_permission": write_check["writable"] is True,
        "samples_generated": len(sample_paths) >= 8,
        "healthy_status_ok": healthy["status"] == "ok",
        "serial_vs_concurrent_equal": concurrency["equal"],
        "roundtrip_equal": roundtrip["comparison"]["equal"],
        "incremental_vs_full_equal": edits["final_vs_full_rebuild"]["equal"],
        "incremental_second_run_all_reused": edits["second_run_all_reused"],
        "fault_injections_all_hit": len(faults["fault_report"]) == 3,
        "fault_unaffected_equal": unaffected_equal,
        "budget_serial_concurrent_equal": all(case["comparison"]["equal"] for case in budget_cases.values()),
        "snapshot_recovery": recovery["resume"] is True,
        "width_reference_available": healthy["width_reference"]["available"] is True,
    }
    report = {
        "status": "ok" if all(checks.values()) else "failed",
        "checks": checks,
        "write_permission": write_check,
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "numpy": np.__version__,
            "pytest": package_version("pytest"),
        },
        "healthy_summary": {
            "token_count": len(healthy["token_results"]),
            "cache_record_count": len(healthy["cache_records"]),
            "config_hash": healthy["config_hash"],
            "chain_fingerprint": healthy["chain_fingerprint"],
            "first_width_exceeded": healthy["width_reference"]["first_exceeded_stage"],
            "scheduler": {k: healthy["scheduler"][k] for k in ("max_workers", "max_observed_workers", "consumed_budget", "status")},
        },
        "concurrency_comparison": concurrency,
        "roundtrip_comparison": roundtrip["comparison"],
        "edit_summary": {
            "steps": edits["steps"],
            "second_run_all_reused": edits["second_run_all_reused"],
            "final_vs_full_rebuild": edits["final_vs_full_rebuild"],
        },
        "fault_summary": {
            "report": faults["fault_report"],
            "targeted_token_uids": sorted(targeted),
            "unaffected_equal": unaffected_equal,
        },
        "budget_cases": budget_cases,
        "snapshot_manifest": manifest,
        "snapshot_recovery": recovery,
        "artifact_dir": str(output),
    }
    _write_json(output / "acceptance_report.json", report)
    return report

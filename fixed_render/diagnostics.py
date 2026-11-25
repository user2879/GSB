from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from .config import RenderConfig
from .hashing import fingerprint_json
from .lease import LeaseClient, LeaseServer, now_ms
from .miniyaml import dump_yaml
from .storage import atomic_write_json
from .workbench import CommandSession


INJECTIONS: list[dict[str, Any]] = [
    {
        "id": "INJ-DIM-001",
        "fault": "dim_mismatch",
        "target": "encode:text",
        "expected_code": "E_DIM_MISMATCH",
        "field_name": "condition_vector",
        "expected_dim": 24,
        "actual_dim": 299,
        "description": "replace text condition with a 299-dimensional vector before dimension check",
    },
    {
        "id": "INJ-NAN-002",
        "fault": "nonfinite_latent",
        "target": "denoise:step:0001",
        "expected_code": "E_NONFINITE_LATENT",
        "shape": [2, 16, 16, 4],
        "description": "poison step input latent with a real NaN floating-point entry",
    },
    {
        "id": "INJ-TILE-003",
        "fault": "tile_out_of_range",
        "target": "tile:frame:00:row:00",
        "expected_code": "E_TILE_OUT_OF_RANGE",
        "tile_index": 99,
        "tile_count": 8,
        "description": "request tile row 99 when only eight row bands exist",
    },
    {
        "id": "INJ-QUANT-004",
        "fault": "quant_invalid",
        "target": "fuse",
        "expected_code": "E_QUANT_INVALID",
        "frac_bits": 99,
        "mode": "unsupported_mode",
        "description": "switch to an out-of-enum fixed fractional width and quant mode",
    },
    {
        "id": "INJ-LEASE-005",
        "fault": "lease_stale",
        "target": "fuse",
        "expected_code": "E_LEASE_STALE",
        "generation_offset": 1,
        "description": "authorize one stage with a generation newer than the recognized token",
    },
]


def run_diagnostics(root: Path, report_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = root / "configs" / "render.yaml"
    config = RenderConfig.load(config_path)
    injection_path = root / "diagnostics" / "injection_manifest.yaml"
    injection_path.parent.mkdir(parents=True, exist_ok=True)
    injection_path.write_text(
        dump_yaml({"manifest_fingerprint": fingerprint_json(INJECTIONS), "injections": INJECTIONS}),
        encoding="utf-8",
    )

    cases: list[dict[str, Any]] = []
    aggregate_codes: dict[str, int] = {}
    aggregate_reused = 0
    aggregate_recomputed = 0
    started_all = time.perf_counter()

    for injection in INJECTIONS:
        workspace = root / "runs" / "diagnostics" / injection["id"].lower().replace("-", "_")
        if workspace.exists():
            shutil.rmtree(workspace)
        case_started = time.perf_counter()
        with CommandSession(root, workspace, config) as session:
            result = session.render_current(injection=injection, ledger_name=injection["id"])
            manifest = result["manifest"]
            ledger = result["ledger"]
            matching = [
                entry
                for entry in ledger["entries"]
                if entry.get("injection_id") == injection["id"]
                and entry.get("business_code") == injection["expected_code"]
            ]
            hit = bool(matching)
            for code, count in ledger.get("business_code_counts", {}).items():
                aggregate_codes[code] = aggregate_codes.get(code, 0) + int(count)
            aggregate_reused += int(ledger["counts"]["reused"])
            aggregate_recomputed += int(ledger["counts"]["recomputed"])
            preserved = [
                entry["node_id"]
                for entry in ledger["entries"]
                if entry["status"] in {"recomputed", "reused"}
            ]
            cases.append(
                {
                    "injection_id": injection["id"],
                    "fault": injection["fault"],
                    "target": injection["target"],
                    "expected_code": injection["expected_code"],
                    "hit": hit,
                    "hit_entry": matching[0] if matching else None,
                    "description": injection["description"],
                    "workspace": str(workspace),
                    "ledger_path": manifest["ledger_path"],
                    "stage_timings_ms": ledger["timings_ms"],
                    "lease_generation_trajectory": ledger["lease_trajectory"],
                    "counts": ledger["counts"],
                    "recompute_order": ledger["recompute_order"],
                    "reuse_order": ledger["reuse_order"],
                    "failed_order": ledger["failed_order"],
                    "blocked_order": ledger["blocked_order"],
                    "pending": manifest["pending"],
                    "business_code_counts": ledger["business_code_counts"],
                    "preserved_completed_nodes": preserved,
                    "case_elapsed_ms": (time.perf_counter() - case_started) * 1000,
                }
            )

    lease_audit = audit_lease_reclamation()
    report = {
        "report_type": "fixed_render_diagnostics",
        "generated_at_ms": now_ms(),
        "injection_manifest": str(injection_path),
        "injection_manifest_fingerprint": fingerprint_json(INJECTIONS),
        "lease_reclamation": lease_audit,
        "cases": cases,
        "summary": {
            "total_injections": len(INJECTIONS),
            "hit_injections": sum(1 for case in cases if case["hit"]),
            "failed_cases": sum(1 for case in cases if not case["hit"]),
            "aggregate_business_code_counts": aggregate_codes,
            "aggregate_reused": aggregate_reused,
            "aggregate_recomputed": aggregate_recomputed,
            "lease_slot_released": lease_audit["slot_released"],
            "lease_expired_code_observed": lease_audit["expired_code_observed"],
            "lease_stale_code_observed": lease_audit["stale_code_observed"],
            "elapsed_ms": (time.perf_counter() - started_all) * 1000,
        },
    }
    report_path = Path(report_path)
    if not report_path.is_absolute():
        report_path = root / report_path
    atomic_write_json(report_path, report)
    return report


def audit_lease_reclamation(ttl_ms: int = 60) -> dict[str, Any]:
    with LeaseServer("127.0.0.1", 0, max_slots=1) as server:
        client = LeaseClient("127.0.0.1", server.port)
        issued = client.acquire(
            lease_key="acceptance:reclamation",
            scopes=["acceptance"],
            ttl_ms=ttl_ms,
            holder="verifier",
        )
        before = client.snapshot()
        deadline = time.perf_counter() + 2.0
        after = before
        while time.perf_counter() < deadline:
            after = client.snapshot()
            if after["used_slots"] == 0:
                break
            time.sleep(0.01)

        expired_code = None
        try:
            client.renew(issued["lease_id"], int(issued["generation"]), ttl_ms)
        except Exception as exc:
            expired_code = getattr(exc, "code", type(exc).__name__)

        replacement = client.acquire(
            lease_key="acceptance:reclamation",
            scopes=["acceptance"],
            ttl_ms=1000,
            holder="verifier2",
        )
        stale_code = None
        try:
            client.authorize(replacement["lease_id"], int(replacement["generation"]) + 1)
        except Exception as exc:
            stale_code = getattr(exc, "code", type(exc).__name__)
        final_snapshot = client.snapshot()
        return {
            "ttl_ms": ttl_ms,
            "issued": issued,
            "used_slots_before_expiry": before["used_slots"],
            "used_slots_after_reaper": after["used_slots"],
            "reaper_observed_release": after["used_slots"] == 0,
            "slot_released": before["used_slots"] == 1 and after["used_slots"] == 0,
            "expired_code_observed": expired_code == "E_LEASE_EXPIRED",
            "stale_code_observed": stale_code == "E_LEASE_STALE",
            "expired_code": expired_code,
            "stale_code": stale_code,
            "replacement_generation": replacement["generation"],
            "generation_monotonic": replacement["generation"] > issued["generation"],
            "executor_events": final_snapshot["events"],
        }

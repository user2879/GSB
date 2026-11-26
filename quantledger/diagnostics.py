"""Fault-injection diagnostics over the real worker protocol."""

from __future__ import annotations

import copy
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Any

from .config import RuntimeState
from .engine import LedgerEngine
from .errors import E_NONFINITE_WEIGHT, E_QUANT_LEVEL, E_STALE_GENERATION, QLabError
from .protocol import JsonLineClient
from .ledger import empty_ledger, recovery_report
from .util import atomic_write_json, read_json


def _valid_submit_payload(state: RuntimeState, batch_id: str | None = None, *, ttl_ms: int = 5000) -> dict[str, Any]:
    batch = next((item for item in state.batches if item.batch_id == batch_id), state.batches[0])
    layers = [
        {
            "name": layer.name,
            "rows": layer.rows,
            "cols": layer.cols,
            "group_size": layer.group_size,
            "weight": layer.weight.astype(float).tolist(),
            "weight_seed": layer.weight_seed,
            **({"bias": layer.bias.astype(float).tolist()} if layer.bias is not None else {}),
        }
        for layer in state.layers
    ]
    return {
        "op": "submit",
        "batch_id": batch.batch_id,
        "samples": batch.samples.astype(float).tolist(),
        "layers": layers,
        "revision": state.revision,
        "frac_bits": state.frac_bits,
        "quant_level": state.quant_level,
        "ttl_ms": ttl_ms,
    }


def _submit_and_release(client: JsonLineClient, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.request(payload)
    if not response.get("ok"):
        return response
    lease = response["lease"]
    lease_id = lease["lease_id"]
    generation = lease["generation"]
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        fetched = client.request({"op": "fetch", "lease_id": lease_id})
        if not fetched.get("ok") or fetched.get("status") == "completed":
            break
        time.sleep(0.002)
    release = client.request(
        {
            "op": "release",
            "lease_id": lease_id,
            "generation": generation,
            "revision": payload.get("revision"),
        }
    )
    return {"submit": response, "fetch": fetched, "release": release}


def run_fault_injections(
    config_path: str | Path,
    injections_path: str | Path,
    *,
    runtime_dir: str | Path = "runtime/diagnostics",
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    engine = LedgerEngine(config_path, runtime_dir=runtime_dir)
    state = engine.base_state.clone()
    raw = read_json(injections_path)
    injections = raw.get("injections", raw) if isinstance(raw, dict) else raw
    if not isinstance(injections, list):
        raise QLabError("E_CONFIG", "injection configuration must contain an injections list")
    results: list[dict[str, Any]] = []
    with engine.workers() as handles:
        worker_id = sorted(handles)[0]
        if isinstance(injections and injections[0], dict) and injections[0].get("worker_id"):
            worker_id = str(injections[0]["worker_id"])
        handle = handles[worker_id]
        client = handle.client
        default_batch = sorted(batch.batch_id for batch in state.batches)[0]
        for index, injection in enumerate(injections, start=1):
            injection_id = str(injection.get("id", f"INJ-{index:03d}"))
            kind = str(injection.get("kind", injection.get("type", "")))
            batch_id = str(injection.get("batch_id", default_batch))
            expected = {
                "dimension_mismatch": "E_DIMENSION",
                "level_out_of_range": E_QUANT_LEVEL,
                "nonfinite_weight": E_NONFINITE_WEIGHT,
                "stale_generation": E_STALE_GENERATION,
            }.get(kind)
            payload = _valid_submit_payload(state, batch_id)
            observed = None
            details: dict[str, Any] = {"worker_id": worker_id, "batch_id": batch_id}
            try:
                if kind == "dimension_mismatch":
                    broken = copy.deepcopy(payload)
                    broken["samples"] = [row[:-1] for row in broken["samples"]]
                    response = client.request(broken)
                    observed = response.get("code")
                    details["response"] = response
                elif kind == "level_out_of_range":
                    broken = copy.deepcopy(payload)
                    broken["quant_level"] = int(injection.get("value", 99))
                    response = client.request(broken)
                    observed = response.get("code")
                    details["response"] = response
                elif kind == "nonfinite_weight":
                    broken = copy.deepcopy(payload)
                    layer_name = injection.get("layer")
                    target_index = 0
                    if layer_name:
                        target_index = next(
                            (i for i, layer in enumerate(broken["layers"]) if layer["name"] == layer_name),
                            0,
                        )
                    broken["layers"][target_index]["weight"][0][0] = injection.get("value", "Infinity")
                    response = client.request(broken)
                    observed = response.get("code")
                    details["response"] = response
                elif kind == "stale_generation":
                    valid = client.request(payload)
                    if not valid.get("ok"):
                        raise QLabError(
                            str(valid.get("code", "E_PROTOCOL")),
                            str(valid.get("message", "could not lease for stale-generation injection")),
                        )
                    lease = valid["lease"]
                    stale = client.request(
                        {
                            "op": "renew",
                            "lease_id": lease["lease_id"],
                            "generation": int(lease["generation"]) - 1,
                            "ttl_ms": 1000,
                            "revision": state.revision,
                        }
                    )
                    observed = stale.get("code")
                    status = client.request({"op": "status"})
                    current_generation = None
                    for active in status.get("status", {}).get("active", []):
                        if active.get("lease_id") == lease["lease_id"]:
                            current_generation = active.get("generation")
                    release = client.request(
                        {
                            "op": "release",
                            "lease_id": lease["lease_id"],
                            "generation": lease["generation"],
                            "revision": state.revision,
                        }
                    )
                    details.update(
                        {
                            "response": stale,
                            "generation_before": lease["generation"],
                            "generation_after": current_generation,
                            "generation_unchanged": current_generation == lease["generation"],
                            "release": release,
                        }
                    )
                else:
                    observed = "E_UNKNOWN_INJECTION"
                    details["message"] = f"unsupported injection kind {kind!r}"
            except Exception as exc:
                observed = getattr(exc, "code", "E_INJECTION_EXECUTION")
                details["exception"] = f"{type(exc).__name__}: {exc}"
            hit = observed == expected and (kind != "stale_generation" or details.get("generation_unchanged") is True)
            results.append(
                {
                    "id": injection_id,
                    "kind": kind,
                    "expected_code": expected,
                    "observed_code": observed,
                    "hit": hit,
                    "details": details,
                }
            )

        isolation = _submit_and_release(client, _valid_submit_payload(state))
        isolation_ok = bool(
            isolation["submit"].get("ok")
            and isolation.get("fetch", {}).get("ok")
            and isolation.get("release", {}).get("ok")
        )
        worker_status = client.request({"op": "status"}).get("status", {})

    report = {
        "schema": "quantledger-fault-report-v1",
        "injections": results,
        "injection_count": len(results),
        "hit_count": sum(1 for item in results if item["hit"]),
        "all_hit": all(item["hit"] for item in results),
        "isolation": {
            "valid_request_after_faults": isolation_ok,
            "active_leases_after": worker_status.get("active_leases"),
            "running_after": worker_status.get("running"),
        },
        "worker_status": worker_status,
    }
    destination = Path(runtime_dir) / "fault-report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, report)
    report["report_path"] = str(destination)
    return report


def run_lease_checks(
    config_path: str | Path,
    *,
    runtime_dir: str | Path = "runtime/diagnostics/leases",
) -> dict[str, Any]:
    """Exercise real expiry, real slot reclamation, renew and stale-generation rejection."""

    engine = LedgerEngine(config_path, runtime_dir=runtime_dir)
    state = engine.base_state.clone()
    report: dict[str, Any] = {}
    with engine.workers() as handles:
        handle = handles[sorted(handles)[0]]
        client = handle.client

        expiring = _valid_submit_payload(state, ttl_ms=100)
        expiring["compute_delay_ms"] = 750
        submitted = client.request(expiring)
        if not submitted.get("ok"):
            raise QLabError(str(submitted.get("code")), str(submitted.get("message")))
        lease = submitted["lease"]
        time.sleep(0.20)
        fetch = client.request({"op": "fetch", "lease_id": lease["lease_id"]})
        status_after_expiry = client.request({"op": "status"}).get("status", {})
        events = client.request({"op": "events"}).get("events", [])
        expiry_events = [event for event in events if event.get("code") == "E_LEASE_EXPIRED"]
        reconcile_events = [event for event in events if event.get("code") == "E_RECONCILE_PENDING"]
        good_after_expiry = _submit_and_release(client, _valid_submit_payload(state))
        report["expiry"] = {
            "generation": lease["generation"],
            "fetch_code": fetch.get("code"),
            "E_LEASE_EXPIRED": len(expiry_events),
            "E_RECONCILE_PENDING": len(reconcile_events),
            "active_leases_after": status_after_expiry.get("active_leases"),
            "running_after": status_after_expiry.get("running"),
            "slot_reclaimed": len(expiry_events) == 1
            and status_after_expiry.get("active_leases") == 0
            and bool(good_after_expiry["submit"].get("ok")),
            "event": expiry_events[0] if expiry_events else None,
        }

        renewed_payload = _valid_submit_payload(state, ttl_ms=120)
        renewed_payload["compute_delay_ms"] = 180
        renewed = client.request(renewed_payload)
        if not renewed.get("ok"):
            raise QLabError(str(renewed.get("code")), str(renewed.get("message")))
        renewed_lease = renewed["lease"]
        time.sleep(0.04)
        renew_response = client.request(
            {
                "op": "renew",
                "lease_id": renewed_lease["lease_id"],
                "generation": renewed_lease["generation"],
                "ttl_ms": 1000,
                "revision": state.revision,
            }
        )
        time.sleep(0.25)
        renewed_fetch = client.request({"op": "fetch", "lease_id": renewed_lease["lease_id"]})
        stale_release = client.request(
            {
                "op": "release",
                "lease_id": renewed_lease["lease_id"],
                "generation": int(renewed_lease["generation"]) - 1,
                "revision": state.revision,
            }
        )
        status_after_stale = client.request({"op": "status"}).get("status", {})
        current_generation = next(
            (
                item.get("generation")
                for item in status_after_stale.get("active", [])
                if item.get("lease_id") == renewed_lease["lease_id"]
            ),
            None,
        )
        current_release = client.request(
            {
                "op": "release",
                "lease_id": renewed_lease["lease_id"],
                "generation": renewed_lease["generation"],
                "revision": state.revision,
            }
        )
        report["renewal"] = {
            "renew_ok": bool(renew_response.get("ok")),
            "completed_after_renew": renewed_fetch.get("status") == "completed",
            "stale_release_code": stale_release.get("code"),
            "generation_before": renewed_lease["generation"],
            "generation_after_stale_release": current_generation,
            "generation_unchanged": current_generation == renewed_lease["generation"],
            "correct_release_ok": bool(current_release.get("ok")),
        }
        final_status = client.request({"op": "status"}).get("status", {})
        report["final_status"] = final_status
    report["ok"] = bool(report["expiry"]["slot_reclaimed"] and report["renewal"]["completed_after_renew"] and report["renewal"]["stale_release_code"] == "E_STALE_GENERATION" and report["renewal"]["generation_unchanged"] and report["renewal"]["correct_release_ok"])
    atomic_write_json(Path(runtime_dir) / "lease-report.json", report)
    return report


def run_forced_kill_recovery(
    config_path: str | Path,
    *,
    seed_ledger: str | Path | None = None,
    runtime_dir: str | Path = "runtime/diagnostics/kill-recovery",
) -> dict[str, Any]:
    """Commit a pending ledger, kill a worker with SIGKILL, then recover from ledger only."""

    destination = Path(runtime_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    ledger_path = destination / "ledger.json"
    if seed_ledger is not None and Path(seed_ledger).exists():
        shutil.copyfile(seed_ledger, ledger_path)
    else:
        engine_for_state = LedgerEngine(config_path, runtime_dir=destination)
        ledger = empty_ledger(engine_for_state.base_state.runtime_config_hash())
        ledger["state"] = {
            "workers": [worker.to_dict() for worker in engine_for_state.base_state.workers],
            "operator": engine_for_state.base_state.operator_dict(include_weights=True),
            "batches": [batch.to_dict() for batch in engine_for_state.base_state.batches],
        }
        atomic_write_json(ledger_path, ledger)

    engine = LedgerEngine(config_path, runtime_dir=destination, ledger_path=ledger_path)
    engine.ledger = engine.store.load() or empty_ledger(engine.base_state.runtime_config_hash())
    state = engine._replay(engine.ledger.get("applied_commands", []))
    batch = state.batches[0]
    old = engine.ledger.get("run", {}).get("batches", {}).get(batch.batch_id, {})
    prefix = copy.deepcopy(old.get("layers", [])[:1])
    activation = batch.samples if not prefix else __import__("numpy").asarray(prefix[-1]["dequantized_output"], dtype=float)
    prepared = {
        batch.batch_id: {
            "batch": batch,
            "state": state,
            "start_layer": len(prefix),
            "prefix_records": prefix,
            "activation": activation,
            "reason": "forced_kill_test",
        }
    }
    engine._mark_pending(prepared, state, int(engine.ledger["applied_commands"][-1]["seq"]) if engine.ledger["applied_commands"] else 0)

    killed_pid = None
    with engine.workers() as handles:
        handle = handles[sorted(handles)[0]]
        payload = _valid_submit_payload(state, ttl_ms=10_000)
        payload["compute_delay_ms"] = 10_000
        submitted = handle.client.request(payload)
        if not submitted.get("ok"):
            raise QLabError(str(submitted.get("code")), str(submitted.get("message")))
        killed_pid = handle.process.pid
        os.kill(killed_pid, signal.SIGKILL)
        handle.process.wait(timeout=5.0)

    ledger_after_kill = engine.store.load()
    report = recovery_report(ledger_after_kill, ledger_path=ledger_path)
    report.update(
        {
            "schema": "quantledger-kill-recovery-report-v1",
            "worker_force_killed": killed_pid is not None,
            "killed_worker_pid": killed_pid,
            "ledger_commit_generation": ledger_after_kill.get("commit_generation"),
        }
    )
    atomic_write_json(destination / "recovery-report.json", report)
    return report


def run_reuse_check(
    config_path: str | Path,
    *,
    runtime_dir: str | Path = "runtime/diagnostics/reuse",
) -> dict[str, Any]:
    """Prove that replacing one layer reuses the untouched upstream prefix."""

    destination = Path(runtime_dir).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    command_path = destination / "commands.json"
    atomic_write_json(
        command_path,
        {"commands": [{"seq": 1, "type": "replace_weight", "layer": "l2", "weight_seed": 987654321}]},
    )
    engine = LedgerEngine(config_path, commands_path=command_path, runtime_dir=destination, request_timeout_s=20)
    engine.run(rebuild=True)
    ledger = engine.store.load()
    reuse_events = [event for event in ledger["events"] if event.get("kind") == "reuse"]
    invalidated_layers = [
        event for event in ledger["events"] if event.get("kind") == "invalidate" and event.get("entity") == "layer"
    ]
    invalidated_ledgers = [
        event
        for event in ledger["events"]
        if event.get("kind") == "invalidate" and event.get("entity") == "ledger_entry"
    ]
    report = {
        "schema": "quantledger-reuse-report-v1",
        "ok": bool(
            reuse_events
            and {event["layer"] for event in reuse_events} == {"l1"}
            and {event["layer"] for event in invalidated_layers} == {"l2", "l3"}
            and invalidated_ledgers
        ),
        "reuse_entries": len(reuse_events),
        "reused_layers": sorted({event["layer"] for event in reuse_events}),
        "invalidated_layers": sorted({event["layer"] for event in invalidated_layers}),
        "invalidated_ledger_entries": len(invalidated_ledgers),
        "cache_layer_entries": len(ledger["cache"]["layers"]),
        "cache_ledger_entries": len(ledger["cache"]["ledger_entries"]),
        "ledger_path": str((destination / "ledger.json").resolve()),
        "events": reuse_events + invalidated_layers + invalidated_ledgers,
    }
    atomic_write_json(destination / "reuse-report.json", report)
    return report

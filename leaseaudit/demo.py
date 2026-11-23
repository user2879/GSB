from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .clock import ManualClock, RealMonotonicClock
from .driver import MockSourceClient
from .engine import LeaseEngine
from .errors import BusinessError, ErrorCode
from .ledger import AuditLedger, EventRecord
from .logging_utils import StructuredLogger
from .numeric import numpy_watermark_ratios
from .replay import replay_ledger
from .scenario import Scenario, validate_scenario
from .util import atomic_copy, read_json_strict, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "scenarios" / "demo_scene.json"
DEFAULT_SOURCE_SCRIPT = ROOT / "scripts" / "mock_source.py"


def _find_free_loopback_port(start: int, used: set[int]) -> int:
    for port in range(max(1024, start), 65536):
        if port in used:
            continue
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        finally:
            sock.close()
        used.add(port)
        return port
    raise BusinessError(ErrorCode.E_ENV, "no free loopback high ports available", "port")


def _prepare_scenario(template_path: Path, workdir: Path, base_port: int) -> Scenario:
    raw = read_json_strict(template_path)
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        raise BusinessError(ErrorCode.E_SCENE, "demo template has no source array", "sources")
    used: set[int] = set()
    next_port = base_port
    for source in raw["sources"]:
        if not isinstance(source, dict):
            raise BusinessError(ErrorCode.E_SCENE, "demo source must be an object", "sources")
        port = _find_free_loopback_port(next_port, used)
        source["port"] = port
        next_port = port + 1
    scenario = validate_scenario(raw)
    write_json_atomic(workdir / "scenario.json", scenario.to_json_dict())
    return scenario


@dataclass
class _SourceProcess:
    source_id: str
    mode: str
    process: subprocess.Popen[str]


class _ServerProcesses:
    def __init__(self, scenario: Scenario, source_script: Path) -> None:
        self.scenario = scenario
        self.source_script = source_script
        self.processes: list[_SourceProcess] = []

    def __enter__(self) -> "_ServerProcesses":
        for source in self.scenario.ordered_sources():
            command = [
                sys.executable,
                str(self.source_script),
                "--source-id",
                source.source_id,
                "--port",
                str(source.port),
                "--mode",
                source.mode,
                "--processing-ms",
                str(source.processing_ms),
            ]
            kwargs: dict[str, Any] = {
                "cwd": str(ROOT),
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "env": {**os.environ, "PYTHONPATH": str(ROOT)},
            }
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            process = subprocess.Popen(command, **kwargs)
            self.processes.append(_SourceProcess(source.source_id, source.mode, process))
        self._wait_ready()
        return self

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 5.0
        pending = {(item.source_id, item.mode): item for item in self.processes}
        while pending and time.monotonic() < deadline:
            for key, item in list(pending.items()):
                if item.process.poll() is not None:
                    stderr = item.process.stderr.read() if item.process.stderr else ""
                    raise BusinessError(
                        ErrorCode.E_ENV,
                        "mock source process exited before readiness",
                        field_path=f"sources[{item.source_id}].port",
                        details={"source_id": item.source_id, "stderr": stderr[-2000:]},
                    )
                source = self.scenario.source_map()[item.source_id]
                if item.mode == "refuse":
                    # The refusal source intentionally has no listener.
                    time.sleep(0.05)
                    pending.pop(key)
                    continue
                try:
                    probe = socket.create_connection(("127.0.0.1", source.port), timeout=0.1)
                    probe.close()
                    pending.pop(key)
                except OSError:
                    pass
            if pending:
                time.sleep(0.03)
        if pending:
            raise BusinessError(
                ErrorCode.E_ENV,
                "mock source readiness timeout",
                field_path="sources[*].port",
                details={"pending": sorted(key[0] for key in pending)},
            )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for item in self.processes:
            if item.process.poll() is None:
                item.process.terminate()
        for item in self.processes:
            try:
                item.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                item.process.kill()
                item.process.wait(timeout=2.0)
            if item.process.stdout is not None:
                item.process.stdout.close()
            if item.process.stderr is not None:
                item.process.stderr.close()


def _issue_if_headroom(engine: LeaseEngine, source_id: str) -> Any:
    source = engine.sources[source_id]
    amount = source.spec.lease_share - source.active_granted
    if amount <= 0:
        return None
    return engine.issue(source_id, amount)


def _phase_event_summary(events: list[EventRecord], start_seq: int, end_seq: int) -> list[dict]:
    selected = [event for event in events if start_seq < event.seq <= end_seq]
    return [
        {
            "seq": event.seq,
            "type": event.event_type,
            "source_id": event.source_id,
            "lease_id": event.lease_id,
            "amount": event.amount,
            "decision": event.payload.get("decision"),
            "basis": event.payload.get("basis"),
        }
        for event in sorted(selected, key=lambda item: item.seq)
    ]


def run_demo(
    workdir: str | Path,
    *,
    template_path: str | Path = DEFAULT_TEMPLATE,
    source_script: str | Path = DEFAULT_SOURCE_SCRIPT,
    base_port: int = 55441,
    fresh: bool = True,
) -> dict:
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    template_path = Path(template_path).resolve()
    source_script = Path(source_script).resolve()
    if not source_script.is_file():
        raise BusinessError(ErrorCode.E_ENV, "mock source script is missing", "$.source_script", path=str(source_script))
    scenario = _prepare_scenario(template_path, workdir, base_port)
    ledger_path = workdir / "audit.sqlite"
    if fresh:
        for path in (ledger_path, Path(str(ledger_path) + "-wal"), Path(str(ledger_path) + "-shm")):
            if path.exists():
                path.unlink()
    logger = StructuredLogger(workdir / "structured.jsonl", echo=False)

    with _ServerProcesses(scenario, source_script):
        ledger = AuditLedger(ledger_path, scenario=scenario, verify=True)
        clock = RealMonotonicClock()
        engine = LeaseEngine(scenario, ledger, clock=clock, logger=logger)
        clients = {source.source_id: MockSourceClient(source) for source in scenario.ordered_sources()}
        phase = {"cold_start": False, "expiry_reclaim": False, "quota_reclaim": False}
        phase_marks: dict[str, int] = {}
        initial_issues: list[dict] = []
        expired_events: list[dict] = []
        reissued_events: list[dict] = []
        quota_events: list[dict] = []
        try:
            for source in scenario.ordered_sources():
                if source.effective_initial_issue > 0:
                    lease = engine.issue(source.source_id, source.effective_initial_issue, now_ms=clock.now_ms())
                    initial_issues.append(lease.as_dict())
                    clients[source.source_id].next_request_at_ms = clock.now_ms()
                    clients[source.source_id].next_renew_at_ms = clock.now_ms() + scenario.renew_interval_ms
            phase["cold_start"] = len(initial_issues) == sum(
                1 for source in scenario.sources if source.effective_initial_issue > 0
            )
            phase_marks["cold_start"] = ledger.meta()["last_seq"]

            start_ms = clock.now_ms()
            deadline = start_ms + scenario.load_duration_ms
            while clock.now_ms() <= deadline:
                now = clock.now_ms()
                expired = engine.process_expirations(now_ms=now)
                for event in expired:
                    expired_events.append(
                        {
                            "seq": event.seq,
                            "source_id": event.source_id,
                            "lease_id": event.lease_id,
                            "amount": event.amount,
                        }
                    )
                if not phase["expiry_reclaim"] and any(
                    event.source_id == "beta" for event in expired
                ):
                    before = ledger.meta()["last_seq"] + 1
                    if "alpha" in engine.sources:
                        issued = _issue_if_headroom(engine, "alpha")
                        if issued is not None:
                            reissued_events.append(
                                {"source_id": "alpha", "amount": issued.amount, "lease_id": issued.lease_id}
                            )
                    if "delta" in engine.sources:
                        issued = _issue_if_headroom(engine, "delta")
                        if issued is not None:
                            reissued_events.append(
                                {"source_id": "delta", "amount": issued.amount, "lease_id": issued.lease_id}
                            )
                    phase["expiry_reclaim"] = bool(reissued_events)
                    phase_marks["expiry_reclaim"] = ledger.meta()["last_seq"]
                    phase_marks["expiry_reclaim_start"] = before - 1

                for source in scenario.ordered_sources():
                    stopped = (
                        source.stop_renew_at_ms is not None
                        and now - start_ms >= source.stop_renew_at_ms
                    )
                    if not stopped:
                        runtime = engine.sources[source.source_id]
                        for lease in list(runtime._ordered_leases()):
                            last_renew = lease.last_renewed_at_ms or lease.issued_at_ms
                            if now >= last_renew + scenario.renew_interval_ms:
                                try:
                                    engine.renew(lease.lease_id, source.source_id, now_ms=now)
                                except BusinessError as exc:
                                    if exc.code != ErrorCode.E_RACE:
                                        raise
                    runtime = engine.sources[source.source_id]
                    if runtime.active_leases and now >= clients[source.source_id].next_request_at_ms:
                        lease = runtime._ordered_leases()[0]
                        if clients[source.source_id].lease_is_valid_at(lease, now):
                            clients[source.source_id].request(lease, now)
                            clients[source.source_id].next_request_at_ms = (
                                now + clients[source.source_id].next_request_delay_ms()
                            )

                observation_events = engine.maybe_observe(now_ms=now)
                for event in observation_events:
                    if event.event_type == "QUOTA_RECLAIMED":
                        quota_events.append(
                            {
                                "seq": event.seq,
                                "source_id": event.source_id,
                                "amount": event.amount,
                                "basis": event.payload.get("basis"),
                                "global_allocatable_after": event.payload.get("global_allocatable_after"),
                            }
                        )
                if not phase["quota_reclaim"] and quota_events:
                    if "gamma" in engine.sources:
                        issued = _issue_if_headroom(engine, "gamma")
                        if issued is not None:
                            reissued_events.append(
                                {"source_id": "gamma", "amount": issued.amount, "lease_id": issued.lease_id}
                            )
                    phase["quota_reclaim"] = True
                    phase_marks["quota_reclaim"] = ledger.meta()["last_seq"]
                time.sleep(0.01)

            engine.process_expirations(now_ms=clock.now_ms())
            final_state = engine.snapshot()
            events = ledger.events()
            integrity = ledger.verify_integrity()
            last_event_mono = events[-1].mono_ms if events else clock.now_ms()
            snapshot_manifest = ledger.create_snapshot(workdir / "snapshots")
        finally:
            engine.close()

    audit = replay_ledger(ledger_path, scenario)
    observed_events = [event for event in events if event.event_type == "OBSERVED"]
    ratios = numpy_watermark_ratios(
        [event.payload["observation"]["active_granted"] for event in observed_events],
        [event.payload["observation"]["lease_share"] for event in observed_events],
        scenario.watermark_scale,
    )
    ratio_mismatches = [
        {
            "seq": event.seq,
            "source_id": event.source_id,
            "engine_ppm": event.payload["observation"]["watermark_ratio_ppm"],
            "numpy_ppm": ratio,
        }
        for event, ratio in zip(observed_events, ratios)
        if event.payload["observation"]["watermark_ratio_ppm"] != ratio
    ]
    ratio_check = {
        "all_match": not ratio_mismatches,
        "sample_count": len(observed_events),
        "truncation": "FLOOR_TOWARD_ZERO",
        "mismatches": ratio_mismatches,
    }

    restore_check = _demonstrate_snapshot_restore(
        workdir=workdir,
        scenario=scenario,
        snapshot_manifest=snapshot_manifest,
        last_event_mono_ms=last_event_mono,
        logger=logger,
    )

    phase_codes = {
        "cold_start": "OK" if phase["cold_start"] else "E_DEMO_SEQUENCE",
        "expiry_reclaim": "OK" if phase["expiry_reclaim"] else "E_DEMO_SEQUENCE",
        "quota_reclaim": "OK" if phase["quota_reclaim"] else "E_DEMO_SEQUENCE",
        "numeric_watermark": "OK" if ratio_check["all_match"] else "E_QUOTA",
        "audit": audit["business_code"],
        "snapshot_replace_restore": restore_check["business_code"],
    }
    network = [clients[key].summary() for key in sorted(clients)]
    summary = {
        "scenario_version": scenario.scenario_version,
        "business_codes": phase_codes,
        "overall_code": (
            "OK"
            if all(code == "OK" for code in phase_codes.values()) and audit["status"] == "OK"
            else "E_DEMO"
        ),
        "phases": {
            "cold_start": {
                "code": phase_codes["cold_start"],
                "initial_issues": sorted(initial_issues, key=lambda item: (item["source_id"], item["lease_id"])),
                "events": _phase_event_summary(
                    events, 0, phase_marks.get("cold_start", 0)
                ),
            },
            "expiry_reclaim": {
                "code": phase_codes["expiry_reclaim"],
                "expired": sorted(expired_events, key=lambda item: (item["source_id"] or "", item["seq"])),
                "reissued": sorted(reissued_events[:2], key=lambda item: item["source_id"]),
                "events": _phase_event_summary(
                    events,
                    phase_marks.get("expiry_reclaim_start", phase_marks.get("cold_start", 0)),
                    phase_marks.get("expiry_reclaim", phase_marks.get("cold_start", 0)),
                ),
            },
            "quota_reclaim": {
                "code": phase_codes["quota_reclaim"],
                "reclaims": sorted(quota_events, key=lambda item: (item["source_id"] or "", item["seq"])),
                "reissued_after_reclaim": [
                    item for item in reissued_events if item["source_id"] == "gamma"
                ],
                "events": _phase_event_summary(
                    events,
                    phase_marks.get("expiry_reclaim", 0),
                    phase_marks.get("quota_reclaim", 0),
                ),
            },
        },
        "final_state": final_state,
        "global_allocatable": final_state["global_allocatable"],
        "retained_plus_global_equals_capacity": (
            final_state["retained_total"] + final_state["global_allocatable"]
            == final_state["cluster_capacity"]
        ),
        "numpy_watermark_verification": ratio_check,
        "audit": audit,
        "snapshot": snapshot_manifest,
        "snapshot_replace_restore": restore_check,
        "network": network,
        "ledger": {
            "path": str(ledger_path),
            "format_version": integrity["format_version"],
            "content_digest": integrity["content_digest"],
            "event_count": integrity["event_count"],
            "params_version": integrity["params_version"],
        },
        "exit_report_available": final_state["exit_report_available"],
    }
    write_json_atomic(workdir / "summary.json", summary)
    _write_human_report(workdir / "report.txt", summary)
    return summary


def _demonstrate_snapshot_restore(
    *,
    workdir: Path,
    scenario: Scenario,
    snapshot_manifest: dict,
    last_event_mono_ms: int,
    logger: StructuredLogger,
) -> dict:
    snapshot_db = workdir / "snapshots" / snapshot_manifest["snapshot_id"] / snapshot_manifest["ledger_file"]
    tampered = workdir / "restore-target.sqlite"
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(tampered) + suffix)
        if path.exists():
            path.unlink()
    atomic_copy(snapshot_db, tampered)
    ledger = AuditLedger(tampered, scenario=scenario, verify=True)
    clock = ManualClock(value_ms=last_event_mono_ms + 1)
    engine = LeaseEngine(scenario, ledger, clock=clock, logger=logger)
    try:
        engine.issue("beta", 1, now_ms=clock.now_ms())
        changed_digest = ledger.verify_integrity()["content_digest"]
    finally:
        engine.close()
    restored = AuditLedger.restore_snapshot(
        workdir / "snapshots" / snapshot_manifest["snapshot_id"], tampered
    )
    restored_audit = replay_ledger(tampered, scenario)
    code = (
        "OK"
        if changed_digest != snapshot_manifest["content_digest"]
        and restored_audit["content_digest"] == snapshot_manifest["content_digest"]
        else "E_SNAPSHOT"
    )
    return {
        "business_code": code,
        "tampered_digest": changed_digest,
        "snapshot_digest": snapshot_manifest["content_digest"],
        "restored_digest": restored_audit["content_digest"],
        "restore_status": restored["status"],
        "audit_status": restored_audit["status"],
    }


def _write_human_report(path: Path, summary: dict) -> None:
    lines = []
    for phase in ("cold_start", "expiry_reclaim", "quota_reclaim"):
        detail = summary["phases"][phase]
        if phase == "cold_start":
            facts = "issued=" + ",".join(
                f"{item['source_id']}:{item['amount']}"
                for item in detail["initial_issues"]
            )
        elif phase == "expiry_reclaim":
            facts = "expired=" + ",".join(
                f"{item['source_id']}:{item['amount']}" for item in detail["expired"]
            )
            facts += ";reissued=" + ",".join(
                f"{item['source_id']}:{item['amount']}" for item in detail["reissued"]
            )
        else:
            facts = "reclaimed=" + ",".join(
                f"{item['source_id']}:{item['amount']}" for item in detail["reclaims"]
            )
        lines.append(f"SCENARIO {phase} code={detail['code']} {facts}")
    lines.append(
        f"INVARIANT retained+global=capacity code="
        f"{'OK' if summary['retained_plus_global_equals_capacity'] else 'E_QUOTA'} "
        f"retained={summary['final_state']['retained_total']} "
        f"global={summary['global_allocatable']} capacity={summary['final_state']['cluster_capacity']}"
    )
    lines.append(
        f"WATERMARK code={summary['business_codes']['numeric_watermark']} "
        f"samples={summary['numpy_watermark_verification']['sample_count']} "
        f"truncation={summary['numpy_watermark_verification']['truncation']}"
    )
    lines.append(
        f"AUDIT code={summary['audit']['business_code']} events={summary['audit']['event_count']} "
        f"digest={summary['audit']['content_digest']}"
    )
    lines.append(
        f"SNAPSHOT_RESTORE code={summary['snapshot_replace_restore']['business_code']} "
        f"status={summary['snapshot_replace_restore']['restore_status']} "
        f"digest={summary['snapshot_replace_restore']['restored_digest']}"
    )
    for item in summary["network"]:
        outcomes = ",".join(f"{key}:{value}" for key, value in item["outcomes"].items()) or "none"
        lines.append(
            f"NETWORK source={item['source_id']} requests={item['requests']} outcomes={outcomes}"
        )
    lines.append(f"OVERALL code={summary['overall_code']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from .engine import LedgerEngine
from .ports import check_worker_ports
from .report import build_report
from .snapshot import SnapshotStore
from .utils import write_json
from .workspace import load_workspace, mark_injection_observations


def run_pipeline(
    data_dir: str | Path,
    run_dir: str | Path,
    *,
    injection_config: str | Path | None = None,
    injection_ids: list[str] | None = None,
    jobs: int = 4,
    restart_each_command: bool = False,
    reset: bool = False,
    check_ports: bool = False,
    max_commands: int | None = None,
    signal_gate: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    data_path = Path(data_dir)
    run_path = Path(run_dir)
    if reset and run_path.exists():
        shutil.rmtree(run_path)
    run_path.mkdir(parents=True, exist_ok=True)

    inputs, receipt_texts, config_hash, injection_results = load_workspace(
        data_path,
        injection_config_path=injection_config,
        injection_ids=injection_ids,
    )
    port_checks: list[dict[str, Any]] = []
    if check_ports:
        port_checks, _port_failures = check_worker_ports(inputs["workers"])

    store = SnapshotStore(run_path / "snapshots")
    store.clean_staging_directories()
    state, manifest, recovery = store.load_compatible_state(config_hash)
    if state is not None:
        engine = LedgerEngine.from_state(state, jobs=jobs)
    else:
        engine = LedgerEngine(
            inputs,
            receipt_texts,
            config_hash,
            jobs=jobs,
            injection_results=injection_results,
        )

    commands = inputs["commands"]
    global_version = int((manifest or {}).get("global_version", 0))
    last_manifest = manifest

    def commit_current() -> dict[str, Any]:
        nonlocal global_version, last_manifest
        mark_injection_observations(engine.injection_results, engine.current_failures())
        ledger = engine.export_ledger()
        global_version += 1
        last_manifest = store.commit(
            state=engine.to_state(),
            ledger=ledger,
            config_hash=config_hash,
            command_cursor=engine.command_cursor,
            global_version=global_version,
        )
        return last_manifest

    if restart_each_command:
        processed_this_call = 0
        while engine.command_cursor < len(commands):
            if signal_gate is not None:
                signal_gate.raise_if_stopped()
            if max_commands is not None and processed_this_call >= max_commands:
                break
            command = commands[engine.command_cursor]
            engine.apply_command(command)
            commit_current()
            loaded_state, loaded_manifest, _issue = store.load_compatible_state(config_hash)
            if loaded_state is None:
                raise RuntimeError("committed snapshot could not be recovered")
            engine = LedgerEngine.from_state(loaded_state, jobs=jobs)
            last_manifest = loaded_manifest
            global_version = int(loaded_manifest["global_version"])
            processed_this_call += 1
    else:
        pending_commands = commands[engine.command_cursor:]
        if max_commands is not None:
            pending_commands = pending_commands[:max_commands]
        for command in pending_commands:
            if signal_gate is not None:
                signal_gate.raise_if_stopped()
            engine.apply_command(command)
            commit_current()

    if last_manifest is None:
        last_manifest = commit_current()

    mark_injection_observations(engine.injection_results, engine.current_failures())
    ledger = engine.export_ledger()
    ledger_path = run_path / "ledger.json"
    write_json(ledger_path, ledger)
    write_json(run_path / "sidecars.json", {
        "sidecar_version": "receipt-sidecar.v1",
        "sidecars": ledger["sidecars"],
        "truncated_fields": ledger["r3_to_r1_truncated_fields"],
    })
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    report = build_report(
        ledger,
        elapsed_wall_ms=elapsed_ms,
        port_checks=port_checks,
        snapshot_manifest=last_manifest,
        recovery=recovery,
    )
    report_path = run_path / "report.json"
    write_json(report_path, report)
    return {
        "ledger": ledger,
        "report": report,
        "ledger_path": ledger_path,
        "report_path": report_path,
        "snapshot_manifest": last_manifest,
        "recovery": recovery,
    }

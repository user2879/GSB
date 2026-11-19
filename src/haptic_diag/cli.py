"""Command-line entry points for deterministic diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json_text, write_canonical_json
from .checker import reconcile_receipts
from .codec import decode_frame, encode_frame, migrate_to_v2, transcode_frame
from .compiler import compile_graph
from .engine import EngineState, apply_update, initial_state
from .generator import load_bundle, load_config, generate_bundle, generate_sample, write_bundle
from .model import EffectGraph
from .persistence import AtomicWorkspaceStore
from .transport import (
    DispatchConfig,
    DispatchTermination,
    LoopbackClient,
    LoopbackServer,
    dispatch_concurrently,
)


def _project_root() -> Path:
    for candidate in (Path.cwd(), Path(__file__).resolve().parents[2]):
        if (candidate / "configs" / "samples.json").exists():
            return candidate
    return Path.cwd()


def _default_config() -> Path:
    return _project_root() / "configs" / "samples.json"


def _read_graph(sample: str, config_path: Path, graphs_path: Path | None) -> EffectGraph:
    if graphs_path is not None and graphs_path.exists():
        for graph in load_bundle(graphs_path):
            if graph.id == sample:
                return graph
        raise SystemExit(f"sample {sample!r} not found in {graphs_path}")
    config = load_config(config_path)
    for sample_config in config["samples"]:
        if sample_config["name"] == sample:
            return generate_sample(sample_config)
    raise SystemExit(f"sample {sample!r} not found in {config_path}")


def _start_server_process(workspace: Path) -> tuple[subprocess.Popen[bytes], str, int, Path]:
    run_dir = workspace / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    port_file = run_dir / f"device-{os.getpid()}-{time.time_ns()}.port"
    env = os.environ.copy()
    src = str(_project_root() / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "haptic_diag.device_server",
            "--host",
            "127.0.0.1",
            "--port-file",
            str(port_file),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if port_file.exists():
            text = port_file.read_text(encoding="ascii").strip()
            host, port_text = text.rsplit(":", 1)
            if host != "127.0.0.1":
                raise RuntimeError(f"mock device bound unexpected host {host!r}")
            return process, host, int(port_text), port_file
        if process.poll() is not None:
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            raise RuntimeError(f"mock device process exited early: {stderr}")
        time.sleep(0.02)
    process.terminate()
    raise RuntimeError("mock device process did not publish its port file")


def _stop_server_process(process: subprocess.Popen[bytes], port_file: Path) -> None:
    try:
        process.terminate()
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)
    try:
        port_file.unlink()
    except FileNotFoundError:
        pass


def _dispatch_summary(state: EngineState, host: str, port: int, config: DispatchConfig) -> dict[str, Any]:
    client = LoopbackClient(host, port, timeout_s=config.socket_timeout_s)
    result = dispatch_concurrently(state.frames, client, config)
    check = reconcile_receipts(
        state.frames,
        result.receipts,
        expected_batch_digests=state.digests,
        observed_batch_digests=result.observed_batch_digests,
        observed_batch_ranges=result.observed_batch_ranges,
        profile=state.profile,
    )
    return {
        "dispatch": result.to_dict(),
        "check": check.to_dict(),
    }


def _protocol_roundtrip(state: EngineState) -> dict[str, Any]:
    frame_count_checked = 0
    v1_migrations_idempotent = True
    timestamp_mismatches = 0
    amplitude_mismatches = 0
    for frame in state.frames:
        wire = frame.wire_frame()
        decoded = decode_frame(encode_frame(wire))
        if decoded != wire:
            raise AssertionError(f"binary roundtrip changed frame {frame.seq}")
        if wire.protocol_version == 1:
            first = migrate_to_v2(wire)
            second = migrate_to_v2(first)
            if encode_frame(first) != encode_frame(second):
                v1_migrations_idempotent = False
            upgraded = transcode_frame(wire, 2)
            if upgraded.timestamp_us != wire.timestamp_us:
                timestamp_mismatches += 1
            expected_amp = {item.channel: item.amplitude_code for item in wire.channels}
            actual_amp = {item.channel: item.amplitude_code for item in upgraded.channels}
            if expected_amp != actual_amp:
                amplitude_mismatches += 1
        else:
            downgraded = transcode_frame(wire, 1)
            upgraded = migrate_to_v2(downgraded)
            if upgraded.timestamp_us != wire.timestamp_us:
                timestamp_mismatches += 1
            if {item.channel: item.amplitude_code for item in upgraded.channels} != {
                item.channel: item.amplitude_code for item in wire.channels
            }:
                amplitude_mismatches += 1
        frame_count_checked += 1
    return {
        "frame_count_checked": frame_count_checked,
        "v1_migration_idempotent": v1_migrations_idempotent,
        "timestamp_mismatches": timestamp_mismatches,
        "amplitude_mismatches": amplitude_mismatches,
        "note": "v1 has no frequency field; migration fills the fixed table",
    }


def cmd_generate(args: argparse.Namespace) -> int:
    config = Path(args.config)
    graphs = Path(args.graphs)
    truth = Path(args.truth)
    graphs.parent.mkdir(parents=True, exist_ok=True)
    truth.parent.mkdir(parents=True, exist_ok=True)
    graph_bytes, truth_bytes = write_bundle(config, graphs, truth)
    print(
        canonical_json_text(
            {
                "graphs": str(graphs),
                "graphs_sha256": __import__("hashlib").sha256(graph_bytes).hexdigest(),
                "truth": str(truth),
                "truth_sha256": __import__("hashlib").sha256(truth_bytes).hexdigest(),
            }
        ),
        end="",
    )
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    graph = _read_graph(args.sample, Path(args.config), Path(args.graphs) if args.graphs else None)
    result = compile_graph(graph, args.protocol)
    report_path = Path(args.report) if args.report else None
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_canonical_json(report_path, result.report.to_dict(include_frames=True))
    summary = {
        "sample": graph.id,
        "protocol_version": args.protocol,
        "frame_count": len(result.frames),
        "batch_count": len(result.digests),
        "total_encoded_bytes": sum(len(frame.encode()) for frame in result.frames),
        "first_frame_hash": result.frames[0].content_hash,
        "last_frame_hash": result.frames[-1].content_hash,
        "quantization": {
            "amplitude_step": result.report.profile["amplitude_step"],
            "frequency_step": result.report.profile["frequency_step"],
            "amplitude_error_bound": result.report.amplitude_error_bound,
            "frequency_error_bound": result.report.frequency_error_bound,
            "max_amplitude_error": result.report.max_amplitude_error,
            "max_frequency_error": result.report.max_frequency_error,
            "bound_satisfied": result.report.error_bound_satisfied,
        },
        "report": str(report_path) if report_path else None,
    }
    print(canonical_json_text(summary), end="")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    store = AtomicWorkspaceStore(workspace)
    if not args.keep_state:
        store.clear()
    graph = _read_graph(args.sample, Path(args.config), Path(args.graphs) if args.graphs else None)
    state = initial_state(graph, args.protocol)
    initial_commit = store.commit(state)
    process, host, port, port_file = _start_server_process(workspace)
    try:
        initial_dispatch = _dispatch_summary(
            state,
            host,
            port,
            DispatchConfig(
                max_workers=args.workers,
                batch_size=args.batch_size,
                batch_memory_limit_bytes=args.memory_limit,
                batch_time_budget_ms=args.time_budget_ms,
            ),
        )
        if initial_dispatch["check"]["status"] != "confirmed":
            raise AssertionError(f"initial receipts were not confirmed: {initial_dispatch['check']}")

        first_primitive = graph.primitives[0]
        tiny = 0.000001
        no_op_state, no_op_update = apply_update(
            state,
            {first_primitive.id: {"frequency_hz": first_primitive.frequency_hz + tiny}},
        )
        if args.protocol == 2:
            effective_patch = {
                "frequency_hz": (
                    first_primitive.frequency_hz + 10.0
                    if first_primitive.frequency_hz <= 1990.0
                    else first_primitive.frequency_hz - 10.0
                )
            }
        else:
            effective_patch = {
                "amplitude": (
                    first_primitive.amplitude + 0.07
                    if first_primitive.amplitude <= 0.9
                    else first_primitive.amplitude - 0.07
                )
            }
        incremental_state, incremental_update = apply_update(
            no_op_state,
            {first_primitive.id: effective_patch},
        )
        incremental_dispatch = _dispatch_summary(
            incremental_state,
            host,
            port,
            DispatchConfig(
                max_workers=args.workers,
                batch_size=args.batch_size,
                batch_memory_limit_bytes=args.memory_limit,
                batch_time_budget_ms=args.time_budget_ms,
            ),
        )
        if incremental_dispatch["check"]["status"] != "confirmed":
            raise AssertionError(
                f"incremental receipts were not confirmed: {incremental_dispatch['check']}"
            )
        target_protocol = 1 if args.protocol == 2 else 2
        migrated_state, migration_update = apply_update(
            incremental_state,
            {},
            new_protocol_version=target_protocol,
        )
        migrated_dispatch = _dispatch_summary(
            migrated_state,
            host,
            port,
            DispatchConfig(
                max_workers=args.workers,
                batch_size=args.batch_size,
                batch_memory_limit_bytes=args.memory_limit,
                batch_time_budget_ms=args.time_budget_ms,
            ),
        )
        if migrated_dispatch["check"]["status"] != "confirmed":
            raise AssertionError(
                f"migrated receipts were not confirmed: {migrated_dispatch['check']}"
            )
    finally:
        _stop_server_process(process, port_file)

    roundtrip = _protocol_roundtrip(migrated_state)
    if (
        roundtrip["timestamp_mismatches"]
        or roundtrip["amplitude_mismatches"]
        or not roundtrip["v1_migration_idempotent"]
    ):
        raise AssertionError(f"protocol roundtrip mismatch: {roundtrip}")
    final_commit = store.commit(migrated_state)
    recovery = store.recover()
    if recovery.state is None:
        raise AssertionError(f"recovery failed: {recovery.reason}")
    recovered_hashes = [frame.content_hash for frame in recovery.state.frames]
    current_hashes = [frame.content_hash for frame in migrated_state.frames]
    if recovered_hashes != current_hashes:
        raise AssertionError("recovered frame hashes differ from the committed state")
    recovered_digests = [digest.digest for digest in recovery.state.digests]
    current_digests = [digest.digest for digest in migrated_state.digests]
    if recovered_digests != current_digests:
        raise AssertionError("recovered digest set differs from the committed state")

    summary = {
        "status": "ok",
        "sample": graph.id,
        "workspace": str(workspace),
        "initial_protocol": args.protocol,
        "final_protocol": target_protocol,
        "initial_commit": initial_commit.to_dict(),
        "final_commit": final_commit.to_dict(),
        "initial_dispatch_check": initial_dispatch,
        "no_op_update": no_op_update.summary(),
        "incremental_update": incremental_update.summary(),
        "migration_update": migration_update.summary(),
        "incremental_dispatch_check": incremental_dispatch,
        "migrated_dispatch_check": migrated_dispatch,
        "protocol_roundtrip": roundtrip,
        "recovery": recovery.to_dict(),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output, summary)
    console_summary = {
        "status": summary["status"],
        "sample": graph.id,
        "initial_protocol": args.protocol,
        "final_protocol": target_protocol,
        "initial_commit": initial_commit.generation,
        "final_commit": final_commit.generation,
        "no_op_invalidated_frames": len(no_op_update.invalidated_frames),
        "incremental_recompiled_frames": len(incremental_update.recompiled_frames),
        "migration_recompiled_frames": len(migration_update.recompiled_frames),
        "initial_check_status": initial_dispatch["check"]["status"],
        "incremental_check_status": incremental_dispatch["check"]["status"],
        "migrated_check_status": migrated_dispatch["check"]["status"],
        "protocol_roundtrip": roundtrip,
        "recovered_generation": recovery.current_generation,
        "orphan_count": len(recovery.orphans),
        "report": str(args.output) if args.output else None,
    }
    print(canonical_json_text(console_summary), end="")
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    store = AtomicWorkspaceStore(Path(args.workspace))
    recovery = store.recover()
    summary = recovery.to_dict()
    if recovery.state is not None:
        summary["frame_count"] = len(recovery.state.frames)
        summary["digest_count"] = len(recovery.state.digests)
        summary["ledger_entries"] = len(recovery.state.ledger.entries)
        summary["archived_ledgers"] = len(recovery.state.ledger_history)
    print(canonical_json_text(summary), end="")
    return 0 if recovery.state is not None else 2


def cmd_crash_worker(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    store = AtomicWorkspaceStore(workspace)
    loaded = store.load_committed()
    graph = _read_graph(args.sample, Path(args.config), Path(args.graphs) if args.graphs else None)
    if loaded is None:
        baseline = initial_state(graph, args.protocol)
        store.commit(baseline)
    else:
        baseline = loaded[0]
    primitive = baseline.graph.primitives[0]
    amplitude = primitive.amplitude + 0.07
    if amplitude > 1.0:
        amplitude = primitive.amplitude - 0.07
    pending_state, _ = apply_update(baseline, {primitive.id: {"amplitude": amplitude}})
    with LoopbackServer() as server:
        try:
            result = dispatch_concurrently(
                pending_state.frames,
                LoopbackClient(server.host, server.port),
                DispatchConfig(
                    max_workers=args.workers,
                    batch_size=args.batch_size,
                    batch_memory_limit_bytes=10_000_000,
                    batch_time_budget_ms=1000.0,
                    abort_after_batches=args.crash_after_batches,
                ),
            )
            # Reaching this point means the abort hook was not triggered.
            if result.missing_sequences:
                raise SystemExit("unexpected missing receipts before crash injection")
        except DispatchTermination:
            pass
    # Simulate a hard termination while the next snapshot is being written.
    store.commit(
        pending_state,
        crash_after_files=args.commit_crash_after_files,
        hard_crash=True,
    )
    raise AssertionError("hard crash injection did not terminate the process")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="haptic-diag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic samples and truth")
    generate.add_argument("--config", default=str(_default_config()))
    generate.add_argument("--graphs", default=str(_project_root() / "samples" / "effect_graphs.json"))
    generate.add_argument("--truth", default=str(_project_root() / "samples" / "truth_manifest.json"))
    generate.set_defaults(func=cmd_generate)

    compile_parser = subparsers.add_parser("compile", help="compile one sample")
    compile_parser.add_argument("--sample", default="cross_slice_boundary")
    compile_parser.add_argument("--protocol", type=int, choices=(1, 2), default=2)
    compile_parser.add_argument("--config", default=str(_default_config()))
    compile_parser.add_argument("--graphs", default=str(_project_root() / "samples" / "effect_graphs.json"))
    compile_parser.add_argument("--report", default=None)
    compile_parser.set_defaults(func=cmd_compile)

    verify = subparsers.add_parser("verify", help="run the local end-to-end verification")
    verify.add_argument("--sample", default="cross_slice_boundary")
    verify.add_argument("--protocol", type=int, choices=(1, 2), default=2)
    verify.add_argument("--config", default=str(_default_config()))
    verify.add_argument("--graphs", default=str(_project_root() / "samples" / "effect_graphs.json"))
    verify.add_argument("--workspace", default=str(_project_root() / "workbench"))
    verify.add_argument("--workers", type=int, default=4)
    verify.add_argument("--batch-size", type=int, default=20)
    verify.add_argument("--memory-limit", type=int, default=262144)
    verify.add_argument("--time-budget-ms", type=float, default=250.0)
    verify.add_argument("--output", default=str(_project_root() / "reports" / "verification.json"))
    verify.add_argument("--keep-state", action="store_true")
    verify.set_defaults(func=cmd_verify)

    recover = subparsers.add_parser("recover", help="recover the CURRENT committed state")
    recover.add_argument("--workspace", default=str(_project_root() / "workbench"))
    recover.set_defaults(func=cmd_recover)

    crash = subparsers.add_parser("crash-worker", help="internal forced-termination worker")
    crash.add_argument("--sample", default="cross_slice_boundary")
    crash.add_argument("--protocol", type=int, choices=(1, 2), default=2)
    crash.add_argument("--config", default=str(_default_config()))
    crash.add_argument("--graphs", default=str(_project_root() / "samples" / "effect_graphs.json"))
    crash.add_argument("--workspace", default=str(_project_root() / "workbench-crash"))
    crash.add_argument("--workers", type=int, default=2)
    crash.add_argument("--batch-size", type=int, default=20)
    crash.add_argument("--crash-after-batches", type=int, default=2)
    crash.add_argument("--commit-crash-after-files", type=int, default=3)
    crash.set_defaults(func=cmd_crash_worker)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

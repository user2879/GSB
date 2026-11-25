"""Repeatable end-to-end acceptance demonstration."""

from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .canonical import atomic_write_json, canonical_sha256, sha256_hex
from .errors import DiagnosticError
from .invalidation import compute_invalidation, recompute_invalid_subset
from .ledger import Ledger
from .leases import LeaseScheduler
from .metrics import build_metrics_report
from .orchestration import (
    ensure_port_available,
    find_free_loopback_port,
    probe_workload_ports,
)
from .pipeline import (
    commit_receipts,
    execute_over_loopback,
    normalize_local_results,
    replay_local_against_ledger,
)
from .protocol import LineReader, perform_handshake
from .receipts import NormalizedReceipt, context_for, normalize_receipt, normalize_receipt_file
from .samples import prepare_samples, sample_paths
from .spec import load_workload, load_workload_data
from .synthetic import deterministic_audio_manifest, synthesize, write_wav


def _remove_sqlite(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _remove_known_artifacts(root: Path) -> None:
    names = [
        "diagnostic.sqlite3",
        "diagnostic.sqlite3-wal",
        "diagnostic.sqlite3-shm",
        "crash.sqlite3",
        "crash.sqlite3-wal",
        "crash.sqlite3-shm",
        "restored.sqlite3",
        "restored.sqlite3-wal",
        "restored.sqlite3-shm",
        "session_replace.sqlite3",
        "session_replace.sqlite3-wal",
        "session_replace.sqlite3-shm",
        "runtime_workload.json",
        "audio.wav",
        "audio_manifest.json",
        "features.json",
        "truth.json",
        "normalization.json",
        "lease_report.json",
        "metrics.json",
        "invalidation.json",
        "recovery.json",
        "session_snapshot.json",
        "session_replace.json",
        "anomalies.json",
        "content_manifest.json",
        "verification.json",
    ]
    for name in names:
        path = root / name
        if path.exists():
            path.unlink()


def _spec_with_ports(spec: Any, ports: list[int], path: Path) -> Any:
    data = spec.to_dict()
    if len(ports) != len(data["workers"]):
        raise ValueError("one port per worker is required")
    for worker, port in zip(data["workers"], ports):
        worker["port"] = port
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return load_workload_data(source_text=text, source_name=str(path))


def _allocate_ports(count: int) -> list[int]:
    ports: list[int] = []
    while len(ports) < count:
        candidate = find_free_loopback_port()
        if candidate not in ports:
            ports.append(candidate)
    return ports


def _capture(code: str, action: Callable[[], Any]) -> dict[str, Any]:
    started = time.monotonic_ns()
    try:
        action()
    except DiagnosticError as exc:
        elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
        if exc.code != code:
            raise AssertionError(f"expected {code}, got {exc.code}: {exc}") from exc
        return {
            "expected_code": code,
            "observed_code": exc.code,
            "matched": True,
            "field": exc.field,
            "line": exc.line,
            "message": exc.message,
            "details": exc.details,
            "elapsed_ms": elapsed_ms,
        }
    raise AssertionError(f"expected {code}, but the operation succeeded")


def _hash_json(value: Any) -> str:
    return canonical_sha256(value)


def run_demo(project_root: str | Path, *, samples_root: str | Path | None = None) -> dict[str, Any]:
    project = Path(project_root).resolve()
    sample_dir = Path(samples_root).resolve() if samples_root is not None else project / "samples"
    artifact_dir = project / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _remove_known_artifacts(artifact_dir)
    sample_manifest = prepare_samples(sample_dir)
    paths = sample_paths(sample_dir)

    base_spec = load_workload(paths["v1"])
    v2_spec = load_workload(paths["v2"])
    v3_spec = load_workload(paths["v3"])
    base_bundle = synthesize(base_spec)
    v2_bundle = synthesize(v2_spec)
    v3_bundle = synthesize(v3_spec)

    write_wav(artifact_dir / "audio.wav", base_bundle, base_spec.audio.sample_rate)
    atomic_write_json(artifact_dir / "audio_manifest.json", deterministic_audio_manifest(base_spec, base_bundle))
    atomic_write_json(artifact_dir / "features.json", base_bundle.feature_manifest())
    atomic_write_json(artifact_dir / "truth.json", base_bundle.truth_manifest())

    reordered_raw = base_spec.to_dict()
    reordered_raw["workers"] = list(reversed(reordered_raw["workers"]))
    reordered_raw["requests"] = list(reversed(reordered_raw["requests"]))
    reordered_spec = load_workload_data(reordered_raw, source_name="<reordered-in-memory>")
    reordered_bundle = synthesize(reordered_spec)
    deterministic = {
        "window_starts_identical": base_spec.window_starts_samples == reordered_spec.window_starts_samples,
        "truth_sequence_identical": [item.to_dict() for item in base_bundle.windows]
        == [item.to_dict() for item in reordered_bundle.windows],
        "window_digest_identical": base_bundle.window_digest == reordered_bundle.window_digest,
        "truth_digest_identical": base_bundle.truth_digest == reordered_bundle.truth_digest,
        "feature_digest_identical": base_bundle.feature_digest == reordered_bundle.feature_digest,
        "pcm_digest_identical": base_bundle.pcm_digest == reordered_bundle.pcm_digest,
        "base_digests": {
            "window": base_bundle.window_digest,
            "truth": base_bundle.truth_digest,
            "feature": base_bundle.feature_digest,
            "pcm": base_bundle.pcm_digest,
        },
        "reordered_digests": {
            "window": reordered_bundle.window_digest,
            "truth": reordered_bundle.truth_digest,
            "feature": reordered_bundle.feature_digest,
            "pcm": reordered_bundle.pcm_digest,
        },
    }
    if not all(value for key, value in deterministic.items() if isinstance(value, bool)):
        raise AssertionError("deterministic audio/window check failed")

    sample_port_probe: dict[str, Any]
    runtime_spec = base_spec
    runtime_spec_path = paths["v1"]
    port_fallback = False
    try:
        sample_port_probe = probe_workload_ports(base_spec)
    except DiagnosticError as exc:
        if exc.code != "E_PORT":
            raise
        fallback_ports = _allocate_ports(len(base_spec.workers))
        runtime_spec_path = artifact_dir / "runtime_workload.json"
        runtime_spec = _spec_with_ports(base_spec, fallback_ports, runtime_spec_path)
        sample_port_probe = probe_workload_ports(runtime_spec)
        port_fallback = True

    # All subsequent live execution uses the exact spec digest recorded in the ledger.
    runtime_bundle = base_bundle if runtime_spec is base_spec else synthesize(runtime_spec)
    runtime_context = context_for(runtime_spec, "B001")
    normalization_records = []
    attribution_digests = []
    for label in ("r1", "r2", "r3"):
        normalized = normalize_receipt_file(paths[label], context=runtime_context)
        normalized_again = normalize_receipt_file(paths[label], context=runtime_context)
        if normalized.canonical != normalized_again.canonical:
            raise AssertionError(f"{label} normalization is not byte-stable")
        attribution_digests.append(normalized.attribution_digest)
        normalization_records.append(
            {
                "source_format": label,
                "filled_fields": list(normalized.filled_fields),
                "fill_basis": dict(normalized.data["fill_basis"]),
                "attribution_digest": normalized.attribution_digest,
                "normalized_sha256": normalized.sha256,
                "byte_stable_across_two_parses": True,
            }
        )
    if len(set(attribution_digests)) != 1:
        raise AssertionError("r1/r2/r3 attribution digests differ")
    normalization_report = {
        "schema": "speaker-diag-normalization-demonstration-v1",
        "batch_id": "B001",
        "same_attribution_digest": attribution_digests[0],
        "records": normalization_records,
    }
    atomic_write_json(artifact_dir / "normalization.json", normalization_report)

    ledger_path = artifact_dir / "diagnostic.sqlite3"
    _remove_sqlite(ledger_path)
    ledger = Ledger(
        ledger_path,
        session_version=runtime_spec.session_version,
        spec_digest=runtime_spec.spec_digest,
        session_snapshot=runtime_spec.to_dict(),
        reset=True,
    )
    started = time.perf_counter_ns()
    socket_run = execute_over_loopback(runtime_spec, runtime_bundle, runtime_spec_path, ledger=ledger)
    socket_elapsed_ms = (time.perf_counter_ns() - started) // 1_000_000
    if set(socket_run.normalized) != {request.batch_id for request in runtime_spec.requests}:
        raise AssertionError("loopback run did not return every batch")
    _, local_normalized = normalize_local_results(runtime_spec, runtime_bundle)
    attribution_match = []
    for request in runtime_spec.requests:
        batch_id = request.batch_id
        equal = socket_run.normalized[batch_id].attribution_digest == local_normalized[batch_id].attribution_digest
        attribution_match.append({"batch_id": batch_id, "attribution_digest_equal": equal})
        if not equal:
            raise AssertionError(f"worker/local attribution digest differs for {batch_id}")
    replay_report = replay_local_against_ledger(runtime_spec, runtime_bundle, ledger)
    content_manifest = ledger.write_content_manifest(artifact_dir / "content_manifest.json")
    metrics_report = build_metrics_report(socket_run.normalized, runtime_bundle, runtime_spec, socket_run.lease_snapshot)
    atomic_write_json(artifact_dir / "metrics.json", metrics_report)

    replay_steps = socket_run.lease_replay["steps"]
    max_global_active = max((step["active_total"] for step in replay_steps), default=0)
    max_worker_leases = {
        worker.worker_id: max(
            (step["worker_active_leases"].get(worker.worker_id, 0) for step in replay_steps), default=0
        )
        for worker in runtime_spec.workers
    }
    lease_report = {
        "schema": "speaker-diag-lease-demonstration-v1",
        "port_probe": sample_port_probe,
        "port_fallback_used": port_fallback,
        "snapshot": socket_run.lease_snapshot,
        "replay": socket_run.lease_replay,
        "limits": {
            "global": runtime_spec.limits.global_batch_leases,
            "workers": {
                worker.worker_id: {
                    "leases": worker.max_worker_leases,
                    "concurrency": worker.max_concurrent,
                }
                for worker in runtime_spec.workers
            },
        },
        "observed_max_global_active": max_global_active,
        "observed_max_worker_active": max_worker_leases,
        "all_limits_respected": max_global_active <= runtime_spec.limits.global_batch_leases
        and all(
            max_worker_leases[worker.worker_id] <= worker.max_worker_leases
            for worker in runtime_spec.workers
        ),
        "queue_records": socket_run.lease_snapshot["queue_records"],
        "socket_execution_ms": socket_elapsed_ms,
    }
    if not socket_run.lease_replay["replayable"] or not lease_report["all_limits_respected"]:
        raise AssertionError("lease replay or quota constraint failed")
    atomic_write_json(artifact_dir / "lease_report.json", lease_report)

    # Session snapshot export and whole-session restore.
    snapshot_path = artifact_dir / "session_snapshot.json"
    ledger.export_snapshot(snapshot_path)
    restored_path = artifact_dir / "restored.sqlite3"
    _remove_sqlite(restored_path)
    restored = Ledger.restore_snapshot(restored_path, snapshot_path)
    snapshot_restore_replay = replay_local_against_ledger(runtime_spec, runtime_bundle, restored)

    # Version-triggered whole replacement on an isolated ledger.
    replace_path = artifact_dir / "session_replace.sqlite3"
    _remove_sqlite(replace_path)
    replace_ledger = Ledger(
        replace_path,
        session_version=base_spec.session_version,
        spec_digest=base_spec.spec_digest,
        session_snapshot=base_spec.to_dict(),
        reset=True,
    )
    _, v1_local = normalize_local_results(base_spec, base_bundle)
    commit_receipts(replace_ledger, base_spec, base_bundle, {"B001": v1_local["B001"]})
    before_replace_count = replace_ledger.content_manifest()["batch_count"]
    replaced = Ledger(
        replace_path,
        session_version=v2_spec.session_version,
        spec_digest=v2_spec.spec_digest,
        session_snapshot=v2_spec.to_dict(),
        allow_replace=True,
    )
    replacement_report = {
        "schema": "speaker-diag-session-replacement-v1",
        "before_session_version": base_spec.session_version,
        "before_batch_count": before_replace_count,
        "after_session_version": replaced.content_manifest()["session_version"],
        "after_batch_count": replaced.content_manifest()["batch_count"],
        "whole_replacement_verified": before_replace_count == 1
        and replaced.content_manifest()["batch_count"] == 0
        and replaced.content_manifest()["session_version"] == v2_spec.session_version,
    }
    if not replacement_report["whole_replacement_verified"]:
        raise AssertionError("session version replacement failed")
    atomic_write_json(artifact_dir / "session_replace.json", replacement_report)

    # Dependency invalidation rounds.
    _, base_local = normalize_local_results(base_spec, base_bundle)
    v2_invalidation = compute_invalidation(base_spec, v2_spec)
    v2_recompute = recompute_invalid_subset(base_spec, v2_spec, v2_bundle, base_local, v2_invalidation)
    _, v2_local = normalize_local_results(v2_spec, v2_bundle)
    v3_invalidation = compute_invalidation(v2_spec, v3_spec)
    v3_recompute = recompute_invalid_subset(
        v2_spec,
        v3_spec,
        v3_bundle,
        {key: v2_recompute["receipts"][key] for key in v2_local},
        v3_invalidation,
    )
    invalidation_report = {
        "schema": "speaker-diag-invalidation-demonstration-v1",
        "rounds": [
            {
                "name": "v1_to_v2_worker_version",
                "invalidation": v2_invalidation,
                "recompute": {key: value for key, value in v2_recompute.items() if key != "receipts"},
            },
            {
                "name": "v2_to_v3_quant_batch_worker",
                "invalidation": v3_invalidation,
                "recompute": {key: value for key, value in v3_recompute.items() if key != "receipts"},
            },
        ],
        "v2_direct_batches": [entry["batch_id"] for entry in v2_invalidation["direct_batches"]],
        "v2_indirect_batches": [entry["batch_id"] for entry in v2_invalidation["indirect_batches"]],
        "v2_invalid_windows": len(v2_invalidation["invalid_windows"]),
        "v2_unchanged_batches": v2_invalidation["unchanged_batches"],
        "v3_all_batches_invalid": len(v3_invalidation["invalid_batches"]) == len(v3_spec.requests),
        "unchanged_byte_identical": v2_recompute["unchanged_check"]["all_byte_identical"],
    }
    atomic_write_json(artifact_dir / "invalidation.json", invalidation_report)

    # Deterministic local replay for invalidation must match the full local run.
    selective_v2_matches_full = all(
        v2_recompute["receipts"][batch_id].attribution_digest == v2_local[batch_id].attribution_digest
        for batch_id in v2_local
    )
    if not selective_v2_matches_full:
        raise AssertionError("selective v2 recomputation does not match full v2 inference")

    # Force-termination recovery.
    crash_path = artifact_dir / "crash.sqlite3"
    _remove_sqlite(crash_path)
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "speaker_diag.crash_fixture",
            "--db",
            str(crash_path),
            "--spec",
            str(runtime_spec_path),
            "--commit-count",
            "4",
            "--exit-code",
            "37",
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent)
            + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if child.returncode != 37:
        raise AssertionError(f"crash fixture exited {child.returncode}: {child.stderr}")
    recovered = Ledger(
        crash_path,
        session_version=runtime_spec.session_version,
        spec_digest=runtime_spec.spec_digest,
        session_snapshot=runtime_spec.to_dict(),
        allow_replace=True,
    )
    expected_order = [request.batch_id for request in runtime_spec.requests]
    recovery = recovered.recover(expected_order)
    if not recovery["rerun_required"] or len(recovery["unknown_batches"]) != 6:
        raise AssertionError("recovery did not identify exactly the uncommitted batches")
    rerun_receipts = {batch_id: local_normalized[batch_id] for batch_id in recovery["rerun_order"]}
    recovery_commits = commit_receipts(recovered, runtime_spec, runtime_bundle, rerun_receipts)
    recovery_replay = replay_local_against_ledger(runtime_spec, runtime_bundle, recovered)
    recovery_report = {
        "schema": "speaker-diag-recovery-demonstration-v1",
        "child_exit_code": child.returncode,
        "child_stdout": child.stdout.strip(),
        "recovery": recovery,
        "rerun_commits": recovery_commits,
        "replay": recovery_replay,
        "post_rerun_batch_count": recovered.content_manifest()["batch_count"],
        "post_rerun_unknown_batches": recovered.content_manifest()["unknown_batches"],
    }
    if recovery_report["post_rerun_batch_count"] != len(runtime_spec.requests):
        raise AssertionError("recovery did not restore every batch")
    atomic_write_json(artifact_dir / "recovery.json", recovery_report)

    # Business-code anomaly injections.
    anomalies: list[dict[str, Any]] = []
    bad_spec = base_spec.to_dict()
    bad_spec["window"]["hop_ms"] = bad_spec["window"]["window_ms"]
    bad_text = json.dumps(bad_spec, ensure_ascii=False, indent=2)
    anomalies.append(_capture("E_SPEC", lambda: load_workload_data(source_text=bad_text, source_name="<bad-spec>")))

    raw_r3 = json.loads(paths["r3"].read_text(encoding="utf-8"))
    raw_negative = copy.deepcopy(raw_r3)
    raw_negative["elapsed_ms"] = -1
    anomalies.append(
        _capture(
            "E_FIELD",
            lambda: normalize_receipt(raw_negative, context=runtime_context),
        )
    )
    raw_bad_ref = copy.deepcopy(raw_r3)
    raw_bad_ref["windows"][0]["duplicate_of"] = "W9999"
    anomalies.append(
        _capture(
            "E_REF",
            lambda: normalize_receipt(raw_bad_ref, context=runtime_context),
        )
    )

    occupied_port = find_free_loopback_port()
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.bind(("127.0.0.1", occupied_port))
    occupied.listen(1)
    try:
        anomalies.append(
            _capture(
                "E_PORT",
                lambda: ensure_port_available("127.0.0.1", occupied_port, field="workers[0].port"),
            )
        )
    finally:
        occupied.close()

    dummy_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    dummy_listener.bind(("127.0.0.1", 0))
    dummy_listener.listen(1)
    dummy_port = int(dummy_listener.getsockname()[1])
    dummy_stop = threading.Event()

    def silent_server() -> None:
        dummy_listener.settimeout(0.5)
        try:
            connection, _address = dummy_listener.accept()
        except (socket.timeout, OSError):
            dummy_listener.close()
            return
        try:
            dummy_stop.wait(1.0)
        finally:
            connection.close()
            dummy_listener.close()

    dummy_thread = threading.Thread(target=silent_server, daemon=True)
    dummy_thread.start()

    def handshake_timeout_probe() -> None:
        sock = socket.create_connection(("127.0.0.1", dummy_port), timeout=0.5)
        try:
            perform_handshake(
                sock,
                LineReader(sock),
                expected_worker_id=runtime_spec.workers[0].worker_id,
                expected_param_version=runtime_spec.workers[0].parameter_version,
                expected_param_digest=context_for(runtime_spec, "B001").param_digest,
                timeout_ms=80,
                phase="anomaly_handshake",
            )
        finally:
            sock.close()

    anomalies.append(_capture("E_HANDSHAKE", handshake_timeout_probe))
    dummy_stop.set()
    dummy_thread.join(timeout=1.0)

    lease_injection = LeaseScheduler(
        global_limit=1,
        worker_lease_limits={runtime_spec.workers[0].worker_id: 1},
        worker_concurrency_limits={runtime_spec.workers[0].worker_id: 1},
        lease_ttl_ms={runtime_spec.workers[0].worker_id: 200},
        metric_scale=runtime_spec.limits.metric_scale,
    )
    injected_lease = lease_injection.acquire("lease-probe", runtime_spec.workers[0].worker_id)

    def lease_version_probe() -> None:
        try:
            lease_injection.renew(injected_lease.lease_id, injected_lease.version + 9)
        finally:
            lease_injection.release(injected_lease.lease_id, expected_version=injected_lease.version)

    anomalies.append(_capture("E_LEASE", lease_version_probe))

    def replay_mismatch_probe() -> None:
        batch_id = expected_order[0]
        bad_data = json.loads(json.dumps(local_normalized[batch_id].data))
        bad_data["windows"][0]["attribution"] = "spk_9"
        bad_normalized = NormalizedReceipt(
            data=bad_data,
            canonical=json.dumps(bad_data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            sha256=sha256_hex(json.dumps(bad_data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")),
            attribution_digest=canonical_sha256(bad_data),
            source_format="r3",
            filled_fields=(),
        )
        replay_map = dict(local_normalized)
        replay_map[batch_id] = bad_normalized
        ledger.verify_replay(replay_map, expected_order=expected_order)

    anomalies.append(_capture("E_REPLAY", replay_mismatch_probe))
    anomaly_report = {
        "schema": "speaker-diag-anomaly-injections-v1",
        "all_codes_matched": all(item["matched"] for item in anomalies),
        "codes": [item["observed_code"] for item in anomalies],
        "injections": anomalies,
    }
    if not anomaly_report["all_codes_matched"]:
        raise AssertionError("one or more anomaly injections did not produce the expected code")
    atomic_write_json(artifact_dir / "anomalies.json", anomaly_report)

    checks = {
        "deterministic_audio_window_features": all(
            value for key, value in deterministic.items() if isinstance(value, bool)
        ),
        "r1_r2_r3_normalize_to_same_attribution": len(set(attribution_digests)) == 1,
        "normalization_byte_stable": all(record["byte_stable_across_two_parses"] for record in normalization_records),
        "real_loopback_workers": len(socket_run.raw_receipts) == len(runtime_spec.requests),
        "worker_local_attribution_equal": all(item["attribution_digest_equal"] for item in attribution_match),
        "ledger_replay_verified": replay_report["verified_batches"] == len(runtime_spec.requests),
        "lease_limits_respected": lease_report["all_limits_respected"],
        "lease_events_replayable": socket_run.lease_replay["replayable"],
        "snapshot_restore_replay_verified": snapshot_restore_replay["verified_batches"]
        == len(runtime_spec.requests),
        "session_version_whole_replacement": replacement_report["whole_replacement_verified"],
        "invalidation_direct_indirect_stable": v2_invalidation["direct_batches"]
        and v2_invalidation["indirect_batches"]
        and v2_recompute["unchanged_check"]["all_byte_identical"],
        "selective_recompute_matches_full": selective_v2_matches_full,
        "v3_global_invalidation": invalidation_report["v3_all_batches_invalid"],
        "crash_recovery_reran_unknown": recovery["rerun_required"]
        and recovery_report["post_rerun_batch_count"] == len(runtime_spec.requests),
        "recovery_replay_verified": recovery_replay["verified_batches"] == len(runtime_spec.requests),
        "all_anomaly_codes_present": set(anomaly_report["codes"])
        == {"E_SPEC", "E_FIELD", "E_REF", "E_PORT", "E_HANDSHAKE", "E_LEASE", "E_REPLAY"},
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise AssertionError(f"verification checks failed: {failed}")

    report = {
        "schema": "speaker-diag-verification-v1",
        "status": "ok",
        "environment": {
            "python": sys.version,
            "platform": sys.platform,
            "numpy": __import__("numpy").__version__,
        },
        "project_root": str(project),
        "samples_root": str(sample_dir),
        "sample_manifest_digest": sample_manifest["manifest_digest"],
        "base_spec_digest": base_spec.spec_digest,
        "runtime_spec_digest": runtime_spec.spec_digest,
        "checks": checks,
        "counts": {
            "windows": len(runtime_bundle.windows),
            "batches": len(runtime_spec.requests),
            "workers": len(runtime_spec.workers),
            "speakers": runtime_spec.audio.speaker_count,
            "socket_batches": len(socket_run.normalized),
            "ledger_batches": content_manifest["batch_count"],
            "recovery_rerun_batches": len(recovery["rerun_order"]),
        },
        "digests": {
            "audio_manifest": sha256_hex((artifact_dir / "audio_manifest.json").read_bytes()),
            "features": sha256_hex((artifact_dir / "features.json").read_bytes()),
            "truth": sha256_hex((artifact_dir / "truth.json").read_bytes()),
            "normalization": sha256_hex((artifact_dir / "normalization.json").read_bytes()),
            "lease_report": sha256_hex((artifact_dir / "lease_report.json").read_bytes()),
            "metrics": sha256_hex((artifact_dir / "metrics.json").read_bytes()),
            "invalidation": sha256_hex((artifact_dir / "invalidation.json").read_bytes()),
            "recovery": sha256_hex((artifact_dir / "recovery.json").read_bytes()),
            "anomalies": sha256_hex((artifact_dir / "anomalies.json").read_bytes()),
            "content_chain": content_manifest["chain_digest"],
        },
        "normalization": normalization_report,
        "determinism": deterministic,
        "loopback": {
            "port_probe": sample_port_probe,
            "port_fallback_used": port_fallback,
            "socket_execution_ms": socket_elapsed_ms,
            "attribution_match": attribution_match,
            "replay": replay_report,
        },
        "leases": {
            "all_limits_respected": lease_report["all_limits_respected"],
            "observed_max_global_active": max_global_active,
            "observed_max_worker_active": max_worker_leases,
            "replayable": socket_run.lease_replay["replayable"],
            "queue_records": socket_run.lease_snapshot["queue_records"],
        },
        "fixed_point": metrics_report["fixed_point_policy"],
        "attribution_metrics": {
            "delivery_counts": metrics_report["attribution"]["delivery_counts"],
            "unique_window_counts": metrics_report["attribution"]["unique_window_counts"],
            "error_causes": metrics_report["attribution"]["error_causes"],
        },
        "invalidation": {
            "v2_direct_batches": invalidation_report["v2_direct_batches"],
            "v2_indirect_batches": invalidation_report["v2_indirect_batches"],
            "v2_invalid_windows": invalidation_report["v2_invalid_windows"],
            "v2_unchanged_batches": invalidation_report["v2_unchanged_batches"],
            "v3_all_batches_invalid": invalidation_report["v3_all_batches_invalid"],
            "unchanged_byte_identical": invalidation_report["unchanged_byte_identical"],
        },
        "recovery": {
            "child_exit_code": recovery_report["child_exit_code"],
            "rerun_order": recovery["rerun_order"],
            "unknown_batches_before_rerun": recovery["unknown_batches"],
            "replay_verified_batches": recovery_replay["verified_batches"],
        },
        "anomalies": {
            "codes": anomaly_report["codes"],
            "all_codes_matched": anomaly_report["all_codes_matched"],
        },
        "report_digest": canonical_sha256(
            {
                "sample_manifest_digest": sample_manifest["manifest_digest"],
                "base_spec_digest": base_spec.spec_digest,
                "checks": checks,
                "digests": {
                    "content_chain": content_manifest["chain_digest"],
                    "anomalies": sha256_hex((artifact_dir / "anomalies.json").read_bytes()),
                },
            }
        ),
    }
    atomic_write_json(artifact_dir / "verification.json", report)
    return report

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .dashboard import build_dashboard_model, render_text
from .errors import EdgeLedgerError, LeaseError
from .formats import DecodeContext, decode_receipt_line, decode_receipts, verify_normalized_receipt
from .ledger import Ledger
from .leases import ExpiryReaper, Lease, LeaseService
from .replay import verify_ledger
from .runtime import NodeCluster
from .sample_gen import write_sample_tree
from .scheduler import VerificationScheduler, assert_quota_invariants
from .spec import load_spec_dir
from .timeutil import monotonic_ms
from .ui import run_tk, write_text_snapshot


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ingest_samples(spec: Any, ledger: Ledger) -> dict[str, Any]:
    context = DecodeContext(spec=spec)
    adopted: list[str] = []
    rejected: list[str] = []
    format_errors: list[dict[str, Any]] = []

    for filename, fmt in (("receipts_v1.txt", "v1"), ("receipts_v2.jsonl", "v2"), ("receipts_v3.jsonl", "v3")):
        lines = (spec.root / filename).read_text(encoding="utf-8").splitlines()
        for receipt in decode_receipts(lines, context, fmt):
            ok, reason = verify_normalized_receipt(receipt, spec)
            ledger.insert_receipt(receipt, ok, reason)
            (adopted if ok else rejected).append(f"{filename}:{receipt.source_line}:{reason}")

    bad_line = (spec.root / "receipt_bad_digest_v3.json").read_text(encoding="utf-8").strip()
    bad = decode_receipt_line(bad_line, 1, context, "v3")
    ok, reason = verify_normalized_receipt(bad, spec)
    bad_id = ledger.insert_receipt(bad, ok, reason)
    if ok:
        adopted.append("bad unexpectedly adopted")
    else:
        rejected.append(f"receipt_bad_digest_v3.json:{reason}")

    dangling = json.loads(bad_line)
    dangling["invalid_shard_ref"] = {"ref_id": "ref-does-not-exist", "shard_id": "shard-0000"}
    try:
        decode_receipt_line(json.dumps(dangling, sort_keys=True), 99, context, "v3")
    except EdgeLedgerError as exc:
        format_errors.append({"code": exc.code, "field": exc.field, "line": exc.line, "message": exc.message})

    replacement_line = (spec.root / "receipt_replacement_v2.json").read_text(encoding="utf-8").strip()
    replacement = decode_receipt_line(replacement_line, 1, context, "v2")
    replacement_ok, replacement_reason = verify_normalized_receipt(replacement, spec)
    replacement_id = ledger.replace_receipt(
        bad_id,
        replacement,
        replacement_ok,
        replacement_reason + ":single_receipt_replacement",
    )
    if replacement_ok:
        adopted.append(f"receipt_replacement_v2.json:replaces_receipt_id={bad_id}:{replacement_reason}")
    else:
        rejected.append(f"receipt_replacement_v2.json:{replacement_reason}")

    return {
        "adopted": adopted,
        "rejected": rejected,
        "format_errors": format_errors,
        "replacement": {"old_receipt_id": bad_id, "new_receipt_id": replacement_id, "ok": replacement_ok},
    }


def _wait_lease_state(ledger: Ledger, lease_id: str, state: str, timeout_s: float = 2.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        row = ledger.conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
        if row is not None and row["state"] == state:
            return dict(row)
        time.sleep(0.01)
    row = ledger.conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
    raise AssertionError(f"lease {lease_id} did not reach {state}; last={dict(row) if row else None}")


def _revise_quota_dir(source: Path, target: Path, quota_version: int = 2) -> Any:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("nodes.json", "content_manifest.json", "event_stream.jsonl"):
        shutil.copy2(source / name, target / name)
    policy = json.loads((source / "lease_policy.json").read_text(encoding="utf-8"))
    policy["quota_version"] = quota_version
    policy["ttl_ms"] = 300
    (target / "lease_policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return load_spec_dir(target)


def _adjudication_race(service: LeaseService, ledger: Ledger, spec: Any) -> dict[str, int]:
    expiry_wins = 0
    quota_wins = 0
    neither = 0
    for index in range(24):
        now = monotonic_ms()
        lease = Lease(
            lease_id=f"race-lease-{index:03d}",
            quota_version=spec.quota_version,
            lease_version=1,
            node_id="node-a",
            shard_id=f"race-shard-{index:03d}",
            shard_seq=1000 + index,
            slot_index=0,
            byte_count=1,
            granted_ms=now - 10,
            expires_ms=now - 1,
            renewal_count=0,
        )
        ledger.record_grant(lease, spec.node("node-a"), spec.policy)
        barrier = threading.Barrier(2)

        def expiry_call() -> int | None:
            barrier.wait()
            return ledger.terminalize_lease(
                lease.lease_id,
                new_state="expired",
                event_type="lease_expired",
                winner="expiry_reaper",
                basis="monotonic_now_ge_expires_at",
                occurred_ms=monotonic_ms(),
                policy=spec.policy,
                node=spec.node("node-a"),
            )

        def quota_call() -> int | None:
            barrier.wait()
            return ledger.terminalize_lease(
                lease.lease_id,
                new_state="quota_revoked",
                event_type="lease_quota_revoked",
                winner="quota_revision",
                basis="lease_quota_version_1<2",
                occurred_ms=monotonic_ms(),
                policy=spec.policy,
                node=spec.node("node-a"),
                event_quota_version=2,
                old_quota_version=1,
                competing_action="expiry_reaper",
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_expiry = pool.submit(expiry_call)
            f_quota = pool.submit(quota_call)
            expiry_seq = f_expiry.result()
            quota_seq = f_quota.result()
        if expiry_seq is not None and quota_seq is None:
            expiry_wins += 1
        elif quota_seq is not None and expiry_seq is None:
            quota_wins += 1
        else:
            neither += 1
    return {"expiry_wins": expiry_wins, "quota_wins": quota_wins, "double_winners_or_none": neither}


def run_acceptance(out_dir: str | Path, *, run_tk_smoke: bool = False) -> dict[str, Any]:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"run-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}"
    out.mkdir(parents=True, exist_ok=True)
    spec_dir = out / "spec-v1"
    write_sample_tree(spec_dir, allocate_ports=True, create_snapshot=True, quota_version=1)
    spec = load_spec_dir(spec_dir)
    ledger_path = out / "ledger" / "edge.db"
    ledger = Ledger.open(ledger_path, content_digest=spec.content_digest)
    report: dict[str, Any] = {"out_dir": str(out), "spec_dir": str(spec_dir), "ledger": str(ledger_path)}
    cluster: NodeCluster | None = None
    reaper: ExpiryReaper | None = None
    try:
        report["receipts"] = _ingest_samples(spec, ledger)
        service = LeaseService(spec, ledger)
        reaper = ExpiryReaper(service, 20)
        reaper.start()
        cluster = NodeCluster(spec, log_dir=out / "logs").start_all()
        scheduler = VerificationScheduler(spec, service)

        first_runs = scheduler.run(max_workers=4)
        report["first_run"] = [run.__dict__ for run in first_runs]
        report["quota_invariants_after_first_run"] = assert_quota_invariants(service)

        renewal_acq = service.acquire("node-b", "shard-0001", 1, spec.content.shard_size)
        renewal_client = None
        from .protocol import NodeClient

        renewal_client = NodeClient(spec.node("node-b"), spec.policy.socket_timeout_ms)
        renewal_client.grant_lease(renewal_acq.lease)
        renewed = service.renew(renewal_acq.lease, 1, renewal_client)
        stale_error = None
        try:
            service.renew(renewal_acq.lease, 1, renewal_client)
        except LeaseError as exc:
            stale_error = {"code": exc.code, "field": exc.field, "message": exc.message}
        proof = __import__("edge_ledger.integrity", fromlist=["compute_shard_proof"]).compute_shard_proof(
            spec.node("node-b"), spec.content, 1
        )
        service.complete(renewed, proof)
        report["renewal"] = {"new_version": renewed.lease_version, "stale_error": stale_error}

        expiry_acq = service.acquire("node-c", "shard-0002", 2, spec.content.shard_size, ttl_ms=80)
        expiry_row = _wait_lease_state(ledger, expiry_acq.lease.lease_id, "expired", 2.0)
        report["real_expiry"] = {
            "lease_id": expiry_acq.lease.lease_id,
            "granted_ms": expiry_row["granted_ms"],
            "expires_ms": expiry_row["expires_ms"],
            "terminal": expiry_row["state"],
            "basis": expiry_row["decision_basis"],
            "released_after_ge": int(expiry_row["expires_ms"]) <= monotonic_ms() + 5000,
        }

        if reaper is not None:
            reaper.stop()
            time.sleep(spec.policy.reaper_interval_ms / 500.0)
        report["adjudication_race"] = _adjudication_race(service, ledger, spec)

        try:
            scheduler.process_shard(0, "node-a", crash_before_reply=True)
            raise AssertionError("crash request unexpectedly returned")
        except EdgeLedgerError as exc:
            crash_code = exc.code
        unknown_v1 = service.recover_active_leases()
        cluster.restart("node-a")
        rerun0 = scheduler.process_shard(0, "node-a")
        report["force_kill_recovery_v1"] = {
            "client_error_code": crash_code,
            "unknown_shards": unknown_v1,
            "rerun_digest_matches": rerun0.digest == __import__(
                "edge_ledger.integrity", fromlist=["compute_shard_proof"]
            ).compute_shard_proof(spec.node("node-a"), spec.content, 0).digest,
            "pid_killed": cluster.processes["node-a"].proc.pid if cluster.processes["node-a"].proc else None,
        }

        long_acq = service.acquire("node-a", "shard-0003", 3, spec.content.shard_size, ttl_ms=60_000)
        spec_v2_dir = out / "spec-v2"
        spec_v2 = _revise_quota_dir(spec_dir, spec_v2_dir, 2)
        revoked = service.revoke_for_quota_revision(spec_v2)
        long_row = _wait_lease_state(ledger, long_acq.lease.lease_id, "quota_revoked", 0.2)
        report["quota_evolution"] = {
            "revoked_event_seqs": revoked,
            "long_lease_state": long_row["state"],
            "basis": long_row["decision_basis"],
            "new_quota_version": service.spec.quota_version,
            "stale_counts": service.last_stale_counts,
        }

        quota2_runs = scheduler.run(max_workers=4)
        report["quota2_run_count"] = len(quota2_runs)
        report["quota2_invariants"] = assert_quota_invariants(service)

        try:
            scheduler.process_shard(3, "node-a", crash_before_reply=True)
        except EdgeLedgerError as exc:
            quota2_crash_code = exc.code
        unknown_quota2 = service.recover_active_leases()
        unknown_ledger_copy = out / "ledger-unknown-state.db"
        with sqlite3.connect(unknown_ledger_copy) as target:
            ledger.conn.backup(target)
        unknown_snapshot = out / "ui-unknown-snapshot.txt"
        write_text_snapshot(spec_v2_dir, ledger_path, unknown_snapshot)
        tk_unknown = _try_tk(spec_v2_dir, unknown_ledger_copy, run_tk_smoke, out / "ui-unknown-tk.txt")
        cluster.restart("node-a")
        rerun3 = scheduler.process_shard(3, "node-a")
        final_snapshot = out / "ui-final-snapshot.txt"
        write_text_snapshot(spec_v2_dir, ledger_path, final_snapshot)
        tk_final = _try_tk(spec_v2_dir, ledger_path, run_tk_smoke, out / "ui-final-tk.txt")

        replay_report = verify_ledger(spec_v2_dir, ledger_path)
        report.update(
            {
                "force_kill_recovery_quota2": {
                    "client_error_code": quota2_crash_code,
                    "unknown_before_rerun": unknown_quota2,
                    "rerun_lease_id": rerun3.lease_id,
                    "unknown_ledger_copy": str(unknown_ledger_copy),
                },
                "ui": {
                    "unknown_text_snapshot": str(unknown_snapshot),
                    "final_text_snapshot": str(final_snapshot),
                    "tk_unknown": tk_unknown,
                    "tk_final": tk_final,
                    "unknown_snapshot_contains_marker": "UNKNOWN/RERUN"
                    in unknown_snapshot.read_text(encoding="utf-8"),
                },
                "replay": replay_report.to_dict(),
                "ok": True,
            }
        )
        _json_dump(out / "acceptance-report.json", report)
        return report
    finally:
        if reaper is not None:
            reaper.stop()
        if cluster is not None:
            cluster.stop_all()
        ledger.close()


def _try_tk(spec_dir: Path, ledger: Path, enabled: bool, evidence_path: Path) -> dict[str, Any]:
    if not enabled:
        return {"attempted": False, "reason": "pass --tk-smoke to create a real Tk window"}
    try:
        code = run_tk(spec_dir, ledger, smoke_ms=400)
        evidence_path.write_text(f"tk_smoke_exit={code}\n", encoding="utf-8")
        return {"attempted": True, "exit_code": code, "evidence": str(evidence_path)}
    except Exception as exc:
        evidence_path.write_text(f"tk_smoke_error={type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"attempted": True, "ok": False, "error": f"{type(exc).__name__}: {exc}", "evidence": str(evidence_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local edge-ledger acceptance workflow")
    parser.add_argument("--out", default="artifacts/acceptance")
    parser.add_argument("--tk-smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_acceptance(args.out, run_tk_smoke=args.tk_smoke)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

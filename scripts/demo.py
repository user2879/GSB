#!/usr/bin/env python3
"""Repeatable end-to-end demonstration with local deterministic inputs only."""

from __future__ import annotations

import dataclasses
import json
import random
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lossdiag.diagnosis import diagnose_streams
from lossdiag.echo import EchoStateStore, recover_echo_state, run_echo_replay
from lossdiag.errors import DiagnosticError
from lossdiag.events import load_event_bundle
from lossdiag.pipeline import build_session
from lossdiag.rates import aggregate_rates
from lossdiag.replay import verify_ledger
from lossdiag.store import WorkbenchStore, restore_snapshot, restored_session_view
from lossdiag.util import canonical_json, content_digest, pretty_json
from lossdiag.wire import read_wire_bundle


SAMPLES = ROOT / "samples"
WIRE = SAMPLES / "observations.bin"
MANIFEST = SAMPLES / "wire_manifest.json"
ARTIFACTS = ROOT / "artifacts" / "demo"


class Demo:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            details = function()
        except Exception as exc:
            self.checks.append(
                {
                    "name": name,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            return {}
        self.checks.append({"name": name, "ok": True, "details": details})
        print(f"[PASS] {name}")
        return details

    def expect_error(self, name: str, code: str, function: Callable[[], Any]) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            try:
                function()
            except DiagnosticError as exc:
                if exc.code != code:
                    raise AssertionError(f"expected {code}, got {exc.code}") from exc
                return {"code": exc.code, "details": exc.details}
            raise AssertionError(f"expected {code}, no error was raised")

        return self.check(name, run)


def _free_high_port() -> int:
    for port in range(47731, 47831):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port in 47731..47830")


def _wait_for_peer(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise RuntimeError(f"echo peer exited early: {stdout}\n{stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("echo peer did not become ready")


def _wait_for_sent_job(state_db: str, request_id: str) -> str:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        store = EchoStateStore(state_db)
        try:
            jobs = {job["request_id"]: job for job in store.jobs()}
        finally:
            store.close()
        if request_id in jobs and jobs[request_id]["status"] == "sent":
            return jobs[request_id]["status"]
        time.sleep(0.05)
    raise RuntimeError(f"echo job {request_id} did not reach persisted sent state")


def run_demo() -> int:
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    ARTIFACTS.mkdir(parents=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_samples.py")], check=True)

    demo = Demo()
    db_path = ARTIFACTS / "workbench.db"
    echo_db = ARTIFACTS / "echo_state.db"

    wire_summary = demo.check(
        "decode 27 compact observation frames",
        lambda: {
            "frame_count": len(read_wire_bundle(WIRE, MANIFEST).frames),
            "wire_digest": read_wire_bundle(WIRE, MANIFEST).wire_digest,
        },
    )
    if wire_summary.get("frame_count") != 27:
        raise AssertionError("sample wire frame count changed")

    def normalize_versions() -> dict[str, Any]:
        result = {}
        for version in (1, 2, 3):
            bundle = load_event_bundle(SAMPLES / f"events_v{version}.json")
            result[bundle.version] = {
                "records": len(bundle.records),
                "padded_fields_first": list(bundle.records[0].padded_fields),
                "first_window": (
                    bundle.records[0].time_window_ms.to_dict()
                    if bundle.records[0].time_window_ms
                    else None
                ),
            }
        assert result["v1"]["padded_fields_first"] == [
            "link_id",
            "reorder_depth",
            "retransmit",
            "time_window_ms",
        ]
        assert result["v2"]["padded_fields_first"] == ["reorder_depth", "retransmit"]
        assert result["v3"]["padded_fields_first"] == []
        return result

    demo.check("normalize event-list v1/v2/v3 with explicit defaults", normalize_versions)

    session = build_session(
        wire_path=str(WIRE),
        wire_manifest_path=str(MANIFEST),
        event_paths=[str(SAMPLES / "events_v3.json")],
        window_ms=100,
    )

    def alignment_and_gaps() -> dict[str, Any]:
        flow_a = next(flow for flow in session.diagnoses if "10.10.0.1" in flow.flow_id)
        classes = {
            (gap.start_seq, gap.end_seq): gap.classification for gap in flow_a.gaps
        }
        assert classes[(3, 3)] == "true_loss"
        assert classes[(5, 5)] == "retransmission"
        assert classes[(6, 7)] == "reordering"
        assert session.alignment.unaligned_flow_ids == []
        return {
            "offset_to_frame_ms": session.alignment.records[0].offset_to_frame_ms,
            "flow_a_gap_classes": {
                f"{key[0]}..{key[1]}": value for key, value in classes.items()
            },
        }

    demo.check("translate time bases and classify true loss/retransmission/reordering", alignment_and_gaps)

    def traversal_independence() -> dict[str, Any]:
        base = {
            "diagnosis": [diagnosis.to_dict() for diagnosis in session.diagnoses],
            "rates": session.rate_report,
        }
        rng = random.Random(20261122)
        frames = list(session.frame_bundle.frames)
        rng.shuffle(frames)
        shuffled_frames = dataclasses.replace(session.frame_bundle, frames=frames)
        shuffled_bundles = []
        for bundle in session.event_bundles:
            records = list(bundle.records)
            rng.shuffle(records)
            shuffled_bundles.append(dataclasses.replace(bundle, records=records))
        shuffled_diagnoses = diagnose_streams(shuffled_frames, shuffled_bundles)
        shuffled_rates = aggregate_rates(shuffled_frames, shuffled_diagnoses, 100)
        shuffled = {
            "diagnosis": [diagnosis.to_dict() for diagnosis in shuffled_diagnoses],
            "rates": shuffled_rates,
        }
        assert canonical_json(base) == canonical_json(shuffled)
        return {
            "base_digest": content_digest(base),
            "shuffled_digest": content_digest(shuffled),
            "byte_identical": True,
        }

    demo.check("shuffle frames/events and obtain byte-identical conclusions", traversal_independence)

    def rate_policy() -> dict[str, Any]:
        ranking = session.rate_report["ranking"]
        assert ranking[0]["rate_ppm"] == 200000
        assert ranking[1]["rate_ppm"] == 181818
        assert ranking[2]["rate_ppm"] == 181818
        assert ranking[1]["error_affected"] and ranking[2]["error_affected"]
        assert session.rate_report["rate_comparison"][0]["tie_breaker"] == "flow_id_asc"
        return {
            "rounding": session.rate_report["rounding"],
            "combined_error_bound_ppm": session.rate_report["combined_error_bound_ppm"],
            "ranking": ranking,
        }

    demo.check("integer ppm windows with error-bound stable tie-break", rate_policy)

    snapshot_details: dict[str, Any] = {}

    def snapshot_replacement() -> dict[str, Any]:
        with WorkbenchStore(db_path) as store:
            snapshot_id = store.save_snapshot(session)
            current_v2 = build_session(
                wire_path=str(WIRE),
                wire_manifest_path=str(MANIFEST),
                event_paths=[str(SAMPLES / "events_v2.json")],
                window_ms=100,
            )
            restored = restore_snapshot(store, snapshot_id, current_session=current_v2)
        comparison = restored["comparison"]
        assert comparison["replaced"] is True
        assert comparison["replacement_mode"] == "whole_payload_replace"
        assert any(item["field"] == "event_versions" for item in comparison["differences"])
        snapshot_details.update({"snapshot_id": snapshot_id, "comparison": comparison})
        return {
            "snapshot_uid": snapshot_id,
            "replaced": comparison["replaced"],
            "difference_fields": [item["field"] for item in comparison["differences"]],
        }

    demo.check("snapshot v2 current -> v3 whole-state replacement", snapshot_replacement)

    demo.expect_error(
        "inject snapshot digest mismatch -> E_SNAPSHOT",
        "E_SNAPSHOT",
        lambda: restore_snapshot(
            WorkbenchStore(db_path),
            snapshot_details["snapshot_id"],
            inject_digest_mismatch=True,
        ),
    )

    ledger_details: dict[str, Any] = {}

    def ledger_roundtrip() -> dict[str, Any]:
        with WorkbenchStore(db_path) as store:
            ledger_id = store.save_ledger(session)
            result = verify_ledger(store, ledger_id)
        assert result["matches"] is True
        ledger_details["ledger_id"] = ledger_id
        return {
            "ledger_uid": ledger_id,
            "matches": result["matches"],
            "report_digest": result["ledger_report_digest"],
        }

    demo.check("ledger replay reproduces every item", ledger_roundtrip)

    def restored_snapshot_then_ledger_replay() -> dict[str, Any]:
        current_v2 = build_session(
            wire_path=str(WIRE),
            wire_manifest_path=str(MANIFEST),
            event_paths=[str(SAMPLES / "events_v2.json")],
            window_ms=100,
        )
        with WorkbenchStore(db_path) as store:
            restored = restore_snapshot(
                store,
                snapshot_details["snapshot_id"],
                current_session=current_v2,
            )
            view = restored_session_view(restored)
            replay = verify_ledger(store, ledger_details["ledger_id"])
        assert view.event_versions == ["v3"]
        assert replay["matches"] is True
        return {
            "restored_event_versions": view.event_versions,
            "restored_report_digest": view.report_digest,
            "ledger_replay_matches": replay["matches"],
        }

    demo.check(
        "restore snapshot by whole replacement, then replay ledger item-by-item",
        restored_snapshot_then_ledger_replay,
    )

    def ledger_first_difference() -> dict[str, Any]:
        document = json.loads((SAMPLES / "events_v3.json").read_text(encoding="utf-8"))
        document["records"][0]["loss_count"] = 2
        changed = ARTIFACTS / "events_v3_changed.json"
        changed.write_text(pretty_json(document) + "\n", encoding="utf-8")
        try:
            with WorkbenchStore(db_path) as store:
                verify_ledger(
                    store,
                    ledger_details["ledger_id"],
                    event_paths=[str(changed)],
                )
        except DiagnosticError as exc:
            assert exc.code == "E_REPLAY"
            assert exc.details["flow_id"] == "10.10.0.1:40000->10.20.0.1:5000/tcp"
            assert exc.details["gap_interval"]["start_seq"] == 3
            assert exc.details["field"] == "gap_presence"
            return exc.to_dict()
        raise AssertionError("ledger drift did not raise E_REPLAY")

    demo.check("ledger drift localizes first flow/gap/field as E_REPLAY", ledger_first_difference)

    demo.expect_error(
        "inject wire corruption -> E_WIRE",
        "E_WIRE",
        lambda: read_wire_bundle(WIRE, MANIFEST, corrupt=True),
    )
    demo.expect_error(
        "inject event version conflict -> E_FORMAT",
        "E_FORMAT",
        lambda: load_event_bundle(SAMPLES / "events_v3.json", inject_version_conflict=True),
    )

    def echo_flow() -> dict[str, Any]:
        port = _free_high_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "lossdiag.cli",
                "echo-serve",
                "--port",
                str(port),
                "--state-db",
                str(echo_db),
            ],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_peer(process, port)
            normal = run_echo_replay(
                session,
                host="127.0.0.1",
                port=port,
                request_id="demo-echo-normal",
                send_ack=True,
            )
            assert normal["verification"]["matches"] is True
            no_ack = run_echo_replay(
                session,
                host="127.0.0.1",
                port=port,
                request_id="demo-echo-forced-kill",
                send_ack=False,
            )
            assert no_ack["verification"]["matches"] is True
            _wait_for_sent_job(str(echo_db), "demo-echo-forced-kill")
            process.kill()
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        recovery = recover_echo_state(str(echo_db))
        forced_job = next(
            job for job in recovery["jobs"] if job["request_id"] == "demo-echo-forced-kill"
        )
        assert recovery["complete_incident_reconstruction"] is False
        assert forced_job["peer_state_confirmed"] is False
        assert "client_ack" in forced_job["unconfirmed_fields"]
        return {
            "port": port,
            "normal_echo_matches": normal["verification"]["matches"],
            "forced_kill_status": forced_job["persisted_status"],
            "unconfirmed_fields": forced_job["unconfirmed_fields"],
            "complete_incident_reconstruction": recovery["complete_incident_reconstruction"],
        }

    demo.check("loopback replay, forced SIGKILL, persisted-state-only recovery", echo_flow)

    cli_wire = subprocess.run(
        [
            sys.executable,
            "-m",
            "lossdiag.cli",
            "inspect-wire",
            "--wire",
            str(WIRE),
            "--manifest",
            str(MANIFEST),
            "--inject",
            "wire-corrupt",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    cli_event = subprocess.run(
        [
            sys.executable,
            "-m",
            "lossdiag.cli",
            "inspect-events",
            "--events",
            str(SAMPLES / "events_v3.json"),
            "--inject",
            "event-version-conflict",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    cli_snapshot = subprocess.run(
        [
            sys.executable,
            "-m",
            "lossdiag.cli",
            "snapshot-restore",
            "--db",
            str(db_path),
            "--snapshot-id",
            snapshot_details["snapshot_id"],
            "--inject",
            "snapshot-digest-mismatch",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    cli_codes = {
        "wire": json.loads(cli_wire.stdout)["code"],
        "format": json.loads(cli_event.stdout)["code"],
        "snapshot": json.loads(cli_snapshot.stdout)["code"],
    }
    assert cli_codes == {"wire": "E_WIRE", "format": "E_FORMAT", "snapshot": "E_SNAPSHOT"}
    assert cli_wire.returncode == cli_event.returncode == cli_snapshot.returncode == 2
    demo.check(
        "CLI anomaly injection contract",
        lambda: {"exit_codes": 2, "codes": cli_codes},
    )

    all_ok = all(item["ok"] for item in demo.checks)
    report = {
        "ok": all_ok,
        "deterministic": True,
        "network_scope": "127.0.0.1 only",
        "report_digest": session.report_digest,
        "checks": demo.checks,
    }
    report_path = ARTIFACTS / "demo_report.json"
    report_path.write_text(pretty_json(report) + "\n", encoding="utf-8")
    print(f"\nDemo report: {report_path}")
    print(f"Report digest: {session.report_digest}")
    print(f"Result: {'PASS' if all_ok else 'FAIL'} ({sum(item['ok'] for item in demo.checks)}/{len(demo.checks)} checks)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(run_demo())

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .client import dispatch_batch, handshake_only, renew_lease, grant_lease
from .cluster import NodeCluster
from .demo_spec import build_demo_declaration
from .errors import CloudShardError, FieldError, PortInUseError, ProtocolTimeoutError, VerifyError
from .pipeline import check_batch_boundaries, execute_pipeline
from .snapshot import atomic_write_json
from .spec import NodeSpec, load_spec
from .verifier import verify_spec_file


REPO_ROOT = Path(__file__).resolve().parents[2]


def _wait_port_free(port: int, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.05)


def _blackhole_server() -> tuple[int, socket.socket, list[socket.socket]]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    connected: list[socket.socket] = []

    def accept_loop() -> None:
        while True:
            try:
                conn, _ = server.accept()
                connected.append(conn)
            except OSError:
                return

    thread = threading.Thread(target=accept_loop, daemon=True)
    thread.start()
    return int(server.getsockname()[1]), server, connected


def _missing_field_server() -> tuple[int, socket.socket]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def serve_once() -> None:
        conn, _ = server.accept()
        try:
            reader = conn.makefile("rb")
            reader.readline()
            conn.sendall(b'{"type":"hello_ack","node_id":"fake"}\n')
            reader.readline()
            conn.sendall(b'{"type":"receipt","node_id":"fake","shard_id":"x","revision":1}\n')
        finally:
            conn.close()

    threading.Thread(target=serve_once, daemon=True).start()
    return int(server.getsockname()[1]), server


def run_acceptance(workdir: str | Path) -> dict[str, Any]:
    root = Path(workdir)
    if root.exists():
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def check(name: str, fn: Callable[[], Any]) -> Any:
        started = time.monotonic()
        try:
            detail = fn()
            checks.append({"name": name, "ok": True, "duration_ms": round((time.monotonic() - started) * 1000, 1), "detail": detail})
            return detail
        except Exception as exc:
            payload = getattr(exc, "to_payload", lambda: {"error": str(exc)})()
            checks.append(
                {
                    "name": name,
                    "ok": False,
                    "duration_ms": round((time.monotonic() - started) * 1000, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                    "payload": payload,
                }
            )
            return None

    spec_path = root / "declaration.json"
    spec_path.write_text(json.dumps(build_demo_declaration()), encoding="utf-8")
    spec = load_spec(spec_path)

    def check_spec_invalid() -> dict[str, str]:
        invalid_path = root / "invalid.json"
        data = build_demo_declaration()
        data["nodes"][1]["port"] = data["nodes"][0]["port"]
        invalid_path.write_text(json.dumps(data), encoding="utf-8")
        try:
            load_spec(invalid_path)
        except CloudShardError as exc:
            if exc.code != "E_SPEC" or exc.field != "port":
                raise
            return {"code": exc.code, "field": exc.field, "path": exc.path or ""}
        raise AssertionError("invalid spec was accepted")

    check("espec_first_invalid_location", check_spec_invalid)
    batch = check("arbitrary_batch_boundary_byte_replay", lambda: check_batch_boundaries(spec))

    happy_dir = root / "happy"
    happy_report = check(
        "real_tcp_dispatch_and_everify",
        lambda: (
            execute_pipeline(spec_path, happy_dir),
            verify_spec_file(spec_path, happy_dir / "ledger.sqlite"),
        )[1],
    )

    def check_eport() -> dict[str, Any]:
        node = spec.node_by_id["n1"]
        occupier = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupier.bind((node.host, node.port))
        occupier.listen(1)
        cluster = NodeCluster(spec, spec_path, root / "eport")
        try:
            try:
                cluster.start_node(node)
            except PortInUseError as exc:
                return {"code": exc.code, "field": exc.field or "port", "port": node.port}
            raise AssertionError("occupied port did not raise E_PORT")
        finally:
            occupier.close()
            cluster.stop_all()
            _wait_port_free(node.port)

    check("eport_real_bind_conflict", check_eport)

    def check_etimeout() -> dict[str, Any]:
        port, server, conns = _blackhole_server()
        try:
            try:
                handshake_only("127.0.0.1", port, timeout_ms=180)
            except ProtocolTimeoutError as exc:
                return {"code": exc.code, "field": exc.field or "readline", "timeout_ms": 180}
            raise AssertionError("blackhole peer did not cause E_TIMEOUT")
        finally:
            server.close()
            for conn in conns:
                conn.close()

    check("etimeout_real_socket_timeout", check_etimeout)

    def check_efield() -> dict[str, Any]:
        port, server = _missing_field_server()
        node = NodeSpec("fake", "127.0.0.1", port, 1, 1, 0, 1)
        try:
            try:
                dispatch_batch(node, shard_id="x", revision=1, samples={"groups": [0], "values_q": [1]}, timeout_ms=1000)
            except FieldError as exc:
                return {"code": exc.code, "field": exc.field or "response"}
            raise AssertionError("incomplete receipt did not cause E_FIELD")
        finally:
            server.close()

    check("efield_missing_receipt_field", check_efield)

    def check_elease_node() -> dict[str, Any]:
        lease_dir = root / "lease"
        with NodeCluster(spec, spec_path, lease_dir) as cluster:
            node = spec.node_by_id["n1"]
            grant_lease(
                node,
                {"lease_id": "acceptance-L", "shard_id": "s1", "lease_version": 1, "expires_at_ms": 10_000},
            )
            ack = renew_lease(node, lease_id="acceptance-L", expected_version=1, new_expires_at_ms=20_000)
            try:
                renew_lease(node, lease_id="acceptance-L", expected_version=1, new_expires_at_ms=30_000)
            except CloudShardError as exc:
                if exc.code != "E_LEASE":
                    raise
                return {"code": exc.code, "new_version": ack.get("lease_version"), "basis": exc.basis}
        raise AssertionError("stale lease version was accepted")

    check("elease_optimistic_node_check", check_elease_node)

    def check_crash_recovery() -> dict[str, Any]:
        crash_dir = root / "crash"
        crash_dir.mkdir(parents=True, exist_ok=True)
        crash_spec = crash_dir / "spec.json"
        crash_spec.write_text(json.dumps(build_demo_declaration()), encoding="utf-8")
        marker = crash_dir / "crash-marker.json"
        log = (crash_dir / "crash-parent.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        src = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [
            sys.executable,
            "-m",
            "cloudshard.cli",
            "run",
            "--spec",
            str(crash_spec),
            "--workdir",
            str(crash_dir),
            "--crash-before-commit",
            "s2",
            "--crash-marker",
            str(marker),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not marker.exists():
            if proc.poll() is not None:
                log.close()
                raise AssertionError(f"crash process exited early with {proc.returncode}; see {crash_dir / 'crash-parent.log'}")
            time.sleep(0.05)
        if not marker.exists():
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
            log.close()
            raise AssertionError("crash marker was not created")
        marker_data = json.loads(marker.read_text(encoding="utf-8"))
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
        log.close()

        report = execute_pipeline(crash_spec, crash_dir)
        verification = verify_spec_file(crash_spec, crash_dir / "ledger.sqlite")
        return {
            "killed_pid": marker_data["pid"],
            "shard_id": marker_data["shard_id"],
            "recovery_marks": report["recovery_marks"],
            "verify": {"ok": verification["ok"], "outcomes_verified": verification["outcomes_verified"]},
        }

    crash_detail = check("sigkill_sqlite_recovery_and_everify", check_crash_recovery)

    def check_everify_tamper() -> dict[str, Any]:
        source_db = happy_dir / "ledger.sqlite"
        tamper_db = root / "tampered.sqlite"
        src = sqlite3_connect(source_db)
        dst = sqlite3_connect(tamper_db)
        with dst:
            src.backup(dst)
        src.close()
        dst.execute("UPDATE meta SET value='tampered' WHERE key='spec'")
        dst.commit()
        dst.close()
        try:
            verify_spec_file(spec_path, tamper_db)
        except VerifyError as exc:
            return {"code": exc.code, "field": exc.field, "shard_id": exc.shard_id}
        raise AssertionError("tampered ledger passed E_VERIFY")

    check("everify_detects_tampering", check_everify_tamper)

    ok = all(item["ok"] for item in checks)
    result = {
        "ok": ok,
        "workdir": str(root),
        "python": sys.executable,
        "checks": checks,
        "crash_detail": crash_detail,
        "batch_all_equal": bool(batch and batch.get("all_equal")),
        "happy_verify_ok": bool(happy_report and happy_report.get("ok")),
    }
    atomic_write_json(root / "acceptance-report.json", result)
    atomic_write_json(Path("docs") / "acceptance-report.generated.json", result)
    return result


def sqlite3_connect(path: Path):
    import sqlite3

    return sqlite3.connect(str(path))

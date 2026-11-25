"""Deterministic local fault injection helpers."""

from __future__ import annotations

import json
import shutil
import socket
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .canonical import canonical_text
from .cluster import NodeCluster
from .errors import LeaseMiddlewareError
from .ledger import Ledger
from .protocol import LeaseNodeClient
from .service import LiveScenarioRunner, prepare_dir
from .spec import load_sample_dir
from .verify import verify_ledger


def _copy_sample(root: Path, destination: Path) -> Path:
    source = root / "samples"
    sample_copy = destination / "sample"
    if sample_copy.exists():
        shutil.rmtree(sample_copy)
    shutil.copytree(source, sample_copy)
    return sample_copy


def _result(kind: str, destination: Path, func: Callable[[], Any]) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        value = func()
        result = {"ok": True, "injected": kind, "result": value}
    except LeaseMiddlewareError as exc:
        result = {
            "ok": False,
            "injected": kind,
            "expected_code": exc.code,
            "error": exc.to_dict(),
        }
    (destination / "injection_result.json").write_text(
        canonical_text(result) + "\n", encoding="utf-8"
    )
    return result


def inject_spec(root: Path) -> dict[str, Any]:
    destination = root / "artifacts" / "injections" / "spec"

    def exercise() -> None:
        sample = _copy_sample(root, destination)
        nodes_path = sample / "nodes.json"
        doc = json.loads(nodes_path.read_text(encoding="utf-8"))
        doc["nodes"][0]["port"] = "corrupted-port"
        nodes_path.write_text(canonical_text(doc) + "\n", encoding="utf-8")
        load_sample_dir(sample)

    return _result("spec", destination, exercise)


def inject_spec_json(root: Path) -> dict[str, Any]:
    destination = root / "artifacts" / "injections" / "spec-json"

    def exercise() -> None:
        sample = _copy_sample(root, destination)
        (sample / "events.json").write_text('{"events": [\n', encoding="utf-8")
        load_sample_dir(sample)

    return _result("spec-json", destination, exercise)


def inject_stale_lease(root: Path) -> dict[str, Any]:
    destination = root / "artifacts" / "injections" / "lease"

    def exercise() -> dict[str, Any]:
        sample = _copy_sample(root, destination)
        events_path = sample / "events.json"
        doc = json.loads(events_path.read_text(encoding="utf-8"))
        for event in doc["events"]:
            if event.get("seq") == 7:
                event["version_tag"] = 1
        events_path.write_text(canonical_text(doc) + "\n", encoding="utf-8")
        bundle = load_sample_dir(sample)
        run_dir = prepare_dir(destination / "run")
        runner = LiveScenarioRunner(bundle, run_dir)
        try:
            return runner.run([node["node_id"] for node in bundle.nodes])
        finally:
            runner.close()

    return _result("lease", destination, exercise)


def inject_snapshot_digest(root: Path) -> dict[str, Any]:
    destination = root / "artifacts" / "injections" / "snapshot"

    def exercise() -> dict[str, Any]:
        prepare_dir(destination)
        ledger_path = destination / "tampered-ledger.sqlite"
        ledger = Ledger(ledger_path)
        ledger.append_records(
            [
                {
                    "entry_kind": "tamper_probe",
                    "event_seq": 1,
                    "lease_id": "lease-probe",
                    "generation": 1,
                    "ok": True,
                }
            ]
        )
        ledger.close()
        connection = sqlite3.connect(str(ledger_path))
        with connection:
            connection.execute(
                "UPDATE ledger_meta SET value=? WHERE key='content_digest'",
                ("sha256:" + "0" * 64,),
            )
            connection.execute(
                "UPDATE ledger_header SET content_digest=? WHERE id=1",
                ("sha256:" + "0" * 64,),
            )
        connection.close()
        Ledger.open(ledger_path)
        return {"unreachable": True}

    return _result("snapshot", destination, exercise)


def inject_verify(root: Path) -> dict[str, Any]:
    destination = root / "artifacts" / "injections" / "verify"

    def exercise() -> None:
        prepare_dir(destination)
        bundle = load_sample_dir(root / "samples")
        ledger = Ledger(destination / "semantic-wrong-ledger.sqlite")
        ledger.append_records(
            [
                {
                    "entry_kind": "ownership",
                    "event_seq": 1,
                    "equity_id": "equity-INJECTED",
                    "lease_id": "lease-a1",
                    "generation": 999,
                }
            ]
        )
        try:
            verify_ledger(bundle, ledger)
        finally:
            ledger.close()

    return _result("verify", destination, exercise)


def inject_port(root: Path) -> dict[str, Any]:
    destination = root / "artifacts" / "injections" / "port"
    destination.mkdir(parents=True, exist_ok=True)
    bundle = load_sample_dir(root / "samples")
    node = bundle.nodes[0]
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    result: dict[str, Any]
    try:
        holder.bind((node["host"], int(node["port"])))
        holder.listen(1)
        cluster = NodeCluster([node], log_dir=destination / "node-logs")
        try:
            cluster.start_all([node["node_id"]])
        finally:
            cluster.stop_all()
        raise RuntimeError("port injection did not fail")
    except LeaseMiddlewareError as exc:
        result = {
            "ok": False,
            "injected": "port",
            "expected_code": exc.code,
            "error": exc.to_dict(),
        }
    except OSError as exc:
        result = {
            "ok": False,
            "injected": "port",
            "expected_code": "E_PORT",
            "error": {"code": "E_PORT", "message": str(exc)},
        }
    finally:
        holder.close()
    (destination / "injection_result.json").write_text(
        canonical_text(result) + "\n", encoding="utf-8"
    )
    return result


def _raw_server(root: Path, kind: str, reply: bytes | None) -> dict[str, Any]:
    destination = root / "artifacts" / "injections" / kind
    destination.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted = threading.Event()

    def serve_once() -> None:
        try:
            conn, _ = server.accept()
            accepted.set()
            with conn:
                conn.settimeout(1.0)
                try:
                    conn.recv(4096)
                except OSError:
                    pass
                if kind == "timeout":
                    time.sleep(1.0)
                elif reply is not None:
                    conn.sendall(reply)
        except OSError:
            pass

    thread = threading.Thread(target=serve_once, daemon=True)
    thread.start()
    try:
        client = LeaseNodeClient("127.0.0.1", port, node_id="fault", timeout=0.2)
        if kind == "timeout":
            client.handshake()
        else:
            client.handshake()
        raise RuntimeError(f"{kind} injection did not fail")
    except LeaseMiddlewareError as exc:
        result = {
            "ok": False,
            "injected": kind,
            "expected_code": exc.code,
            "error": exc.to_dict(),
        }
    finally:
        server.close()
        thread.join(timeout=1)
    (destination / "injection_result.json").write_text(
        canonical_text(result) + "\n", encoding="utf-8"
    )
    return result


def inject_timeout(root: Path) -> dict[str, Any]:
    return _raw_server(root, "timeout", None)


def inject_reply(root: Path) -> dict[str, Any]:
    return _raw_server(root, "reply", b'{"ok":true,"node_id":"fault"}\n')


def run_injection(kind: str, root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root or Path.cwd())
    functions = {
        "spec": inject_spec,
        "spec-json": inject_spec_json,
        "lease": inject_stale_lease,
        "snapshot": inject_snapshot_digest,
        "verify": inject_verify,
        "port": inject_port,
        "timeout": inject_timeout,
        "reply": inject_reply,
    }
    if kind not in functions:
        raise ValueError(f"unknown injection kind: {kind}")
    return functions[kind](root)

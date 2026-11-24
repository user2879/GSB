from __future__ import annotations

import socket
import time
from pathlib import Path

import pytest

from probelease import LeaseOrchestrator, load_declarations


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples" / "config"


@pytest.fixture(scope="session")
def declarations():
    return load_declarations(
        SAMPLES / "sessions.json",
        SAMPLES / "probes.json",
        SAMPLES / "ack_policy.json",
    )


@pytest.fixture(autouse=True)
def wait_for_sample_ports(declarations):
    wait_ports_free(declarations)
    yield
    wait_ports_free(declarations)


@pytest.fixture
def orchestrator(declarations, tmp_path):
    instance = LeaseOrchestrator(declarations, tmp_path / "ledger.sqlite3")
    instance.start_probes()
    try:
        yield instance
    finally:
        instance.close()
        wait_ports_free(declarations)


def wait_ports_free(declarations, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    for session in declarations.sessions:
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.bind(("127.0.0.1", session.port))
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"UDP port {session.port} is still occupied")
                time.sleep(0.01)
            finally:
                sock.close()


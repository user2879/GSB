from __future__ import annotations

import json
import shutil
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any

from meshdiag.canonical import pretty_json
from mock_upstream import start_servers


ROOT = Path(__file__).resolve().parents[1]


class LocalUpstream:
    def __init__(self, count: int = 2) -> None:
        self.servers = start_servers([0] * count)
        self.threads = []
        for server in self.servers:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.threads.append(thread)
        self.ports = [int(server.server_address[1]) for server in self.servers]

    def reset(self) -> None:
        for port in self.ports:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/__admin/reset", timeout=2
            ) as response:
                response.read()

    def state(self, port: int, path: str) -> list[dict[str, Any]]:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/__admin/state", timeout=2
        ) as response:
            rows = json.loads(response.read().decode("utf-8"))
        return [row for row in rows if row["port"] == port and row["path"] == path]

    def close(self) -> None:
        for server in self.servers:
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=2)


class TempWorkspace:
    def __init__(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="meshdiag-test-"))
        self.path = self.directory / "workspace.json"

    def cleanup(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.directory / name
        path.write_text(pretty_json(payload) + "\n", encoding="utf-8")
        return path

    def create(
        self,
        *,
        services: list[dict[str, Any]],
        revision: dict[str, Any],
        requests: list[dict[str, Any]],
        global_concurrency: int = 2,
        defaults: dict[str, Any] | None = None,
        revision_name: str = "revision.json",
        requests_name: str = "requests.json",
    ) -> Path:
        revision_path = self.write_json(revision_name, revision)
        requests_path = self.write_json(requests_name, {"requests": requests})
        payload = {
            "workspace_id": "test-workspace",
            "global_concurrency": global_concurrency,
            "revision_file": revision_path.name,
            "requests_file": requests_path.name,
            "ledger_path": "var/ledger.sqlite3",
            "snapshot_dir": "var/snapshots",
            "defaults": defaults
            or {"timeout_ms": 500, "retries": 0, "cooldown_ms": 1000},
            "services": services,
        }
        self.path.write_text(pretty_json(payload) + "\n", encoding="utf-8")
        return self.path


def service_rows(ports: list[int], *, limit: int = 1, service_id: str = "svc") -> list[dict[str, Any]]:
    return [
        {
            "id": service_id,
            "max_concurrency": limit,
            "instances": [
                {
                    "id": f"{service_id}-{index}",
                    "host": "127.0.0.1",
                    "port": port,
                }
                for index, port in enumerate(ports)
            ],
        }
    ]


def v3_rule(
    *,
    rule_id: str = "route",
    target: str = "svc",
    prefix: str = "/svc",
    weights: dict[str, int] | None = None,
    timeout_ms: int = 100,
    retries: int = 0,
    threshold: int = 2,
    cooldown_ms: int = 5000,
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": rule_id,
        "format_version": 3,
        "prefix": prefix,
        "target_service": target,
        "instance_weights": weights or {"svc-0": 1},
        "timeout_ms": timeout_ms,
        "retries": retries,
        "circuit_threshold": threshold,
        "cooldown_ms": cooldown_ms,
        "depends_on": depends_on or [],
    }

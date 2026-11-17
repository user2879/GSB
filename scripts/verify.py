from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster_diag.runtime import DiagnosticRuntime  # noqa: E402
from cluster_diag.spec import load_spec  # noqa: E402


def _free_ports(count: int) -> list[int]:
    ports: list[int] = []
    sockets: list[socket.socket] = []
    try:
        while len(ports) < count:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
            if port >= 1024 and port not in ports:
                ports.append(port)
                sockets.append(sock)
            else:
                sock.close()
    finally:
        for sock in sockets:
            sock.close()
    return ports


def _versions() -> dict[str, Any]:
    import numpy

    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
    }
    try:
        import tkinter

        result["tkinter"] = str(tkinter.TkVersion)
    except Exception as exc:
        result["tkinter"] = f"unavailable:{type(exc).__name__}"
    return result


def _run_tests() -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-v"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = process.stdout
    ran_line = next((line for line in reversed(output.splitlines()) if line.startswith("Ran ")), "")
    return {
        "ok": process.returncode == 0,
        "command": "python -m unittest discover -v",
        "exit_code": process.returncode,
        "summary": ran_line,
        "output_tail": output.splitlines()[-40:],
    }


def _acceptance() -> dict[str, Any]:
    example = json.loads((ROOT / "examples" / "spec.json").read_text(encoding="utf-8"))
    for cluster, port in zip(example["clusters"], _free_ports(len(example["clusters"]))):
        cluster["port"] = port
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        spec_path = temp / "spec.json"
        spec_path.write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")
        spec = load_spec(spec_path)
        with DiagnosticRuntime(spec=spec, workdir=temp / "runtime", latency_budget_ms=50) as runtime:
            commit1 = runtime.commit_next()
            runtime.query("edge-a")
            commit2 = runtime.commit_next()
            commit3 = runtime.commit_next()
            query = runtime.query("edge-a")
            authoritative = runtime.authoritative()
            recovery = runtime.recover(replay=True)
            snapshot = json.loads((temp / "runtime" / "session_snapshot.json").read_text(encoding="utf-8"))
            switches = []
            for switch in (commit2["root_switch"], commit3["root_switch"]):
                switches.append(
                    {
                        "old_root_revision": switch["old_root_revision"],
                        "new_root_revision": switch["new_root_revision"],
                        "affected_clusters": switch["affected_clusters"],
                        "changed_clusters": [
                            {
                                "cluster_id": change["cluster_id"],
                                "old_revision": change["old_revision"],
                                "old_summary_digest": change["old_summary_digest"],
                                "new_revision": change["new_revision"],
                                "new_summary_digest": change["new_summary_digest"],
                            }
                            for change in switch["changed_clusters"]
                        ],
                        "stale_nodes": switch["stale_nodes"],
                        "incremental_rows_touched": switch["incremental_rows_touched"],
                        "policy": switch["policy"],
                    }
                )
            return {
                "ok": True,
                "roots": [commit1["root_revision"], commit2["root_revision"], commit3["root_revision"]],
                "root_switches": switches,
                "effective_nodes": {
                    item["cluster_id"]: item["node_revision"] for item in runtime.current_items()
                },
                "query": {
                    "cluster_id": query["cluster_id"],
                    "revision": query["revision"],
                    "elapsed_ms_x1000": query["latency"]["elapsed_ms_x1000"],
                    "exceeded": query["latency"]["exceeded"],
                },
                "authoritative": {
                    "root_revision": authoritative["root_revision"],
                    "recomputed_node_count": authoritative["recomputed_node_count"],
                },
                "recovery": {
                    "snapshot_replaced": recovery["snapshot_replaced"],
                    "replayed_node_count": recovery["replay"]["replayed_node_count"],
                    "unknown_uncommitted_nodes": recovery["unknown_uncommitted_nodes"],
                },
                "snapshot": {
                    "root_revision": snapshot["root_revision"],
                    "node_digest_count": len(snapshot["node_digests"]),
                    "latency_schema_version": snapshot["latency_schema_version"],
                    "quantization_error_upper_bound_ms_x1000": snapshot["latency_stats"][
                        "quantization_error_upper_bound_ms_x1000"
                    ],
                },
            }


def main() -> int:
    tests = _run_tests()
    acceptance: dict[str, Any]
    try:
        acceptance = _acceptance()
    except Exception as exc:
        acceptance = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    document = {
        "schema": "cluster-diag-verification-v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": bool(tests["ok"] and acceptance.get("ok")),
        "environment": _versions(),
        "tests": tests,
        "acceptance": acceptance,
        "declared_clock_basis": "time.monotonic_ns",
        "declared_truth_source": "sqlite:latency_events",
    }
    output_path = ROOT / "artifacts" / "verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": document["ok"], "artifact": str(output_path)}, ensure_ascii=False))
    return 0 if document["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

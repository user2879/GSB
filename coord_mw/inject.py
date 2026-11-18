"""Reproducible fault injection for dependency, snapshot and port failures."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from .engine import CoordinatorEngine
from .errors import CoordError
from .util import JsonlLogger, atomic_write_json, canonical_json, read_json


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _result(case: str, expected: str, observed: str, **details: Any) -> dict[str, Any]:
    return {
        "case": case,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
        **details,
    }


def inject_dependency_cycle(base: Path, logger: JsonlLogger) -> dict[str, Any]:
    root = base / "dependency_cycle"
    reset_directory(root)
    source = read_json(PROJECT_ROOT / "sample_cycle.json")
    input_path = root / "injected_members.json"
    atomic_write_json(input_path, source)
    root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    observed = None
    details: dict[str, Any] = {"input": str(input_path)}
    try:
        CoordinatorEngine(root / "engine", definition=read_json(input_path))
    except CoordError as exc:
        observed = exc.code
        details.update(exc.details)
        details["message"] = exc.message
    result = _result("dependency_cycle", "E_DEPEND", observed or "NONE", **details)
    logger.write("injection_result", **result)
    return result


def inject_snapshot_corruption(base: Path, logger: JsonlLogger) -> dict[str, Any]:
    root = base / "snapshot_corruption"
    reset_directory(root)
    engine = CoordinatorEngine(
        root / "engine",
        definition=read_json(PROJECT_ROOT / "sample_members.json"),
    )
    engine.heartbeat(1, "A", 0)
    engine.vote(1, "A", "A")
    snapshot = engine.create_snapshot("snapshots/corrupt_me.json")
    target = Path(snapshot["path"])
    target.write_text("{this is deliberately broken JSON\n", encoding="utf-8")
    observed = None
    details: dict[str, Any] = {"snapshot": str(target)}
    try:
        engine.restore_snapshot(target)
    except CoordError as exc:
        observed = exc.code
        details.update(exc.details)
        details["message"] = exc.message
    result = _result("snapshot_corruption", "E_SNAPSHOT", observed or "NONE", **details)
    logger.write("injection_result", **result)
    return result


def inject_port_conflict(base: Path, logger: JsonlLogger) -> dict[str, Any]:
    root = base / "port_conflict"
    reset_directory(root)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    observed = None
    details: dict[str, Any] = {}
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        details["port"] = port
        command = [
            sys.executable,
            "-m",
            "coord_mw.server",
            "--root",
            str(root / "server"),
            "--port",
            str(port),
            "--members",
            str(PROJECT_ROOT / "sample_members.json"),
        ]
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        details["exit_code"] = completed.returncode
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        response = json.loads(lines[-1]) if lines else {}
        observed = response.get("code")
        details["message"] = response.get("message")
        details["server_log"] = str(root / "server" / "logs" / "server.jsonl")
    finally:
        sock.close()
    result = _result("port_conflict", "E_PORT", observed or "NONE", **details)
    logger.write("injection_result", **result)
    return result


def run_all(base: Path) -> list[dict[str, Any]]:
    base.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(base / "injections.jsonl")
    return [
        inject_dependency_cycle(base, logger),
        inject_snapshot_corruption(base, logger),
        inject_port_conflict(base, logger),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inject deterministic coordination failures")
    parser.add_argument(
        "--case",
        choices=["cycle", "snapshot", "port", "all"],
        default="all",
    )
    parser.add_argument("--base", default="injections")
    args = parser.parse_args(argv)
    base = Path(args.base)
    base.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(base / "injections.jsonl")
    if args.case == "cycle":
        results = [inject_dependency_cycle(base, logger)]
    elif args.case == "snapshot":
        results = [inject_snapshot_corruption(base, logger)]
    elif args.case == "port":
        results = [inject_port_conflict(base, logger)]
    else:
        results = run_all(base)
    summary = {"ok": all(item["passed"] for item in results), "results": results}
    atomic_write_json(base / "summary.json", summary)
    logger.write("injection_summary", **summary)
    print(canonical_json(summary), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())


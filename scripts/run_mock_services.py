"""Start every deterministic mock service declared by a scene file.

Usage:
    python scripts/run_mock_services.py samples/scene_cold_start.json

The process stays in the foreground until Ctrl+C / SIGTERM.  A readiness line
``MOCK_READY <port>`` is printed per bound port so wrapper scripts can wait.
"""
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fanout_ledger.config import load_scene
from fanout_ledger.errors import LedgerError, PortError
from fanout_ledger.logging_util import StructuredLogger
from fanout_ledger.mock_service import start_all, stop_all


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_mock_services.py <scene.json>", file=sys.stderr)
        return 2
    logger = StructuredLogger(Path("run") / "mock_services.jsonl", echo=False)
    try:
        scene = load_scene(sys.argv[1])
        handles = start_all(scene.nodes)
    except PortError as exc:
        logger.error("mock_start_failed", **exc.to_dict())
        print(f"E_PORT {exc.message} {exc.details}", file=sys.stderr)
        return 3
    except LedgerError as exc:
        logger.error("mock_start_failed", **exc.to_dict())
        print(f"{exc.code} {exc.message}", file=sys.stderr)
        return 3

    for handle in handles:
        print(f"MOCK_READY {handle.node.port} {handle.endpoint}", flush=True)
    logger.info("mock_services_started",
                nodes=[n.node_id for n in scene.nodes],
                ports=[n.port for n in scene.nodes])

    stopping = {"flag": False}

    def _shutdown(signum, frame):  # noqa: ANN001
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    try:
        while not stopping["flag"]:
            time.sleep(0.2)
    finally:
        stop_all(handles)
        logger.info("mock_services_stopped",
                    nodes=[n.node_id for n in scene.nodes])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Executable node process entry point.

Usage: ``python -m shardflow.node <config.json>``. The orchestrator writes a
small node configuration JSON (node declaration plus the currently effective
integer configuration mapping) and starts this module as a real child process.
"""

import json
import signal
import sys

from .protocol import NodeServer
from .spec import Node


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        print("usage: python -m shardflow.node <node-config.json>", file=sys.stderr)
        return 2
    with open(argv[0], "r", encoding="utf-8") as handle:
        config = json.load(handle)
    node = Node(config["node"])
    server = NodeServer(node, config.get("effective_config", {}))
    server.start()

    def _shutdown(_signum=None, _frame=None):
        server.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    try:
        while not server._stop.wait(0.2):
            pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

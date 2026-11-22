"""独立确定性模拟服务入口（也可由引擎在进程内托管）。

用法:
    python scripts/run_mock_service.py --topology samples/topology.json \
        --port-map artifacts/port_map.json
所有套接字仅绑定 127.0.0.1，不访问外网。
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from degprop.loader import load_topology  # noqa: E402
from degprop.mock_service import MockCluster  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="samples/topology.json")
    parser.add_argument("--port-map", default="artifacts/port_map.json")
    parser.add_argument("--slow-ms", type=int, default=400)
    args = parser.parse_args()

    topology = load_topology(args.topology)
    cluster = MockCluster(topology, slow_ms=args.slow_ms)
    bindings = cluster.start()
    Path(args.port_map).parent.mkdir(parents=True, exist_ok=True)
    cluster.write_port_map(args.port_map)
    print(f"mock cluster ready, port map -> {args.port_map}")

    def _shutdown(*_):
        cluster.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)
    while True:
        time.sleep(0.5)


if __name__ == "__main__":
    raise SystemExit(main())

"""Run every deterministic simulated upstream defined in a rule file.

Usage:  python scripts/run_upstream.py --rules samples/rules_v1.json
Only 127.0.0.1 high ports are used; the script aborts if any port is busy.
"""

import argparse
import signal
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meshbench.config import load_config
from meshbench.errors import UpstreamError
from meshbench.netutil import assert_ports_free
from meshbench.upstream import start_all, stop_all


def main():
    parser = argparse.ArgumentParser(description="Deterministic mock upstreams")
    parser.add_argument("--rules", default="samples/rules_v1.json")
    args = parser.parse_args()

    config = load_config(args.rules)
    ports = sorted({inst.port for service in config.services.values()
                    for inst in service.instances})
    try:
        assert_ports_free(ports)
    except UpstreamError as exc:
        print(f"[abort] {exc.message} busy={exc.context.get('busy_ports')}", file=sys.stderr)
        return 3

    upstreams = start_all(config)
    print(f"mock upstreams listening on 127.0.0.1: {ports}")

    stopping = {"flag": False}

    def _shutdown(_signum, _frame):
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    try:
        while not stopping["flag"]:
            signal.pause()
    except AttributeError:  # Windows has no signal.pause
        import time
        while not stopping["flag"]:
            time.sleep(0.2)
    finally:
        stop_all(upstreams)
        print("mock upstreams stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

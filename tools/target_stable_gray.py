"""Two loopback targets for stable (19081) and gray (19082) traffic."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.target_common import make_server, serve_forever


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--stable-port", type=int, default=19081)
    parser.add_argument("--gray-port", type=int, default=19082)
    parser.add_argument("--delay-ms", type=int, default=20)
    args = parser.parse_args()
    servers = [
        make_server(args.host, args.stable_port, "stable", args.delay_ms),
        make_server(args.host, args.gray_port, "gray", args.delay_ms),
    ]
    try:
        serve_forever(servers)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

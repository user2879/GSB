"""Independent loopback mirror target (19083) for sampled traffic."""

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
    parser.add_argument("--port", type=int, default=19083)
    parser.add_argument("--delay-ms", type=int, default=40)
    args = parser.parse_args()
    server = make_server(args.host, args.port, "mirror", args.delay_ms)
    try:
        serve_forever([server])
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

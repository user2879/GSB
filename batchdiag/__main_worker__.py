"""Entry point: ``python -m batchdiag.__main_worker__ --port N ...``"""

import argparse

from .worker import ensure_port_free, serve_once


def main():
    parser = argparse.ArgumentParser(description="batchdiag loopback worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--stall-ms", type=int, default=0)
    parser.add_argument("--exit-after", type=int, default=1)
    args = parser.parse_args()
    ensure_port_free(args.host, args.port)
    serve_once(args.host, args.port, stall_ms=args.stall_ms,
               exit_after=args.exit_after)


if __name__ == "__main__":
    main()

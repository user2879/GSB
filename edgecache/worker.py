"""Subprocess entry point for a content source or an edge node."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Sequence

from .errors import EdgeError
from .node_service import EdgeNode
from .protocol import JsonLineServer
from .source_service import ContentSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m edgecache.worker")
    subparsers = parser.add_subparsers(dest="role", required=True)

    source = subparsers.add_parser("source")
    source.add_argument("--spec", required=True)
    source.add_argument("--host", default="127.0.0.1")
    source.add_argument("--port", required=True, type=int)

    node = subparsers.add_parser("node")
    node.add_argument("--node-id", required=True)
    node.add_argument("--source-host", default="127.0.0.1")
    node.add_argument("--source-port", required=True, type=int)
    node.add_argument("--port", required=True, type=int)
    node.add_argument("--host", default="127.0.0.1")
    node.add_argument("--timeout", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server: JsonLineServer | None = None
    try:
        if args.role == "source":
            service = ContentSource(args.spec)
            server = JsonLineServer(args.host, args.port, "content-source", service.handle)
        else:
            service = EdgeNode(
                node_id=args.node_id,
                source_host=args.source_host,
                source_port=args.source_port,
                timeout=args.timeout,
            )
            server = JsonLineServer(args.host, args.port, "edge-node", service.handle)
        server.bind()
        print(
            json.dumps(
                {
                    "ready": True,
                    "role": args.role,
                    "host": args.host,
                    "port": args.port,
                    "pid": os.getpid(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        server.serve_forever()
        return 0
    except EdgeError as exc:
        print(json.dumps(exc.envelope(), sort_keys=True, separators=(",", ":")), file=sys.stderr, flush=True)
        return 2
    except Exception as exc:  # pragma: no cover - process boundary safety
        print(
            json.dumps(
                EdgeError.from_exception(exc).envelope(),
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        return 3
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    raise SystemExit(main())

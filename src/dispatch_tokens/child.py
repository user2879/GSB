"""子进程入口：``python -m dispatch_tokens.child <node_json> <scale>``。"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict

from .spec import NodeSpec
from .worker import worker_main


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        sys.stderr.write("usage: python -m dispatch_tokens.child <node_json> <scale>\n")
        return 64
    node_data = json.loads(argv[0])
    scale = int(argv[1])
    node = NodeSpec(**node_data)
    return worker_main(node, scale)


if __name__ == "__main__":
    raise SystemExit(main())

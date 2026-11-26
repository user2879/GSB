"""CLI entry point used when spawning one worker subprocess."""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_base_config
from .errors import QLabError
from .worker import serve_worker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one quantledger loopback worker")
    parser.add_argument("--config", required=True, help="base JSON configuration")
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args(argv)
    loaded = load_base_config(args.config)
    worker = loaded.state.find_worker(args.worker_id)
    if worker is None:
        print(json.dumps({"ok": False, "code": "E_UNKNOWN_WORKER", "message": args.worker_id}), file=sys.stderr)
        return 2
    try:
        serve_worker(worker)
    except OSError as exc:
        print(json.dumps({"ok": False, "code": "E_PORT_IN_USE", "message": str(exc)}), file=sys.stderr)
        return 3
    except QLabError as exc:
        print(json.dumps(exc.as_dict()), file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

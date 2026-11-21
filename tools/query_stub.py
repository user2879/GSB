"""Run the deterministic producer query endpoint as a standalone process."""

from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from txn_mw.scenario import QueryStubServer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", default=str(ROOT / "samples" / "query_answers.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file")
    args = parser.parse_args()
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    stub = QueryStubServer(answers, host=args.host, port=args.port)
    info = {"ok": True, "host": args.host, "port": stub.port}
    if args.port_file:
        Path(args.port_file).write_text(json.dumps(info) + "\n", encoding="utf-8")
    print(json.dumps(info), flush=True)
    stop = threading.Event()

    def stop_handler(signum, frame):
        stop.set()

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                signal.signal(sig, stop_handler)
            except (ValueError, OSError):
                pass
    try:
        while not stop.wait(0.25):
            pass
    finally:
        stub.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


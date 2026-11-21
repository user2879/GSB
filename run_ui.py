from __future__ import annotations

import argparse

from streamagg.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="artifacts/demo/ledger.sqlite3")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--view-id", default=None)
    args = parser.parse_args()
    command = ["ui", "--ledger", args.ledger]
    if args.session_id:
        command.extend(["--session-id", args.session_id])
    if args.view_id:
        command.extend(["--view-id", args.view_id])
    raise SystemExit(main(command))

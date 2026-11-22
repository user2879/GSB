"""Command line interface.
Usage:
  python -m colval.cli init-sample --base DIR
  python -m colval.cli run        --base DIR [--crash-after N]
  python -m colval.cli verify     --base DIR
  python -m colval.cli gui        --base DIR
All paths stay inside the workspace; no network access is required.
"""

import argparse
import json
import os
import sys

from .errors import ColValError
from .formats import decode_block
from .runner import SessionRunner
from .sample_data import init_sample_workspace
from .store import BlockStore
from .verify import verify_store_against_canonical


def _cmd_init_sample(args):
    paths = init_sample_workspace(args.base)
    print(json.dumps({"ok": True, "paths": paths}, indent=1))


def _cmd_run(args):
    base = args.base
    runner = SessionRunner(
        data_dir=os.path.join(base, "data"),
        tasks_path=os.path.join(base, "tasks", "expressions.json"),
        work_dir=os.path.join(base, "work"))
    try:
        report = runner.run(crash_after=args.crash_after)
    finally:
        runner.close()
    print(json.dumps(report, indent=1, sort_keys=True))


def _cmd_verify(args):
    base = args.base
    with open(os.path.join(base, "canonical.json"), "r",
              encoding="utf-8") as fh:
        canonical = {bid: decode_block(payload)
                     for bid, payload in json.load(fh).items()}
    store = BlockStore(os.path.join(base, "data"))
    try:
        verified = verify_store_against_canonical(store, canonical)
        backups = store.list_backups()
    finally:
        store.close()
    print(json.dumps({"ok": True, "verified_blocks": verified,
                      "backups": backups}, indent=1))


def _cmd_gui(args):
    from .gui import run_gui
    run_gui(args.base)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="colval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("init-sample", _cmd_init_sample),
                     ("run", _cmd_run),
                     ("verify", _cmd_verify),
                     ("gui", _cmd_gui)):
        p = sub.add_parser(name)
        p.add_argument("--base", default=".")
        if name == "run":
            p.add_argument("--crash-after", type=int, default=None)
        p.set_defaults(fn=fn)
    args = parser.parse_args(argv)
    try:
        args.fn(args)
    except ColValError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

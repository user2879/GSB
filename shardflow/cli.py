"""Command line interface: validate, run, verify, inspect, diagnose."""

import argparse
import json
import sys

from .engine import Engine
from .errors import ShardflowError
from .ledger import Ledger
from .replay import verify_ledger
from .spec import load_spec
from .viewmodel import build_view_model


def _print(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_validate(args):
    spec = load_spec(args.spec, args.events)
    _print(
        {
            "status": "ok",
            "nodes": spec.node_order,
            "shards": spec.shard_order,
            "topological_order": spec.topo_order,
            "revisions": spec.revision_order,
            "events": len(spec.events),
        }
    )
    return 0


def cmd_run(args):
    spec = load_spec(args.spec, args.events)
    engine = Engine(spec, args.ledger)
    snapshot = engine.run()
    _print(snapshot)
    return 0


def cmd_verify(args):
    result = verify_ledger(args.spec, args.ledger, args.events)
    _print(result)
    return 0


def cmd_inspect(args):
    ledger = Ledger(args.ledger, replace_on_mismatch=False)
    payload = {
        "format_version": ledger.format_version(),
        "content_digest": ledger.content_digest(),
        "progress_cursor": ledger.progress_cursor(),
        "lease_events": ledger.fetch_lease_events(),
        "outcomes": ledger.fetch_outcomes(),
        "invalidations": ledger.fetch_invalidations(),
        "snapshots": ledger.fetch_snapshots(),
    }
    ledger.close()
    _print(payload)
    return 0


def cmd_diagnose(args):
    # Imported lazily so headless CI without Tk can still use other commands.
    from .diagnostics import launch_tk
    return launch_tk(args.spec, args.ledger, args.events)


def build_parser():
    parser = argparse.ArgumentParser(prog="shardflow")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate declaration files")
    validate.add_argument("spec")
    validate.add_argument("--events", default=None)
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="run the orchestration engine")
    run.add_argument("spec")
    run.add_argument("--events", default=None)
    run.add_argument("--ledger", required=True)
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="independent ledger replay")
    verify.add_argument("spec")
    verify.add_argument("--events", default=None)
    verify.add_argument("--ledger", required=True)
    verify.set_defaults(func=cmd_verify)

    inspect = sub.add_parser("inspect", help="dump ledger contents read-only")
    inspect.add_argument("--ledger", required=True)
    inspect.set_defaults(func=cmd_inspect)

    diagnose = sub.add_parser("diagnose", help="open Tkinter diagnostics")
    diagnose.add_argument("spec")
    diagnose.add_argument("--events", default=None)
    diagnose.add_argument("--ledger", required=True)
    diagnose.set_defaults(func=cmd_diagnose)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ShardflowError as exc:
        _print({"status": "error", **exc.to_dict()})
        return 1


if __name__ == "__main__":
    sys.exit(main())

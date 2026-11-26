"""Command line entry points for the GSB pipeline."""

from __future__ import annotations

import argparse
import json

from .batching import plan_batches
from .bus import Bus
from .client import ProcessorPool
from .gen_sample import generate
from .ledger import Ledger, diff_replay
from .spec import load_spec


def _open_bus(args):
    spec = load_spec(args.spec, args.events)
    return spec, Bus(spec, args.run_dir, crash_after=args.crash_after)


def cmd_gen(args):
    spec_path, events_path = generate(args.out, n_events=args.events)
    print(f"sample written: {spec_path} / {events_path}")


def cmd_run(args):
    spec, bus = _open_bus(args)
    bus.write_manifest()
    with ProcessorPool(spec, processor_dir=f"{args.run_dir}/processors") as pool:
        bus.run(pool)
        if args.demo_stale_renew:
            first_batch = bus.batches[0]
            finding = bus.stale_renewal_demo(pool, first_batch.batch_id)
            print("stale renewal rejected:", finding["code"])
    bus.write_outputs()
    bus.close()
    print(f"committed {len(bus.adoptions)} batch adoptions to {args.run_dir}")


def cmd_recover(args):
    # Reopening only relies on committed rows; nothing uncommitted is trusted.
    spec = load_spec(args.spec, args.events)
    ledger = Ledger(f"{args.run_dir}/ledger.sqlite3", spec.spec_digest)
    batches = plan_batches(spec)
    records = [ledger.lookup_dispatch(batch.batch_id) for batch in batches]
    records = [record for record in records if record]
    committed = ledger.committed_batch_ids()
    unknown = [batch.batch_id for batch in batches if batch.batch_id not in committed]
    finding = diff_replay(batches, records, ledger)
    report = {
        "recovered_committed_batches": len(committed),
        "unknown_uncommitted_batches": unknown,
        "rerun_required": bool(unknown),
        "first_mismatch": finding,
        "snapshot_replaced": ledger.replaced,
    }
    ledger.close()
    print(json.dumps(report, sort_keys=True, indent=2))
    with open(f"{args.run_dir}/recovery.json", "w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True, indent=2))


def cmd_verify(args):
    spec = load_spec(args.spec, args.events)
    batches = plan_batches(spec)
    ledger = Ledger(f"{args.run_dir}/ledger.sqlite3", spec.spec_digest)
    records = [ledger.lookup_dispatch(batch.batch_id) for batch in batches]
    finding = diff_replay(batches, [r for r in records if r], ledger)
    ledger.close()
    if finding:
        print(json.dumps(finding, sort_keys=True, indent=2))
        raise SystemExit(1)
    print(f"OK: {len(batches)} planned batches match committed ledger records")


def build_parser():
    parser = argparse.ArgumentParser(prog="gsb")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen")
    gen.add_argument("--out", default="samples/demo")
    gen.add_argument("--events", type=int, default=48)
    gen.set_defaults(func=cmd_gen)

    run = sub.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--events", default=None)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--crash-after", type=int, default=None)
    run.add_argument("--demo-stale-renew", action="store_true")
    run.set_defaults(func=cmd_run)

    recover = sub.add_parser("recover")
    recover.add_argument("--spec", required=True)
    recover.add_argument("--events", default=None)
    recover.add_argument("--run-dir", required=True)
    recover.set_defaults(func=cmd_recover)

    verify = sub.add_parser("verify")
    verify.add_argument("--spec", required=True)
    verify.add_argument("--events", default=None)
    verify.add_argument("--run-dir", required=True)
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

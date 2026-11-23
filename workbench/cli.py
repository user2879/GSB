"""Command line interface for the diagnostic workbench.

Subcommands:
  run            load spec, start upstreams, execute script in batches
  view           GUI-path quick view from the incremental cache (one read)
  recompute      authoritative replay from the ledger + dual-path check
  snapshot-save  persist a session snapshot
  snapshot-open  open a snapshot (whole-replace on mismatch) + replay check
  recover        scan ledger for uncommitted batches, mark UNKNOWN
  gui            launch the tkinter workbench UI
"""

import argparse
import json
import sys

from . import E_SPEC
from .engine import Engine, IncrementalCache
from .errors import DiagError
from .ledger import Ledger
from .replay import (authoritative_recompute,
                     compare_cache_with_authoritative)
from .snapshot import (build_snapshot, restore_snapshot, save_snapshot,
                       verify_replay_after_restore)
from .spec import load_spec
from .stats import aggregate_by_scope
from .upstream_mgr import UpstreamManager


def _print_json(obj):
    print(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False))


def _open_ledger(args, spec=None):
    return Ledger(args.ledger, scale=args.scale,
                  services_digest=spec.services_digest if spec else None,
                  keys_digest=spec.keys_digest if spec else None)


def cmd_run(args):
    spec = load_spec(args.data_dir)
    ledger = _open_ledger(args, spec)
    unknown = ledger.recover_uncommitted()
    if unknown:
        print(f"recovered: {len(unknown)} request(s) marked UNKNOWN, "
              f"re-run required: seqs={unknown}")
    engine = Engine(spec, ledger, global_limit=args.global_limit,
                    per_service_quota=args.per_service_quota,
                    upstream_timeout_s=args.upstream_timeout)
    with UpstreamManager(spec) as mgr:
        mgr.start()
        rows = engine.run_all(batch_size=args.batch_size)
    counts = {}
    for row in rows:
        counts[row["decision_code"]] = counts.get(row["decision_code"], 0) + 1
    print(f"executed {len(rows)} requests in batches of {args.batch_size}")
    _print_json({"decision_counts": dict(sorted(counts.items())),
                 "ledger": args.ledger,
                 "cursor": ledger.cursor,
                 "content_digest": ledger.meta()["content_digest"]})
    ledger.close()
    return 0


def cmd_view(args):
    spec = load_spec(args.data_dir)
    ledger = _open_ledger(args, spec)
    cache = IncrementalCache()
    cache.update([dict(r) for r in ledger.done_requests()])
    engine = Engine(spec, ledger)
    engine.cache = cache
    view = engine.quick_view()
    _print_json(view)
    print("gui_query observation recorded; stats:")
    _print_json({"gui_query": aggregate_by_scope(
        ledger.observations("gui_query"))})
    ledger.close()
    return 0


def cmd_recompute(args):
    spec = load_spec(args.data_dir)
    ledger = _open_ledger(args, spec)
    cache = IncrementalCache()
    cache.update([dict(r) for r in ledger.done_requests()])
    cache_view = cache.snapshot()
    with UpstreamManager(spec) as mgr:
        mgr.start()
        result = authoritative_recompute(spec, ledger)
    if not result.consistent:
        print(f"RECOMPUTE DIVERGENCE: {result.divergence}")
        ledger.close()
        return 2
    dual = compare_cache_with_authoritative(cache_view, result)
    if dual is not None:
        print(f"DUAL-PATH DIVERGENCE: {dual}")
        ledger.close()
        return 3
    print(f"authoritative recompute OK: {len(result.conclusions)} "
          f"requests replayed, cache view byte-identical "
          f"({result.elapsed_ms} ms)")
    _print_json(result.stats)
    ledger.close()
    return 0


def cmd_snapshot_save(args):
    spec = load_spec(args.data_dir)
    ledger = _open_ledger(args, spec)
    snap = build_snapshot(spec, ledger)
    save_snapshot(args.snapshot, snap)
    print(f"snapshot saved: {args.snapshot} "
          f"(cursor={snap['ledger_cursor']}, "
          f"conclusions={len(snap['conclusions'])})")
    ledger.close()
    return 0


def cmd_snapshot_open(args):
    spec = load_spec(args.data_dir)
    ledger = _open_ledger(args, spec)
    snap, diffs, replaced = restore_snapshot(args.snapshot, spec,
                                             ledger.scale)
    if replaced:
        print("snapshot REPLACED (whole replace, no incremental merge); "
              "differing items:")
        for item in diffs:
            print(f"  - {item}")
    else:
        print(f"snapshot accepted (cursor={snap['ledger_cursor']})")
    try:
        verified = verify_replay_after_restore(spec, ledger)
    except DiagError as exc:
        print(f"replay verification FAILED: {exc}")
        ledger.close()
        return 4
    print(f"replay verification OK: {verified} requests match ledger")
    ledger.close()
    return 0


def cmd_recover(args):
    ledger = Ledger(args.ledger)
    unknown = ledger.recover_uncommitted()
    if unknown:
        print(f"marked UNKNOWN (re-run required): seqs={unknown}")
    else:
        print("no uncommitted batches found")
    ledger.close()
    return 0


def cmd_gui(args):
    try:
        from .gui import launch
    except ImportError as exc:
        print(f"tkinter unavailable on this host ({exc}); the GUI requires "
              f"Windows 10+ with Python 3.11 (tkinter included).",
              file=sys.stderr)
        return 6
    launch(args.data_dir, args.ledger, args.snapshot, args.scale,
           args.batch_size, args.global_limit, args.per_service_quota)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="workbench")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--ledger", default="out/ledger.db")
    parser.add_argument("--snapshot", default="out/snapshot.json")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--global-limit", type=int, default=4)
    parser.add_argument("--per-service-quota", type=int, default=2)
    parser.add_argument("--upstream-timeout", type=float, default=2.0)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("run", cmd_run), ("view", cmd_view),
                     ("recompute", cmd_recompute),
                     ("snapshot-save", cmd_snapshot_save),
                     ("snapshot-open", cmd_snapshot_open),
                     ("recover", cmd_recover), ("gui", cmd_gui)):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DiagError as exc:
        print(str(exc), file=sys.stderr)
        return 1 if exc.code == E_SPEC else 5


if __name__ == "__main__":
    sys.exit(main())

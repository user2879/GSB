"""Command line interface: run / recover-check / diverge-check / verify."""

import argparse
import json
import os
import sys

from . import faults
from .canon import digest_obj
from .catalog import Catalog, param_set_fingerprint
from .engine import Engine
from .errors import PipelineError
from .persistence import (StateLogWriter, read_snapshot, write_snapshot)
from .recovery import (compare_states, rebuild_from_snapshot, replay_log,
                       state_digest)
from .schema import load_queries_file, load_tables_file
from .state import EngineState
from .structured_log import StructuredLogger
from .verify import (check_agg_results, check_window_results, compare_runs)

REPLACED_STATE_FIELDS = ["cursors", "applied", "aggs", "windows"]


def _prepare_workdir(workdir):
    for sub in ("", "snapshots", "statelog", "logs"):
        os.makedirs(os.path.join(workdir, sub), exist_ok=True)
    for sub in ("snapshots", "statelog", "logs"):
        dir_path = os.path.join(workdir, sub)
        for name in os.listdir(dir_path):
            os.unlink(os.path.join(dir_path, name))


def _load_inputs(tables_path, queries_path):
    tables = load_tables_file(tables_path)
    query_set = load_queries_file(queries_path, tables)
    return tables, query_set


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, sort_keys=True, indent=2)


def cmd_run(args):
    _prepare_workdir(args.workdir)
    logger = StructuredLogger(os.path.join(args.workdir, "logs",
                                           "structured.jsonl"))
    tables, query_set = _load_inputs(args.tables, args.queries)

    session = query_set.session
    if args.batch_size:
        session["batch_size"] = args.batch_size
    if args.shuffle_window:
        session["arrival"] = {"shuffle_seed": args.shuffle_seed,
                              "shuffle_window": args.shuffle_window}
    if args.inject == "batch-mid-window":
        session.update(faults.inject_batch_mid_window(session))
        logger.anomaly("E_BATCH", "injected: mid-window batch boundary",
                       batch_size=session["batch_size"])
    elif args.inject == "out-of-order":
        session.update(faults.inject_out_of_order(session))
        logger.anomaly("E_ORDER", "injected: out-of-order ratio overflow",
                       shuffle=session["arrival"])

    catalog = Catalog(args.catalog or
                      os.path.join(args.workdir, "catalog.db"))
    fingerprint = param_set_fingerprint(query_set)
    param_version = catalog.param_version(fingerprint)
    catalog.mark_stale_before(param_version)
    logger.log("INFO", "run_start", "pipeline run started",
               param_version=param_version, fingerprint=fingerprint)

    engine = Engine(tables, query_set, param_version, logger)
    resumed_from = None
    if args.resume_from:
        snap = read_snapshot(args.resume_from)  # E_SNAPSHOT on corruption
        if snap.query_param_version != param_version:
            logger.anomaly(
                "E_SNAPSHOT",
                "snapshot param version is stale; replacing whole state",
                snapshot_version=snap.query_param_version,
                current_version=param_version,
                replaced_fields=REPLACED_STATE_FIELDS)
            print(json.dumps({"code": "E_SNAPSHOT", "stale": True,
                              "replaced_fields": REPLACED_STATE_FIELDS},
                             ensure_ascii=False))
        else:
            engine.state = EngineState.from_dict(snap.state)
            resumed_from = snap.snapshot_id
            logger.log("INFO", "resume", "resumed from snapshot",
                       snapshot_id=snap.snapshot_id)

    writer = StateLogWriter(os.path.join(args.workdir, "statelog"),
                            format_version=args.log_format)
    total_rows = sum(t.row_count for t in tables.values())
    done_rows = sum(sum(end - start for start, end in iv)
                    for iv in engine.state.applied.values())
    snapshot_at_rows = int(total_rows * args.snapshot_at) \
        if args.snapshot_at else None
    snapshot_info = None

    def on_batch(record):
        nonlocal done_rows, snapshot_info
        writer.write_batch(record["table"], record["rows"],
                           record["cursor_after"])
        done_rows += len(record["rows"])
        if snapshot_at_rows and done_rows >= snapshot_at_rows \
                and snapshot_info is None:
            state_dict = engine.state.to_dict()
            writer.write_checkpoint(state_dict)
            snap_path = os.path.join(
                args.workdir, "snapshots",
                f"manifest-v{args.snapshot_format}-mid.json")
            body = write_snapshot(snap_path, state_dict, param_version,
                                  snapshot_id=f"snap-mid-{param_version}",
                                  format_version=args.snapshot_format)
            catalog.register_snapshot(body["snapshot_id"], snap_path,
                                      param_version, args.snapshot_format,
                                      body["digest"])
            snapshot_info = {"path": snap_path, "digest": body["digest"],
                             "snapshot_id": body["snapshot_id"]}
            logger.log("INFO", "snapshot", "mid-run snapshot written",
                       path=snap_path, digest=body["digest"])

    try:
        engine.run_all(on_batch=on_batch)
    except PipelineError as exc:
        logger.anomaly(exc.code, exc.message, **exc.details)
        print(json.dumps(exc.to_dict(), ensure_ascii=False))
        return 2

    final_state = engine.state.to_dict()
    final_snapshot_path = os.path.join(
        args.workdir, "snapshots",
        f"manifest-v{args.snapshot_format}-final.json")
    parent = snapshot_info
    body = write_snapshot(final_snapshot_path, final_state, param_version,
                          snapshot_id=f"snap-final-{param_version}",
                          parent=parent,
                          format_version=args.snapshot_format)
    catalog.register_snapshot(body["snapshot_id"], final_snapshot_path,
                              param_version, args.snapshot_format,
                              body["digest"])

    results = engine.results()
    results["state_digest"] = state_digest(engine.state)
    results["resumed_from"] = resumed_from
    _write_json(os.path.join(args.workdir, "results.json"), results)
    _write_json(os.path.join(args.workdir, "state_dump.json"), final_state)

    # Fault injections that mutate already-written input files.
    if args.inject == "snapshot-digest":
        info = faults.corrupt_snapshot_digest(final_snapshot_path)
        logger.anomaly("E_SNAPSHOT", "injected: snapshot digest corruption",
                       **info)
        try:
            read_snapshot(final_snapshot_path)
        except PipelineError as exc:
            logger.anomaly(exc.code, exc.message, **exc.details)
            print(json.dumps(exc.to_dict(), ensure_ascii=False))
            return 2
    elif args.inject == "log-truncate":
        info = faults.truncate_log_tail(os.path.join(args.workdir,
                                                     "statelog"))
        logger.anomaly("E_LOG", "injected: state log tail truncation", **info)
        try:
            replay_log(os.path.join(args.workdir, "statelog"), tables,
                       query_set, param_version)
        except PipelineError as exc:
            logger.anomaly(exc.code, exc.message, **exc.details)
            print(json.dumps(exc.to_dict(), ensure_ascii=False))
            return 2

    logger.log("INFO", "run_complete", "pipeline run finished",
               state_digest=results["state_digest"])
    print(json.dumps({"code": "OK", "state_digest": results["state_digest"],
                      "workdir": args.workdir}, ensure_ascii=False))
    return 0


def cmd_recover_check(args):
    tables, query_set = _load_inputs(args.tables, args.queries)
    catalog = Catalog(os.path.join(args.workdir, "catalog.db"))
    param_version = catalog.param_version(param_set_fingerprint(query_set))
    log_engine = replay_log(os.path.join(args.workdir, "statelog"), tables,
                            query_set, param_version)
    snap_record, snap_state = rebuild_from_snapshot(args.snapshot)
    if snap_record.query_param_version != param_version:
        raise PipelineError("E_SNAPSHOT", "snapshot param version mismatch",
                            snapshot_version=snap_record.query_param_version,
                            current_version=param_version)
    compare_states(log_engine.state, snap_state)  # E_RECOVER on divergence
    digest = state_digest(log_engine.state)
    report = {"code": "OK", "log_replay_digest": digest,
              "snapshot_digest": state_digest(snap_state),
              "match": True}
    print(json.dumps(report, ensure_ascii=False))
    return 0


def cmd_diverge_check(args):
    tables, query_set = _load_inputs(args.tables, args.queries)
    catalog = Catalog(os.path.join(args.workdir_a, "catalog.db"))
    param_version = catalog.param_version(param_set_fingerprint(query_set))
    engine_a = Engine(tables, query_set, param_version)
    engine_a.state = EngineState.from_dict(
        _read_state_dump(args.workdir_a))
    engine_b = Engine(tables, query_set, param_version)
    engine_b.state = EngineState.from_dict(
        _read_state_dump(args.workdir_b))
    compare_runs(engine_a, engine_b)  # E_DIVERGE on mismatch
    print(json.dumps({"code": "OK",
                      "state_digest": state_digest(engine_a.state)},
                     ensure_ascii=False))
    return 0


def _read_state_dump(workdir):
    with open(os.path.join(workdir, "state_dump.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


def cmd_verify(args):
    tables, query_set = _load_inputs(args.tables, args.queries)
    catalog = Catalog(os.path.join(args.workdir, "catalog.db"))
    param_version = catalog.param_version(param_set_fingerprint(query_set))
    with open(os.path.join(args.workdir, "results.json"),
              encoding="utf-8") as fh:
        results = json.load(fh)
    scale = query_set.session["scale"]
    direction = query_set.session["truncation"]
    check_agg_results(results["aggs"], tables, query_set, param_version,
                      scale, direction)
    check_window_results(results["windows"], tables, query_set, scale,
                         direction)
    state = EngineState.from_dict(_read_state_dump(args.workdir))
    _check_cursors(state, tables)
    print(json.dumps({"code": "OK", "verified": True}, ensure_ascii=False))
    return 0


def _check_cursors(state, tables):
    """No double accumulation and no missing rows relative to cursors."""
    for table_name, table in tables.items():
        intervals = state.applied.get(table_name, [])
        cursor = state.cursors.get(table_name, 0)
        covered = sum(end - start for start, end in intervals)
        for pos in range(1, len(intervals)):
            if intervals[pos][0] < intervals[pos - 1][1]:
                raise PipelineError("E_DIVERGE",
                                    "overlapping applied intervals"
                                    " (double accumulation)",
                                    table=table_name)
        if covered != table.row_count and cursor >= table.row_count:
            raise PipelineError("E_DIVERGE", "missing rows after full run",
                                table=table_name, covered=covered,
                                rows=table.row_count)
    return True


def build_parser():
    parser = argparse.ArgumentParser(
        prog="aggwin", description="streaming agg/window pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--tables", required=True)
    run.add_argument("--queries", required=True)
    run.add_argument("--workdir", required=True)
    run.add_argument("--batch-size", type=int, default=None)
    run.add_argument("--snapshot-at", type=float, default=0.5)
    run.add_argument("--resume-from", default=None)
    run.add_argument("--shuffle-window", type=int, default=0)
    run.add_argument("--shuffle-seed", type=int, default=7)
    run.add_argument("--log-format", type=int, default=2, choices=(1, 2))
    run.add_argument("--snapshot-format", type=int, default=2, choices=(1, 2))
    run.add_argument("--inject", default=None,
                     choices=("snapshot-digest", "log-truncate",
                              "batch-mid-window", "out-of-order"))
    run.add_argument("--catalog", default=None,
                     help="shared catalog.db for cross-run param versions")
    run.set_defaults(func=cmd_run)

    recover = sub.add_parser("recover-check")
    recover.add_argument("--tables", required=True)
    recover.add_argument("--queries", required=True)
    recover.add_argument("--workdir", required=True)
    recover.add_argument("--snapshot", required=True)
    recover.set_defaults(func=cmd_recover_check)

    diverge = sub.add_parser("diverge-check")
    diverge.add_argument("--tables", required=True)
    diverge.add_argument("--queries", required=True)
    diverge.add_argument("--workdir-a", required=True)
    diverge.add_argument("--workdir-b", required=True)
    diverge.set_defaults(func=cmd_diverge_check)

    verify = sub.add_parser("verify")
    verify.add_argument("--tables", required=True)
    verify.add_argument("--queries", required=True)
    verify.add_argument("--workdir", required=True)
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())

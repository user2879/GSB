#!/usr/bin/env python3
"""Standalone verification tool.

Independently reads a snapshot manifest and the state log, checks that
both recovery paths agree (E_RECOVER), and that the reported aggregation
and window results are consistent with the state and cursors: no double
accumulation, no missing rows, canonical window results (E_WINDOW).
"""

import argparse
import json
import os
import sys

from aggwin.catalog import Catalog, param_set_fingerprint
from aggwin.errors import PipelineError
from aggwin.recovery import (compare_states, rebuild_from_snapshot,
                             replay_log, state_digest)
from aggwin.schema import load_queries_file, load_tables_file
from aggwin.verify import check_agg_results, check_window_results


def main(argv=None):
    parser = argparse.ArgumentParser(description="aggwin standalone verifier")
    parser.add_argument("--tables", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--statelog", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--results", default=None,
                        help="optional results.json to cross-check")
    args = parser.parse_args(argv)

    try:
        tables = load_tables_file(args.tables)
        query_set = load_queries_file(args.queries, tables)
        catalog = Catalog(args.catalog)
        param_version = catalog.param_version(
            param_set_fingerprint(query_set))

        log_engine = replay_log(args.statelog, tables, query_set,
                                param_version)
        snap_record, snap_state = rebuild_from_snapshot(args.snapshot)
        if snap_record.query_param_version != param_version:
            raise PipelineError(
                "E_SNAPSHOT", "snapshot param version mismatch",
                snapshot_version=snap_record.query_param_version,
                current_version=param_version)
        compare_states(log_engine.state, snap_state)

        checks = {"log_replay_digest": state_digest(log_engine.state),
                  "snapshot_digest": state_digest(snap_state),
                  "recovery_paths_match": True}

        # Cursor / applied-interval consistency: no double accumulation,
        # no missing rows.
        for table_name, table in tables.items():
            intervals = log_engine.state.applied.get(table_name, [])
            covered = sum(end - start for start, end in intervals)
            for pos in range(1, len(intervals)):
                if intervals[pos][0] < intervals[pos - 1][1]:
                    raise PipelineError(
                        "E_DIVERGE", "double accumulation detected",
                        table=table_name)
            if covered != table.row_count:
                raise PipelineError(
                    "E_DIVERGE", "missing rows relative to cursor",
                    table=table_name, covered=covered,
                    rows=table.row_count)
        checks["cursor_consistent"] = True

        if args.results:
            with open(args.results, encoding="utf-8") as fh:
                results = json.load(fh)
            scale = query_set.session["scale"]
            direction = query_set.session["truncation"]
            check_agg_results(results["aggs"], tables, query_set,
                              param_version, scale, direction)
            check_window_results(results["windows"], tables, query_set,
                                 scale, direction)
            checks["results_consistent"] = True

        report = {"code": "OK", "checks": checks}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())

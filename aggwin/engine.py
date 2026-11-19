"""Streaming engine: batched advance, idempotent row application."""

import json
import random

from .catalog import canonical_query_id
from .errors import PipelineError
from .fixedpoint import to_scaled
from .state import EngineState, GroupAgg, WindowPartition


def build_arrival_order(row_count, session):
    """Deterministic arrival order; optional bounded-displacement shuffle."""
    order = list(range(row_count))
    arrival = session.get("arrival", {})
    window = arrival.get("shuffle_window", 0)
    seed = arrival.get("shuffle_seed")
    if window and seed is not None:
        rng = random.Random(seed)
        order.sort(key=lambda i: (i + rng.uniform(0, window), i))
    return order


class Engine:
    def __init__(self, tables, query_set, param_version, logger=None):
        self.tables = tables
        self.query_set = query_set
        self.session = query_set.session
        self.param_version = param_version
        self.logger = logger
        self.state = EngineState()
        self.scale = self.session["scale"]
        self.direction = self.session["truncation"]
        self.batch_size = self.session["batch_size"]
        self.canonical_ids = {q.qid: canonical_query_id(q)
                              for q in query_set.queries}
        self.arrivals = {name: build_arrival_order(t.row_count, self.session)
                         for name, t in tables.items()}
        for name in tables:
            self.state.cursors[name] = 0
            self.state.applied[name] = []
        for q in query_set.queries:
            if q.kind == "aggregate":
                self.state.aggs[q.qid] = {}
            else:
                self.state.windows[q.qid] = {}

    # -- row application (idempotent) ------------------------------------
    def _row_applied(self, table, row_idx):
        for start, end in self.state.applied[table]:
            if start <= row_idx < end:
                return True
        return False

    def _mark_applied(self, table, row_indices):
        intervals = self.state.applied[table]
        for idx in sorted(row_indices):
            intervals.append([idx, idx + 1])
        intervals.sort()
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self.state.applied[table] = merged

    def apply_rows(self, table_name, row_indices):
        """Apply rows exactly once; re-application is a no-op."""
        table = self.tables[table_name]
        fresh = [i for i in row_indices if not self._row_applied(table_name, i)]
        if not fresh:
            return 0
        for query in self.query_set.queries:
            if query.table != table_name:
                continue
            if query.kind == "aggregate":
                self._apply_agg(query, table, fresh)
            else:
                self._apply_window(query, table, fresh)
        self._mark_applied(table_name, fresh)
        return len(fresh)

    def _apply_agg(self, query, table, row_indices):
        groups = self.state.aggs[query.qid]
        value_col = table.column(query.value_column)
        key_cols = [table.column(c) for c in query.group_by]
        for idx in row_indices:
            if value_col.is_null(idx) or any(c.is_null(idx) for c in key_cols):
                continue
            key = tuple(c.get(idx) for c in key_cols)
            key_json = json.dumps(list(key), ensure_ascii=False)
            group = groups.get(key_json)
            if group is None:
                group = GroupAgg(self.param_version)
                groups[key_json] = group
            group.apply(to_scaled(value_col.get(idx), self.scale,
                                  self.direction))

    def _apply_window(self, query, table, row_indices):
        parts = self.state.windows[query.qid]
        ts_col = table.column(table.timestamp_column)
        value_col = table.column(query.value_column)
        key_cols = [table.column(c) for c in query.partition_by]
        for idx in row_indices:
            if ts_col.is_null(idx) or value_col.is_null(idx) \
                    or any(c.is_null(idx) for c in key_cols):
                continue
            key = tuple(c.get(idx) for c in key_cols)
            key_json = json.dumps(list(key), ensure_ascii=False)
            part = parts.get(key_json)
            if part is None:
                part = WindowPartition(query.max_rows)
                parts[key_json] = part
            part.feed(ts_col.get(idx), idx,
                      to_scaled(value_col.get(idx), self.scale, self.direction),
                      query, self.direction)

    # -- batched advance ---------------------------------------------------
    def _current_table(self):
        for name in self.tables:
            if self.state.cursors[name] < len(self.arrivals[name]):
                return name
        return None

    def advance(self):
        """Process the next batch; returns the batch record or None."""
        table_name = self._current_table()
        if table_name is None:
            return None
        cursor = self.state.cursors[table_name]
        order = self.arrivals[table_name]
        batch = order[cursor:cursor + self.batch_size]
        self.apply_rows(table_name, batch)
        self.state.cursors[table_name] = cursor + len(batch)
        record = {"table": table_name, "rows": batch,
                  "cursor_after": self.state.cursors[table_name]}
        self._check_window_aligned_batches()
        if self.logger:
            self.logger.log("INFO", "batch", "batch applied",
                            table=table_name, rows=len(batch),
                            cursor=self.state.cursors[table_name])
        return record

    def advance_to(self, table_name, target_cursor):
        """Idempotent forward-only advance (re-advancing is a no-op)."""
        records = []
        while self.state.cursors[table_name] < target_cursor:
            record = self.advance()
            if record is None:
                break
            records.append(record)
        return records

    def _check_window_aligned_batches(self):
        """E_BATCH: batch boundary inside an open window (strict mode)."""
        if not self.session.get("window_aligned_batches"):
            return
        if self._current_table() is None:
            return  # end of input: flush is expected
        for qid, parts in self.state.windows.items():
            for key_json, part in parts.items():
                if part.pending:
                    raise PipelineError(
                        "E_BATCH",
                        "batch boundary falls inside an open window",
                        query=qid, partition_key=key_json,
                        pending_rows=len(part.pending))

    def finalize(self):
        """Flush reorder buffers and enforce the out-of-order ratio."""
        for query in self.query_set.queries:
            if query.kind != "window":
                continue
            for key_json, part in self.state.windows[query.qid].items():
                part.flush(query, self.direction)
                if part.total and \
                        part.late / part.total > \
                        self.session["max_out_of_order_ratio"]:
                    raise PipelineError(
                        "E_ORDER",
                        "out-of-order ratio exceeds session limit",
                        query=query.qid, partition_key=key_json,
                        late=part.late, total=part.total,
                        limit=self.session["max_out_of_order_ratio"])

    def run_all(self, on_batch=None):
        records = []
        while True:
            record = self.advance()
            if record is None:
                break
            records.append(record)
            if on_batch:
                on_batch(record)
        self.finalize()
        return records

    # -- results / digest --------------------------------------------------
    def results(self):
        aggs = {}
        for qid, groups in self.state.aggs.items():
            aggs[qid] = {}
            for key_json, group in groups.items():
                body = group.to_dict()
                body["mean"] = group.mean(self.direction)
                aggs[qid][key_json] = body
        windows = {}
        for qid, parts in self.state.windows.items():
            windows[qid] = {key: part.results for key, part in parts.items()}
        return {"param_version": self.param_version,
                "canonical_ids": self.canonical_ids,
                "cursors": self.state.cursors,
                "aggs": aggs, "windows": windows,
                "mean_error_bound": f"1/{self.scale}"}

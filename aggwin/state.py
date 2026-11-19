"""Engine state: grouped aggregation, window ring buffers, cursors."""

from .fixedpoint import mean_scaled


class GroupAgg:
    """Fixed-point per-group aggregation state."""

    __slots__ = ("count", "sum", "min", "max", "rows_processed",
                 "group_version")

    def __init__(self, group_version):
        self.count = 0
        self.sum = 0
        self.min = None
        self.max = None
        self.rows_processed = 0
        self.group_version = group_version

    def apply(self, scaled_value):
        self.count += 1
        self.sum += scaled_value
        self.min = scaled_value if self.min is None else min(self.min,
                                                             scaled_value)
        self.max = scaled_value if self.max is None else max(self.max,
                                                             scaled_value)
        self.rows_processed += 1

    def mean(self, direction):
        return mean_scaled(self.sum, self.count, direction)

    def to_dict(self):
        return {"count": self.count, "sum": self.sum, "min": self.min,
                "max": self.max, "rows_processed": self.rows_processed,
                "group_version": self.group_version}

    @classmethod
    def from_dict(cls, obj):
        state = cls(obj["group_version"])
        state.count = obj["count"]
        state.sum = obj["sum"]
        state.min = obj["min"]
        state.max = obj["max"]
        state.rows_processed = obj["rows_processed"]
        return state


class RingBuffer:
    """Fixed-capacity circular buffer for the active window."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._buf = [None] * capacity
        self._start = 0
        self._size = 0

    def push(self, item):
        if self._size == self.capacity:
            self._buf[self._start] = item
            self._start = (self._start + 1) % self.capacity
        else:
            self._buf[(self._start + self._size) % self.capacity] = item
            self._size += 1

    def clear(self):
        self._buf = [None] * self.capacity
        self._start = 0
        self._size = 0

    def reset(self, items):
        """Replace content wholesale (items oldest -> newest)."""
        self.clear()
        for item in items[-self.capacity:]:
            self.push(item)

    def to_list(self):
        return [self._buf[(self._start + i) % self.capacity]
                for i in range(self._size)]

    def to_dict(self):
        return {"capacity": self.capacity, "items": self.to_list()}

    @classmethod
    def from_dict(cls, obj):
        ring = cls(obj["capacity"])
        ring.reset(obj["items"])
        return ring


def _member_key(item):
    return (item[0], item[1])  # (ts, seq)


class WindowPartition:
    """Per-partition window state.

    history: all arrived rows sorted by (ts, seq) as [ts, seq, value].
    pending: reorder buffer of rows awaiting watermark, same shape.
    results: emitted per-row window aggregates, row seq -> list.
    ring:    ring buffer holding the current window content.
    """

    def __init__(self, max_rows):
        self.max_rows = max_rows
        self.history = []
        self.pending = []
        self.results = {}
        self.ring = RingBuffer(max_rows)
        self.max_ts = None
        self.watermark = None
        self.total = 0
        self.late = 0
        self.dropped = 0
        self.last_emitted = None

    # -- internal helpers -------------------------------------------------
    @staticmethod
    def _insert_sorted(lst, item):
        lo, hi = 0, len(lst)
        key = _member_key(item)
        while lo < hi:
            mid = (lo + hi) // 2
            if _member_key(lst[mid]) < key:
                lo = mid + 1
            else:
                hi = mid
        lst.insert(lo, item)

    def _members(self, ts, preceding, following):
        lo, hi = ts - preceding, ts + following
        return [it for it in self.history if lo <= it[0] <= hi]

    def _emit(self, item, query, direction):
        ts, seq, _val = item
        members = self._members(ts, query.interval["preceding"],
                                query.interval["following"])
        window = members[-self.max_rows:]
        self.results[str(seq)] = _aggregate_window(
            [m[2] for m in window], direction)
        self.ring.reset(window)
        self.last_emitted = seq

    def _recompute_affected(self, late_ts, query, direction):
        preceding = query.interval["preceding"]
        following = query.interval["following"]
        lo, hi = late_ts - following, late_ts + preceding
        for key in list(self.results.keys()):
            row = self._find_history(int(key))
            if row is None:
                continue
            if lo <= row[0] <= hi:
                members = self._members(row[0], preceding, following)
                self.results[key] = _aggregate_window(
                    [m[2] for m in members[-self.max_rows:]], direction)
        if self.last_emitted is not None:
            last = self._find_history(self.last_emitted)
            if last is not None:
                members = self._members(last[0], preceding, following)
                self.ring.reset(members[-self.max_rows:])

    def _find_history(self, seq):
        for item in self.history:
            if item[1] == seq:
                return item
        return None

    # -- public API -------------------------------------------------------
    def feed(self, ts, seq, value, query, direction):
        """Feed one arrived row; returns list of emitted row seqs."""
        self.total += 1
        if self.max_ts is None or ts > self.max_ts:
            self.max_ts = ts
        watermark = self.max_ts - query.out_of_order["max_lateness"]
        if self.watermark is None or watermark > self.watermark:
            self.watermark = watermark
        emitted = []
        if ts < self.watermark:
            self.late += 1
            if query.out_of_order["policy"] == "recompute":
                self._insert_sorted(self.history, [ts, seq, value])
                self._recompute_affected(ts, query, direction)
            else:
                self.dropped += 1
            return emitted
        self._insert_sorted(self.history, [ts, seq, value])
        self._insert_sorted(self.pending, [ts, seq, value])
        emitted.extend(self._drain(query, direction))
        return emitted

    def _drain(self, query, direction):
        emitted = []
        following = query.interval["following"]
        while self.pending and \
                self.pending[0][0] + following <= self.watermark:
            item = self.pending.pop(0)
            self._emit(item, query, direction)
            emitted.append(item[1])
        return emitted

    def flush(self, query, direction):
        """End-of-input reconciliation.

        Rows arriving later than the watermark cannot be emitted
        incrementally with complete windows, so finalisation recomputes
        every result from the full sorted history.  Final results and
        ring content are then independent of batch size and arrival
        order (recompute policy).
        """
        emitted = [item[1] for item in self.pending]
        self.pending = []
        preceding = query.interval["preceding"]
        following = query.interval["following"]
        for item in self.history:
            ts, seq = item[0], item[1]
            members = self._members(ts, preceding, following)
            self.results[str(seq)] = _aggregate_window(
                [m[2] for m in members[-self.max_rows:]], direction)
        if self.history:
            last = self.history[-1]
            members = self._members(last[0], preceding, following)
            self.ring.reset(members[-self.max_rows:])
            self.last_emitted = last[1]
        return emitted

    def to_dict(self):
        return {"max_rows": self.max_rows, "history": self.history,
                "pending": self.pending, "results": self.results,
                "ring": self.ring.to_dict(), "max_ts": self.max_ts,
                "watermark": self.watermark, "total": self.total,
                "late": self.late, "dropped": self.dropped,
                "last_emitted": self.last_emitted}

    @classmethod
    def from_dict(cls, obj):
        part = cls(obj["max_rows"])
        part.history = [list(it) for it in obj["history"]]
        part.pending = [list(it) for it in obj["pending"]]
        part.results = {k: list(v) for k, v in obj["results"].items()}
        part.ring = RingBuffer.from_dict(obj["ring"])
        part.max_ts = obj["max_ts"]
        part.watermark = obj["watermark"]
        part.total = obj["total"]
        part.late = obj["late"]
        part.dropped = obj["dropped"]
        part.last_emitted = obj["last_emitted"]
        return part


def _aggregate_window(scaled_values, direction):
    """[count, sum, min, max, mean] over scaled ints."""
    if not scaled_values:
        return [0, 0, None, None, 0]
    total = sum(scaled_values)
    return [len(scaled_values), total, min(scaled_values),
            max(scaled_values), mean_scaled(total, len(scaled_values),
                                            direction)]


class EngineState:
    """Whole-pipeline state: cursors, applied ranges, aggs, windows."""

    def __init__(self):
        self.cursors = {}          # table -> next arrival position
        self.applied = {}          # table -> list of [start, end) intervals
        self.aggs = {}             # qid -> {group_key_json: GroupAgg}
        self.windows = {}          # qid -> {part_key_json: WindowPartition}

    def to_dict(self):
        return {
            "cursors": self.cursors,
            "applied": self.applied,
            "aggs": {qid: {k: g.to_dict() for k, g in groups.items()}
                     for qid, groups in self.aggs.items()},
            "windows": {qid: {k: p.to_dict() for k, p in parts.items()}
                        for qid, parts in self.windows.items()},
        }

    @classmethod
    def from_dict(cls, obj):
        state = cls()
        state.cursors = {k: int(v) for k, v in obj["cursors"].items()}
        state.applied = {k: [list(iv) for iv in v]
                         for k, v in obj["applied"].items()}
        state.aggs = {qid: {k: GroupAgg.from_dict(g)
                            for k, g in groups.items()}
                      for qid, groups in obj["aggs"].items()}
        state.windows = {qid: {k: WindowPartition.from_dict(p)
                               for k, p in parts.items()}
                         for qid, parts in obj["windows"].items()}
        return state

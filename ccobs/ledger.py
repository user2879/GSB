"""观测账本: 批次切分、窗口封存与窗口内仅失效子集重算。

确定性约定:
  * 事件按脚本事件序号连续进入账本, 时钟为逻辑单调时钟(毫秒)。
  * 批次边界只由 (initial_seq, batch_size) 决定, 与运行次数无关。
  * 已封存窗口的结论值与窗口版本号逐字节不变, 可经 digest 校验。
"""
import hashlib
import json
from dataclasses import dataclass, field

from . import fixedpoint as fp
from .errors import E_CWND, E_SCRIPT, BizError

KIND_ACK = "ack"
KIND_DUP_ACK = "dup_ack"
KIND_TIMEOUT = "timeout"

EVENT_KINDS = (KIND_ACK, KIND_DUP_ACK, KIND_TIMEOUT)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_of(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Event:
    event_seq: int
    kind: str
    rtt_ms: int
    clock_ms: int
    arrival_order: int

    def record(self):
        return {
            "event_seq": self.event_seq,
            "kind": self.kind,
            "rtt_ms": self.rtt_ms,
            "clock_ms": self.clock_ms,
            "arrival_order": self.arrival_order,
        }


@dataclass
class ObservationItem:
    name: str
    value: int
    error_bound: int
    batch_lo: int
    batch_hi: int
    affected: bool
    halted: bool
    version: int
    recomputes: int = 0

    def record(self):
        return {
            "name": self.name,
            "value": self.value,
            "error_bound": self.error_bound,
            "batch_lo": self.batch_lo,
            "batch_hi": self.batch_hi,
            "affected": self.affected,
            "halted": self.halted,
            "version": self.version,
        }


@dataclass
class Batch:
    index: int
    event_lo: int
    event_hi: int
    clock_ms: int
    config_version: int
    complete: bool
    digest: str

    def record(self):
        return {
            "index": self.index,
            "event_lo": self.event_lo,
            "event_hi": self.event_hi,
            "clock_ms": self.clock_ms,
            "config_version": self.config_version,
            "complete": self.complete,
            "digest": self.digest,
        }


@dataclass
class Window:
    index: int
    batch_lo: int
    batch_hi: int
    sealed: bool
    stale: bool
    version: int
    items: dict = field(default_factory=dict)
    digest: str = ""

    def record(self):
        return {
            "index": self.index,
            "batch_lo": self.batch_lo,
            "batch_hi": self.batch_hi,
            "sealed": self.sealed,
            "stale": self.stale,
            "version": self.version,
            "items": {k: self.items[k].record() for k in sorted(self.items)},
        }


class Ledger:
    def __init__(self, config, config_version):
        self.config = config
        self.config_version = config_version
        self.events = []
        self.batches = []
        self.sealed_windows = []
        self.current = None
        self.pending = []
        self.expected_seq = config.initial_seq
        self.live = fp.initial_state(config)
        self.window_checkpoint = self.live
        self.window_events = []
        self.halted_items = set()
        self.window_halted = set()
        self.window_dup_count = 0
        self.gap_raised = False
        self.cwnd_raised_window = -1
        self.anomalies = []
        self.late_events = []
        self.trace = []  # 每项 (event_seq, item, value), 供重放逐项比对

    # ---- 事件进入 ------------------------------------------------------
    def ingest_event(self, event):
        if event.event_seq > self.expected_seq and not self.gap_raised:
            self.gap_raised = True
            missing = event.event_seq - self.expected_seq
            self.anomalies.append(BizError(
                E_SCRIPT, "script event sequence gap",
                gap_start=self.expected_seq, missing=missing))
            self.halted_items = set(fp.ALL_ITEMS)
        if event.event_seq < self.expected_seq:
            # 缺口区域的迟到/重复应答: 登记但不参与推进
            self.late_events.append(event.event_seq)
            return
        self.expected_seq = event.event_seq + 1
        self.events.append(event)
        self._advance_live(event)
        self._append_trace(event)
        self.pending.append(event)
        if len(self.pending) == self.config.batch_size:
            self.finalize_batch()

    def _advance_live(self, event):
        if event.kind == KIND_ACK:
            self.live = fp.apply_ack(fp.apply_send(self.live, self.config),
                                     self.config, event.rtt_ms)
        elif event.kind == KIND_DUP_ACK:
            self.live = fp.apply_dup_ack(fp.apply_send(self.live, self.config),
                                         self.config)
        elif event.kind == KIND_TIMEOUT:
            self.live = fp.apply_timeout(self.live, self.config)
        else:
            raise ValueError("unknown event kind %r" % (event.kind,))

    def _append_trace(self, event):
        for item in fp.ALL_ITEMS:
            if item in self.halted_items or item in self.window_halted:
                value = self._last_trace_value(item)
            else:
                value = fp.item_value(self.live, item)
            self.trace.append((event.event_seq, item, value))

    def _last_trace_value(self, item):
        for _seq, name, value in reversed(self.trace):
            if name == item:
                return value
        return fp.item_value(fp.initial_state(self.config), item)

    # ---- 批次 ----------------------------------------------------------
    def finalize_batch(self):
        if not self.pending:
            return None
        events = self.pending
        self.pending = []
        index = len(self.batches)
        batch = Batch(
            index=index,
            event_lo=events[0].event_seq,
            event_hi=events[-1].event_seq,
            clock_ms=events[-1].clock_ms,
            config_version=self.config_version,
            complete=True,
            digest=digest_of({
                "index": index,
                "events": [e.record() for e in events],
                "config_version": self.config_version,
            }),
        )
        self.batches.append(batch)
        self.window_events.extend(events)
        self._ensure_window(batch)
        self._check_dup_ack_cluster(events)
        self._recompute_affected(events, batch)
        self.current.batch_hi = batch.index
        if self.current.batch_hi - self.current.batch_lo + 1 == self.config.window_batches:
            self._seal_window()
        return batch

    # ---- 观测窗口 ------------------------------------------------------
    def _ensure_window(self, batch):
        if self.current is None:
            self.current = Window(
                index=len(self.sealed_windows),
                batch_lo=batch.index,
                batch_hi=-1,
                sealed=False,
                stale=False,
                version=self.config_version,
            )

    def _check_dup_ack_cluster(self, events):
        self.window_dup_count += sum(1 for e in events if e.kind == KIND_DUP_ACK)
        limit = self.config.dup_ack_limit
        if self.window_dup_count > limit and self.cwnd_raised_window != self.current.index:
            self.cwnd_raised_window = self.current.index
            over = self.window_dup_count - limit
            self.anomalies.append(BizError(
                E_CWND, "duplicate ack cluster over limit",
                over=over, limit=limit, window=self.current.index))
            self.window_halted |= {fp.ITEM_CWND, fp.ITEM_SSTHRESH}

    def _recompute_affected(self, events, batch):
        kinds = {e.kind for e in events}
        if batch.index == self.current.batch_lo:
            affected = set(fp.ALL_ITEMS)
        else:
            affected = {item for item in fp.ALL_ITEMS
                        if kinds & set(fp.ITEM_EVENT_KINDS[item])}
        affected -= self.halted_items
        affected -= self.window_halted
        if not affected:
            return
        state = self.window_checkpoint
        for event in self.window_events:
            state = self._apply(state, event)
        for item in sorted(affected):
            prev = self.current.items.get(item)
            self.current.items[item] = ObservationItem(
                name=item,
                value=fp.item_value(state, item),
                error_bound=fp.item_error_bound(state, item),
                batch_lo=self.current.batch_lo,
                batch_hi=batch.index,
                affected=True,
                halted=False,
                version=self.config_version,
                recomputes=(prev.recomputes + 1) if prev else 1,
            )

    def _apply(self, state, event):
        if event.kind == KIND_ACK:
            return fp.apply_ack(state, self.config, event.rtt_ms)
        if event.kind == KIND_DUP_ACK:
            return fp.apply_dup_ack(state, self.config)
        if event.kind == KIND_TIMEOUT:
            return fp.apply_timeout(state, self.config)
        raise ValueError("unknown event kind %r" % (event.kind,))

    def _seal_window(self):
        window = self.current
        window.sealed = True
        window.version = self.config_version
        window.digest = digest_of(window.record())
        self.sealed_windows.append(window)
        self.current = None
        self.window_events = []
        self.window_checkpoint = self.live
        self.window_dup_count = 0
        self.window_halted = set()

    # ---- 校验与视图 ----------------------------------------------------
    def sealed_bytes(self, window_index):
        window = self.sealed_windows[window_index]
        return canonical(window.record()).encode("utf-8")

    def verify_sealed(self):
        """逐项校验已封存窗口: 返回首个被篡改的窗口序号, 全部一致返回 None。"""
        for window in self.sealed_windows:
            if digest_of(window.record()) != window.digest:
                return window.index
        return None

    def all_windows(self):
        windows = list(self.sealed_windows)
        if self.current is not None:
            windows.append(self.current)
        return windows

    @property
    def finished(self):
        return not self.pending

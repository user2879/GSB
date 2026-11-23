"""恢复后重放校验。

按快照记录的事件序列重新推进账本, 与快照中逐项登记的轨迹比对;
首个不一致处须定位到事件序号与观测项, 并以 E_REPLAY 报告依据。
"""
from .errors import E_REPLAY, BizError
from .ledger import Ledger


class ReplayResult:
    def __init__(self, ok, error=None, compared=0):
        self.ok = ok
        self.error = error
        self.compared = compared

    def __str__(self):
        if self.ok:
            return "REPLAY_OK|compared=%d" % self.compared
        return str(self.error)


def replay_verify(config, config_version, events, stored_trace):
    """重放 events 并与 stored_trace [(event_seq, item, value)] 逐项比对。"""
    ledger = Ledger(config, config_version)
    for event in events:
        ledger.ingest_event(event)
    fresh = ledger.trace
    compared = 0
    for stored_row, fresh_row in zip(stored_trace, fresh):
        compared += 1
        if stored_row != fresh_row:
            event_seq, item, stored_value = stored_row
            actual = fresh_row[2]
            return ReplayResult(False, error=BizError(
                E_REPLAY, "replay trace mismatch",
                event_seq=event_seq, item=item,
                expected=stored_value, actual=actual), compared=compared)
    if len(stored_trace) != len(fresh):
        where = stored_trace[len(fresh)] if len(stored_trace) > len(fresh) \
            else fresh[len(stored_trace)]
        return ReplayResult(False, error=BizError(
            E_REPLAY, "replay trace length mismatch",
            event_seq=where[0], item=where[1],
            expected=len(stored_trace), actual=len(fresh)), compared=compared)
    return ReplayResult(True, compared=compared)

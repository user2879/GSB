"""恢复后的一致性重放校验与差异定位(业务码 E_REPLAY)。

从快照/账本恢复后, 按记录的事件游标重放同一事件序列, 将每事件后的
实例健康状态、域隔离状态、期望隔离结论、指令回执摘要与账本逐项比较;
差异定位到首个不一致的事件序号、实例标识与字段。故障域版本或事件流
摘要不一致时标注为不可复现并说明原因。
"""

from __future__ import annotations

import json

from .errors import DiagError
from .spec import events_digest, domains_version
from .state import Engine
from .util import digest_obj


def _receipts_digest(ledger, event_seq: int) -> str:
    rows = ledger.commands(event_seq)
    return digest_obj([[r["cmd_seq"], r["instance"], r["action"], r["attempt"],
                        r["status"], r["receipt_json"]] for r in rows])


def _confirm_after_event(engine: Engine, intents) -> None:
    """与运行器一致的确定性收敛确认(重放时假定已提交回执有效)。"""
    domains_isolate = {i.domain for i in intents if i.action == "isolate"}
    domains_recover = {i.domain for i in intents if i.action == "recover"}
    for domain in sorted(domains_isolate):
        engine.confirm_isolated(domain)
    for domain in sorted(domains_recover):
        engine.confirm_normal(domain)


def replay_verify(ledger, instances, domains, events) -> dict:
    meta = ledger.meta()
    problems = []
    if meta.get("domains_version") != domains_version(domains):
        problems.append("domains_version mismatch with ledger")
    if meta.get("events_digest") != events_digest(events):
        problems.append("events_digest mismatch with ledger")
    if problems:
        return {"reproducible": False, "reason": "; ".join(problems),
                "checked_events": 0}

    cursor, _ = ledger.cursor()
    engine = Engine(instances, domains)
    checked = 0
    for event in events:
        if event.seq > cursor:
            break
        intents = engine.apply_event(event)
        _confirm_after_event(engine, intents)
        row = ledger.event_state(event.seq)
        if row is None:
            raise DiagError("E_REPLAY",
                            f"event {event.seq}: missing committed event state",
                            {"event_seq": event.seq, "instance": event.instance,
                             "field": "event_state", "expected": "present",
                             "actual": "missing"})
        stored_instances = json.loads(row["instance_states"])
        stored_domains = json.loads(row["domain_states"])
        stored_expected = json.loads(row["expected"])
        for iid in sorted(engine.health):
            if stored_instances.get(iid) != engine.health[iid]:
                raise DiagError("E_REPLAY",
                                f"event {event.seq}: instance {iid} health diverged",
                                {"event_seq": event.seq, "instance": iid,
                                 "field": "health",
                                 "expected": engine.health[iid],
                                 "actual": stored_instances.get(iid)})
        for did in sorted(engine.domain_state):
            if stored_domains.get(did) != engine.domain_state[did]:
                raise DiagError("E_REPLAY",
                                f"event {event.seq}: domain {did} state diverged",
                                {"event_seq": event.seq, "instance": event.instance,
                                 "field": f"domain_state:{did}",
                                 "expected": engine.domain_state[did],
                                 "actual": stored_domains.get(did)})
        recomputed_expected = {did: engine.expected_view(did)
                               for did in sorted(engine.domain_state)}
        if stored_expected != recomputed_expected:
            raise DiagError("E_REPLAY",
                            f"event {event.seq}: expected isolation view diverged",
                            {"event_seq": event.seq, "instance": event.instance,
                             "field": "expected",
                             "expected": digest_obj(recomputed_expected),
                             "actual": digest_obj(stored_expected)})
        recomputed_receipts = _receipts_digest(ledger, event.seq)
        if row["receipts_digest"] != recomputed_receipts:
            raise DiagError("E_REPLAY",
                            f"event {event.seq}: receipts digest diverged",
                            {"event_seq": event.seq, "instance": event.instance,
                             "field": "receipts_digest",
                             "expected": recomputed_receipts,
                             "actual": row["receipts_digest"]})
        checked += 1
    return {"reproducible": True, "checked_events": checked, "cursor": cursor}

"""Deterministic replay of the ledger event sequence.

Recovery rebuilds state from durable records, then re-adjudicates the exact
request sequence on a virtual clock.  Side effects (expiry, preemption and
rollback) are re-derived by the same engine rules and compared against the
persisted event stream.  Per-request outcomes, lease ownership and batch
watermarks are checked field by field; the first divergence is reported as
``E_REPLAY`` with the offending request id and field path.
"""
from __future__ import annotations

from .clock import VirtualClock
from .engine import Engine
from .errors import E_GAP, E_REPLAY, LeaseError
from .model import (
    REPLAYABLE_EVENTS, Request, EV_DENY, EV_EXPIRE, EV_GRANT, EV_RENEW,
    EV_REVOKE, EV_ROLLBACK,
)

# Events produced directly as the decision for a submitted request.
DECISION_EVENTS = frozenset({EV_GRANT, EV_RENEW, EV_DENY})


def _request_index(requests: list[Request]) -> dict[str, Request]:
    return {req.request_id: req for req in requests}


def assert_no_gap(ledger, requests: list[Request]) -> None:
    """Every request id referenced by the ledger must exist in the stream."""
    known = {req.request_id for req in requests}
    missing = [rid for rid in ledger.request_sequence() if rid not in known]
    if missing:
        raise LeaseError(
            E_GAP, "ledger references requests missing from the stream "
            "(e.g. a missing renewal event)",
            [f"request_id={rid}" for rid in missing])


def replay_engine(ledger, pools: dict, rollback_targets: dict,
                  requests: list[Request]) -> tuple[Engine, list[dict]]:
    assert_no_gap(ledger, requests)
    index = _request_index(requests)
    engine = Engine(pools, rollback_targets)
    clock = VirtualClock()
    comparisons: list[dict] = []
    ledger_events = [event for event in ledger.events()
                     if event["event_type"] in REPLAYABLE_EVENTS]
    handled: set[str] = set()

    # The decision event of a request is its GRANT/RENEW/DENY record. A
    # preemption request may have preceding REVOKE/ROLLBACK side effects, so we
    # index the decision records separately and walk them in request order.
    decision_records = [event for event in ledger_events
                        if event.get("request_id")
                        and event["event_type"] in DECISION_EVENTS]
    for record in decision_records:
        rid = record["request_id"]
        if rid in handled:
            continue
        handled.add(rid)
        request = index[rid]
        # Drive the virtual clock with the persisted adjudication timestamp,
        # which is the same monotonic event-time basis used by the live run.
        at_ms = record["at_ms"]
        clock.advance(max(0, at_ms - clock.now_ms()))
        # Expiry is a clock-driven side effect and is reproduced before the
        # request adjudicates; preemption/rollback derive inside submit().
        engine.tick(clock.now_ms())
        decision = engine.submit(request, clock.now_ms())
        comparisons.append(_compare_decision(decision, record, request))

    # Verify side-effect events (revoke/rollback/expire) match one-for-one.
    _compare_side_effects(engine, ledger_events)
    return engine, comparisons


def _compare_decision(decision, record: dict, request: Request) -> dict:
    expected_type = record["event_type"]
    mismatches: list[str] = []
    if decision.event_type != expected_type:
        mismatches.append(
            f"event_type: replay={decision.event_type} ledger={expected_type}")
    if decision.accepted != (expected_type in {EV_GRANT, EV_RENEW}):
        mismatches.append(
            f"accepted: replay={decision.accepted} "
            f"ledger={expected_type in {EV_GRANT, EV_RENEW}}")
    if decision.lease_id != record.get("lease_id"):
        mismatches.append(
            f"lease_id: replay={decision.lease_id} "
            f"ledger={record.get('lease_id')}")
    if mismatches:
        raise LeaseError(
            E_REPLAY, "replay outcome diverges from ledger",
            [f"request_id={request.request_id}", *mismatches],
            {"request_id": request.request_id})
    return {"request_id": request.request_id, "event_type": expected_type,
            "lease_id": decision.lease_id, "match": True}


def _compare_side_effects(engine: Engine, ledger_events: list[dict]) -> None:
    side_kinds = frozenset({EV_EXPIRE, EV_REVOKE, EV_ROLLBACK})
    expected = [(event["event_type"], event.get("lease_id"), event["at_ms"])
                for event in ledger_events
                if event["event_type"] in side_kinds]
    actual = [(event.event_type, event.lease_id, event.at_ms)
              for event in engine.events
              if event.event_type in side_kinds]
    if expected != actual:
        first = next((idx for idx, pair in enumerate(zip(expected, actual))
                      if pair[0] != pair[1]), min(len(expected), len(actual)))
        raise LeaseError(
            E_REPLAY, "side-effect event sequence diverges at first mismatch",
            [f"index={first}", f"ledger={expected[first:first + 1]}",
             f"replay={actual[first:first + 1]}"])


def compare_watermarks(ledger, replay_marks: list[dict]) -> None:
    """Compare each committed batch's deterministic watermark fields."""
    deterministic = {"scope", "effective_leases", "capacity",
                     "used_capacity", "ratio_fp", "scale", "pool_id"}
    for batch in ledger.committed_batches():
        bid = int(batch["batch_id"])
        stored = {mark["scope"]: mark
                  for mark in ledger.watermarks_for_batch(bid)}
        replayed = {mark["scope"]: mark
                    for mark in replay_marks if mark["batch_id"] == bid}
        for scope, stored_mark in stored.items():
            replay_mark = replayed.get(scope)
            if replay_mark is None:
                raise LeaseError(E_REPLAY,
                                 "replay produced no watermark for scope",
                                 [f"batch_id={bid}", f"scope={scope}"])
            for field in deterministic:
                if field not in stored_mark:
                    continue
                if stored_mark[field] != replay_mark.get(field):
                    raise LeaseError(
                        E_REPLAY, "watermark divergence at first mismatch",
                        [f"batch_id={bid}", f"scope={scope}",
                         f"field={field}",
                         f"ledger={stored_mark[field]!r}",
                         f"replay={replay_mark.get(field)!r}"])


def compare_lease_ownership(engine: Engine, ledger) -> None:
    ledger_leases = ledger.latest_leases()
    if set(engine.leases) != set(ledger_leases):
        only_ledger = sorted(set(ledger_leases) - set(engine.leases))
        only_replay = sorted(set(engine.leases) - set(ledger_leases))
        raise LeaseError(E_REPLAY, "lease set divergence",
                         [f"only_in_ledger={only_ledger}",
                          f"only_in_replay={only_replay}"])
    for lease_id, lease in engine.leases.items():
        record = ledger_leases[lease_id]
        for field in ("state", "revision", "renew_count", "expire_ms",
                      "pool_id", "key"):
            if getattr(lease, field) != record.get(field):
                raise LeaseError(
                    E_REPLAY, "lease ownership divergence at first mismatch",
                    [f"lease_id={lease_id}", f"field={field}",
                     f"replay={getattr(lease, field)!r}",
                     f"ledger={record.get(field)!r}"])

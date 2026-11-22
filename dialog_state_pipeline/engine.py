"""Per-session deterministic processing engine with hash-chain caching.

Each turn gets a cache key derived from the previous turn's key and the turn's
content fingerprint plus FSM/config fingerprints::

    key(t) = H(key(t-1) | content(t) | fsm_fp | config_fp)

Because every key depends on every upstream key, editing/deleting one turn
invalidates that turn and all downstream checkpoints; turns before the edit
keep identical keys and are reused without re-processing.
"""

import copy

from .hashing import chain_digest, short_digest
from .models import SessionResult, TurnCheckpoint
from .slots import apply_updates, make_records
from .state_machine import attempt_transition, build_transition_index

GENESIS_KEY = "0" * 32


def turn_payload_fingerprint(turn):
    """Fingerprint of the upstream event content of one materialised turn."""
    material = {
        "session_id": turn["session_id"],
        "turn": turn["turn"],
        "utterance": turn["utterance"],
        "action": turn["action"],
        "slot_updates": turn["slot_updates"],
        "batch_id": turn["batch_id"],
        "source_turn": turn.get("source_turn"),
    }
    return short_digest(material)


def compute_keys(turns, fsm, config):
    """Return the full list of chain keys for a materialised turn list."""
    keys = []
    prev = GENESIS_KEY
    for turn in turns:
        payload = {
            "content": turn_payload_fingerprint(turn),
            "fsm": fsm.fingerprint,
            "config": config.fingerprint,
        }
        prev = chain_digest(prev, payload)
        keys.append(prev)
    return keys


def longest_reused_prefix(keys, cache_by_key):
    """Count how many leading turns are already present in the cache."""
    count = 0
    for key in keys:
        if key in cache_by_key:
            count += 1
        else:
            break
    return count


def run_session(session_id, turns, fsm, config, cache, transition_index,
                quota, turn_limit, quota_injection_id=None):
    """Process one session up to ``turn_limit`` *new* turns.

    ``cache`` maps cache key -> TurnCheckpoint and is shared (read) across
    runs.  ``turn_limit`` is the per-call budget allocation handed out by the
    scheduler; ``quota`` is this session's hard state-entry quota.

    Returns a :class:`SessionResult` plus the list of newly produced
    checkpoints (caller merges them into the global cache).
    """
    keys = compute_keys(turns, fsm, config)
    reused = longest_reused_prefix(keys, cache)

    if reused > 0:
        anchor = cache[keys[reused - 1]]
        state = anchor.state
        slots = copy.deepcopy(anchor.slots)
        transition_trace = copy.deepcopy(anchor.transition_trace)
        rejected = copy.deepcopy(anchor.rejected)
        # Budget-pending markers are scheduler-run scoped, never cached state.
        failures = [item for item in copy.deepcopy(anchor.failures)
                    if item.get("code") != "E_BUDGET_PENDING"]
        consumed = anchor.consumed_entries
        checkpoints = [cache[k] for k in keys[:reused]]
    else:
        state = fsm.initial_state
        slots = make_records(fsm)
        transition_trace = []
        rejected = []
        failures = []
        consumed = 0
        checkpoints = []

    processed_turns = []
    new_checkpoints = []
    status = "complete"
    total = len(turns)
    produced = 0

    for idx in range(reused, total):
        turn = turns[idx]
        turn_no = turn["turn"]
        injection_id = turn.get("injection_id")
        batch_id = turn.get("batch_id", "")

        # Hard per-session quota: stop before consuming another entry.
        if consumed >= quota:
            status = "quota_exceeded"
            failures.append(_quota_failure(session_id, turn_no, batch_id,
                                           injection_id or quota_injection_id,
                                           quota))
            break
        # Scheduler allocation for this invocation.
        if produced >= turn_limit:
            status = "budget_pending"
            break

        try:
            slot_failures = apply_updates(
                slots, turn.get("slot_updates", []), turn_no, batch_id,
                fsm, config, injection_id=injection_id)
            for item in slot_failures:
                item["session_id"] = session_id
                failures.append(item)

            if turn.get("action") is not None:
                entry, rejected_entry = attempt_transition(
                    state, turn["action"], turn_no, batch_id,
                    transition_index, injection_id=injection_id)
                if entry is not None:
                    transition_trace.append(entry)
                    state = entry["to_state"]
                else:
                    rejected_entry["session_id"] = session_id
                    rejected.append(rejected_entry)
                    failures.append({
                        "session_id": session_id,
                        "turn": turn_no,
                        "stage": "transition",
                        "code": "E_TRANSITION_REJECTED",
                        "message": rejected_entry["message"],
                        "batch_id": batch_id,
                        "injection_id": injection_id,
                    })
        except Exception as exc:  # isolation: one bad event never kills a session
            failures.append({
                "session_id": session_id,
                "turn": turn_no,
                "stage": "engine",
                "code": "E_STREAM_INVALID",
                "message": f"unexpected engine error: {type(exc).__name__}: {exc}",
                "batch_id": batch_id,
                "injection_id": injection_id,
            })

        consumed += 1
        produced += 1
        processed_turns.append(turn_no)
        checkpoint = TurnCheckpoint(
            session_id=session_id,
            turn=turn_no,
            key=keys[idx],
            state=state,
            slots=copy.deepcopy(slots),
            transition_trace=copy.deepcopy(transition_trace),
            rejected=copy.deepcopy(rejected),
            failures=copy.deepcopy(failures),
            consumed_entries=consumed,
        )
        new_checkpoints.append(checkpoint)
        checkpoints.append(checkpoint)

    result = SessionResult(
        session_id=session_id,
        status=status,
        final_state=state,
        slots=slots,
        transition_trace=transition_trace,
        rejected=rejected,
        failures=failures,
        checkpoints=checkpoints,
        processed_turns=processed_turns,
        reused_turns=[turns[i]["turn"] for i in range(reused)],
        consumed_entries=consumed,
    )
    return result, new_checkpoints, keys


def _quota_failure(session_id, turn_no, batch_id, injection_id, quota):
    return {
        "session_id": session_id,
        "turn": turn_no,
        "stage": "engine",
        "code": "E_QUOTA_EXCEEDED",
        "message": f"session quota of {quota} state entries exhausted",
        "batch_id": batch_id,
        "injection_id": injection_id,
    }

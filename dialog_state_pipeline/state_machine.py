"""State-machine transition lookup with explicit rejection bookkeeping."""


def build_transition_index(fsm):
    """Map ``(from_state, action) -> TransitionRule`` for O(1) lookup."""
    index = {}
    for rule in fsm.transitions:
        index[(rule.from_state, rule.action)] = rule
    return index


def attempt_transition(current_state, action, turn, batch_id, index,
                       injection_id=None):
    """Return ``(entry, rejected)`` where exactly one is populated.

    A missing rule is *not* a crash: the transition is refused, an
    ``E_TRANSITION_REJECTED`` record is stored and the caller must keep the
    current state and continue with later turns.
    """
    rule = index.get((current_state, action))
    if rule is None:
        rejected = {
            "turn": turn,
            "from_state": current_state,
            "action": action,
            "rule_id": None,
            "batch_id": batch_id,
            "injection_id": injection_id,
            "code": "E_TRANSITION_REJECTED",
            "message": f"no transition from {current_state!r} for action {action!r}",
        }
        return None, rejected
    entry = {
        "turn": turn,
        "rule_id": rule.rule_id,
        "from_state": rule.from_state,
        "to_state": rule.to_state,
        "action": rule.action,
        "batch_id": batch_id,
        "injection_id": injection_id,
    }
    return entry, None

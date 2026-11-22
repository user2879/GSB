from dialog_state_pipeline.state_machine import (
    attempt_transition, build_transition_index,
)


def test_accepted_transition_records_full_provenance(fsm):
    index = build_transition_index(fsm)
    entry, rejected = attempt_transition("greeting", "start", 1, "B1", index)
    assert rejected is None
    assert entry == {
        "turn": 1, "rule_id": "R1", "from_state": "greeting",
        "to_state": "collecting", "action": "start",
        "batch_id": "B1", "injection_id": None,
    }


def test_rejected_transition_keeps_state(fsm):
    index = build_transition_index(fsm)
    entry, rejected = attempt_transition("collecting", "fly", 2, "B2", index)
    assert entry is None
    assert rejected["code"] == "E_TRANSITION_REJECTED"
    assert rejected["from_state"] == "collecting"
    # Caller (engine) stays in collecting and subsequent turns still work.
    entry2, _ = attempt_transition("collecting", "provide", 3, "B2", index)
    assert entry2["rule_id"] == "R2"
    assert entry2["from_state"] == "collecting"


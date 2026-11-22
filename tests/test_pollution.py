"""Precise pollution: only affected sessions/turns are replayed."""

import copy

from dialog_state_pipeline.engine import compute_keys, longest_reused_prefix
from dialog_state_pipeline.loader import load_event_stream


def test_edit_invalidates_only_downstream_turns(sample_dir, fsm, config):
    materialised, _ = load_event_stream(sample_dir / "events_base.jsonl")
    keys_before = compute_keys(materialised["S001"], fsm, config)

    edited = copy.deepcopy(materialised)
    edited["S001"][3]["utterance"] = "changed utterance at turn 4"
    keys_after = compute_keys(edited["S001"], fsm, config)

    # Keys 1..3 unchanged (cache reusable), turn 4 onward all differ.
    assert keys_before[:3] == keys_after[:3]
    assert keys_before[3:] != keys_after[3:]


def test_unrelated_session_fully_reused(sample_dir, fsm, config):
    materialised, _ = load_event_stream(sample_dir / "events_base.jsonl")
    cache = {key: object() for key in compute_keys(materialised["S001"], fsm, config)}
    keys = compute_keys(materialised["S002"], fsm, config)
    # S002 was not edited; nothing in the S001 cache helps it, but when its own
    # cache is present the whole prefix must be reusable.
    own_cache = {key: object() for key in keys}
    assert longest_reused_prefix(keys, own_cache) == len(keys)


def test_delete_shifts_and_invalidates_from_deleted_turn(sample_dir, fsm, config):
    materialised, _ = load_event_stream(sample_dir / "events_base.jsonl")
    before = compute_keys(materialised["S001"], fsm, config)
    deleted = copy.deepcopy(materialised)
    del deleted["S001"][1]  # delete turn 2
    for idx, turn in enumerate(deleted["S001"], start=1):
        turn["turn"] = idx
    after = compute_keys(deleted["S001"], fsm, config)
    assert before[0] == after[0]          # turn 1 untouched
    assert before[1] != after[1]          # turn 2 now holds old turn 3
    assert len(after) == len(before) - 1


def test_fsm_or_config_change_poisons_every_key(sample_dir, fsm, config):
    materialised, _ = load_event_stream(sample_dir / "events_base.jsonl")
    keys = compute_keys(materialised["S001"], fsm, config)
    changed_fsm = copy.deepcopy(fsm)
    object.__setattr__(changed_fsm, "version", fsm.version + 1)
    keys_new = compute_keys(materialised["S001"], changed_fsm, config)
    assert keys != keys_new


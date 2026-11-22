import json

import pytest

from dialog_state_pipeline.config import load_config, load_fsm, parse_config, parse_fsm
from dialog_state_pipeline.errors import PipelineError
from dialog_state_pipeline.loader import load_event_stream, parse_event_stream


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_materialize_append_replace_delete(tmp_path):
    rows = [
        {"session_id": "S1", "op": "append", "turn": 1, "action": "start",
         "slot_updates": [], "batch_id": "B1"},
        {"session_id": "S1", "op": "append", "turn": 2, "action": "provide",
         "slot_updates": [], "batch_id": "B1"},
        {"session_id": "S1", "op": "append", "turn": 3, "action": "confirm",
         "slot_updates": [], "batch_id": "B2"},
        {"session_id": "S1", "op": "replace", "turn": 2, "action": "provide",
         "utterance": "edited", "slot_updates": [], "batch_id": "B3"},
        {"session_id": "S1", "op": "delete", "turn": 1, "batch_id": "B3"},
    ]
    path = tmp_path / "events.jsonl"
    _write_jsonl(path, rows)
    materialised, failures = load_event_stream(path)
    assert failures == []
    turns = materialised["S1"]
    assert [t["turn"] for t in turns] == [1, 2]
    assert turns[0]["utterance"] == "edited"
    assert turns[1]["action"] == "confirm"


def test_stream_failures_are_collected_not_raised(tmp_path):
    rows = [
        "{not valid json",
        {"session_id": "", "op": "append", "turn": 1},
        {"session_id": "S1", "op": "append", "turn": 1, "batch_id": "B1"},
        {"session_id": "S1", "op": "append", "turn": 3, "batch_id": "B1"},
    ]
    path = tmp_path / "events.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(rows[0] + "\n")
        for row in rows[1:]:
            handle.write(json.dumps(row) + "\n")
    _raw, parse_failures = parse_event_stream(path)
    assert {f.code for f in parse_failures} == {"E_STREAM_INVALID"}
    _mat, mat_failures = load_event_stream(path)
    codes = {f.code if hasattr(f, "code") else f["code"] for f in mat_failures}
    assert codes == {"E_STREAM_INVALID"}


def test_fsm_validation_rejects_bad_definitions():
    with pytest.raises(PipelineError) as exc:
        parse_fsm({"states": [], "initial_state": "a"})
    assert exc.value.code == "E_FSM_INVALID"

    with pytest.raises(PipelineError) as exc:
        parse_fsm({"states": ["a", "b"], "initial_state": "c"})
    assert exc.value.code == "E_FSM_INVALID"

    with pytest.raises(PipelineError) as exc:
        parse_fsm({
            "states": ["a", "b"], "initial_state": "a",
            "slots": {"x": {"type": "missing_type", "strategy": "overwrite"}},
            "transitions": [],
        })
    assert exc.value.code == "E_FSM_INVALID"

    with pytest.raises(PipelineError) as exc:
        parse_fsm({
            "states": ["a", "b"], "initial_state": "a",
            "transitions": [{"id": "R1", "from": "a", "action": "go",
                             "to": "nowhere"}],
        })
    assert exc.value.code == "E_FSM_INVALID"


def test_config_validation():
    with pytest.raises(PipelineError) as exc:
        parse_config({"max_workers": 0, "global_budget": 10,
                      "default_quota": 5})
    assert exc.value.code == "E_CONFIG_INVALID"
    with pytest.raises(PipelineError) as exc:
        parse_config({"max_workers": 2, "global_budget": -1,
                      "default_quota": 5})
    assert exc.value.code == "E_CONFIG_INVALID"


def test_sample_files_load(sample_dir):
    fsm = load_fsm(sample_dir / "fsm.yaml")
    cfg = load_config(sample_dir / "pipeline_config.json")
    materialised, failures = load_event_stream(sample_dir / "events.jsonl")
    assert set(materialised) == {"S001", "S002", "S003", "S004"}
    assert failures == []
    assert fsm.initial_state == "greeting"
    assert cfg.max_workers == 3


def test_malformed_slot_operator_is_stream_invalid(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_jsonl(path, [
        {"session_id": "S1", "op": "append", "turn": 1, "action": "start",
         "slot_updates": ["not-a-dict"], "batch_id": "B1"},
    ])
    _raw, failures = parse_event_stream(path)
    assert _raw == []
    assert all(f.code == "E_STREAM_INVALID" for f in failures)


def test_empty_stream_is_valid(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("")
    materialised, failures = load_event_stream(path)
    assert materialised == {} and failures == []

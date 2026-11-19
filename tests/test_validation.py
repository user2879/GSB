"""业务要求 1：解析与 E_xxx 校验。"""
import json
import os

import pytest

from mwb.errors import WorkbenchError
from mwb.parse import parse_score
from mwb.parse_edit import parse_edit_script
from mwb.parse_params import parse_expression
from mwb.samples import generate_expression, generate_score, generate_edit_script


def _write(tmp_path, name, payload):
    path = os.path.join(tmp_path, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def test_valid_samples_parse(tmp_path):
    score_path = _write(tmp_path, "s.json", generate_score())
    params_path = _write(tmp_path, "p.json", generate_expression())
    edits_path = _write(tmp_path, "e.json", generate_edit_script())
    score = parse_score(score_path)
    assert len(score.notes) == 16
    params = parse_expression(params_path)
    assert params.tempo_curve[0].bpm == 96.0
    edits, concurrency, budget = parse_edit_script(edits_path)
    assert len(edits) == 8 and concurrency == 2 and budget == 2


def test_bad_pitch(tmp_path):
    data = generate_score()
    data["notes"][0]["pitch"] = 200
    with pytest.raises(WorkbenchError) as info:
        parse_score(_write(tmp_path, "s.json", data))
    assert info.value.code == "E_PITCH"


def test_bad_beat(tmp_path):
    data = generate_score()
    data["notes"][0]["start_beat"] = -1
    with pytest.raises(WorkbenchError) as info:
        parse_score(_write(tmp_path, "s.json", data))
    assert info.value.code == "E_BEAT"


def test_bad_duration(tmp_path):
    data = generate_score()
    data["notes"][0]["duration_beats"] = 0
    with pytest.raises(WorkbenchError) as info:
        parse_score(_write(tmp_path, "s.json", data))
    assert info.value.code == "E_BEAT"


def test_tempo_not_increasing(tmp_path):
    data = generate_expression()
    data["tempo_curve"][1]["beat"] = 0
    with pytest.raises(WorkbenchError) as info:
        parse_expression(_write(tmp_path, "p.json", data))
    assert info.value.code == "E_TEMPO"


def test_velocity_non_monotonic(tmp_path):
    data = generate_expression()
    data["velocity_curve"] = [{"x": 0, "y": 10}, {"x": 64, "y": 90}, {"x": 100, "y": 40}, {"x": 127, "y": 80}]
    with pytest.raises(WorkbenchError) as info:
        parse_expression(_write(tmp_path, "p.json", data))
    assert info.value.code == "E_VELOCITY_CURVE"


def test_unknown_edit_op(tmp_path):
    data = generate_edit_script()
    data["edits"].append({"op": "frobnicate", "args": {}})
    with pytest.raises(WorkbenchError) as info:
        parse_edit_script(_write(tmp_path, "e.json", data))
    assert info.value.code == "E_EDIT_OP"


def test_missing_edit_target(tmp_path):
    score = parse_score(_write(tmp_path, "s.json", generate_score()))
    params = parse_expression(_write(tmp_path, "p.json", generate_expression()))
    data = generate_edit_script()
    data["edits"] = [{"op": "pedal_remove", "args": {"pedal_id": "NOPE"}}]
    edits, _, _ = parse_edit_script(_write(tmp_path, "e.json", data))
    from mwb.editengine import validate_targets
    from mwb.expand import expand_score
    placed, _, _ = expand_score(score)
    with pytest.raises(WorkbenchError) as info:
        validate_targets(edits, params, placed)
    assert info.value.code == "E_EDIT_TARGET"


def test_bad_json_emits_parse_error(tmp_path):
    path = os.path.join(tmp_path, "broken.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{not json")
    with pytest.raises(WorkbenchError) as info:
        parse_score(path)
    assert info.value.code == "E_PARSE"

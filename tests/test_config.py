from __future__ import annotations

import json

import pytest

from tcp_workbench.config import load_session_config
from tcp_workbench.errors import E_CONFIG, E_SCRIPT, BusinessError
from tests.conftest import SAMPLES


def test_valid_sample_loads_all_documents():
    config, script = load_session_config(SAMPLES / "session.json")
    assert config.config_version == 1
    assert config.first_data_seq == 1001
    assert script.script_version == 1
    assert len(script.responses) == 12
    assert set(script.response_map) == {
        response.for_seq for response in script.responses
    }


def test_invalid_config_is_rejected_wholesale_with_field_paths():
    with pytest.raises(BusinessError) as shown:
        load_session_config(SAMPLES / "invalid" / "session_bad.json")
    error = shown.value
    assert error.code == E_CONFIG
    assert "$.session_id" in error.fields
    assert "$.config_version" in error.fields
    assert "$.budget.max_events" in error.fields
    assert "$.budget.max_elapsed_ms" in error.fields
    assert "$.congestion.rtt_scale" in error.fields
    assert "$.initial_seq" in error.fields
    assert error.fields == sorted(error.fields)


def test_peer_script_negative_delay_and_unreachable_seq_rejected(tmp_path):
    script = {
        "schema_version": 1,
        "script_version": 1,
        "default_policy": "ack",
        "default_delay_ms": 0,
        "responses": [
            {"for_seq": 1501, "action": "ack", "ack_seq": 2001, "delay_ms": -1},
            {"for_seq": 9999, "action": "ack", "ack_seq": 9999, "delay_ms": 0},
        ],
    }
    path = tmp_path / "bad_script.json"
    path.write_text(json.dumps(script), encoding="utf-8")
    with pytest.raises(BusinessError) as shown:
        load_session_config(SAMPLES / "session.json", peer_script_override=path)
    assert shown.value.code == E_CONFIG
    assert any(
        "budget.max_sent_bytes" in field or "for_seq" in field
        for field in shown.value.fields
    )
    assert any("0].delay_ms" in field for field in shown.value.fields)


def test_injected_script_error_uses_e_script():
    with pytest.raises(BusinessError) as shown:
        load_session_config(
            SAMPLES / "session.json",
            peer_script_override=SAMPLES / "invalid" / "peer_script_corrupt.json",
            script_error_code=E_SCRIPT,
        )
    assert shown.value.code == E_SCRIPT
    assert shown.value.fields


def test_duplicate_json_key_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
    with pytest.raises(BusinessError) as shown:
        load_session_config(path)
    assert shown.value.code == E_CONFIG

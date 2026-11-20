import pytest

from cfgdist.errors import BizError, ErrorCode
from cfgdist.manifest import parse_manifest


def _err(text):
    with pytest.raises(BizError) as exc_info:
        parse_manifest(text)
    return exc_info.value


def test_v1_v2_v3_valid():
    v1 = '{"format_version": 1, "entries": [{"key": "k", "value": 1}]}'
    v2 = ('{"format_version": 2, "entries": [{"key": "k", "value": 1, '
          '"node": "n", "config_version": 2}]}')
    v3 = ('{"format_version": 3, "entries": [{"key": "k", "value": 1, '
          '"node": "n", "config_version": 3, '
          '"strategy": {"mode": "targeted", "retries": 1, "timeout_ms": 500}}], '
          '"derivations": []}')
    assert parse_manifest(v1)["format_version"] == 1
    assert parse_manifest(v2)["format_version"] == 2
    assert parse_manifest(v3)["format_version"] == 3


def test_unknown_version_rejected():
    error = _err('{"format_version": 9, "entries": [{"key": "k", "value": 1}]}')
    assert error.code == ErrorCode.E_FORMAT
    assert error.details["field_path"] == "$.format_version"


def test_missing_required_field_rejected_with_path():
    error = _err('{"format_version": 2, "entries": [{"key": "k", "value": 1}]}')
    assert error.code == ErrorCode.E_FORMAT
    assert error.details["field_path"] == "$.entries[0].node"


def test_unsupported_value_type_rejected():
    error = _err('{"format_version": 1, "entries": [{"key": "k", "value": [1, 2]}]}')
    assert error.code == ErrorCode.E_FORMAT
    assert error.details["field_path"] == "$.entries[0].value"


def test_duplicate_json_key_detected():
    text = '{"format_version": 1, "entries": [{"key": "k", "value": 1, "value": 2}]}'
    error = _err(text)
    assert error.code == ErrorCode.E_FORMAT
    assert error.details["field_path"] == "$.entries[0].value"
    assert error.details["reason"] == "duplicate key"


def test_duplicate_entry_key_detected():
    text = (
        '{"format_version": 1, "entries": ['
        '{"key": "k", "value": 1}, {"key": "k", "value": 2}]}'
    )
    error = _err(text)
    assert error.code == ErrorCode.E_FORMAT
    assert error.details["field_path"] == "$.entries[1].key"


def test_invalid_json_syntax_rejected():
    error = _err('{"format_version": 1, "entries": ')
    assert error.code == ErrorCode.E_FORMAT


def test_v2_field_rejected_in_v1():
    text = ('{"format_version": 1, "entries": [{"key": "k", "value": 1, '
            '"node": "n"}]}')
    error = _err(text)
    assert error.code == ErrorCode.E_FORMAT
    assert error.details["field_path"] == "$.entries[0].node"


def test_derivation_unknown_input_rejected():
    text = (
        '{"format_version": 3, "entries": [{"key": "k", "value": 1, '
        '"node": "n", "config_version": 3}], '
        '"derivations": [{"name": "d", "inputs": ["missing"], "op": "sum", '
        '"target_node": "n"}]}'
    )
    error = _err(text)
    assert error.code == ErrorCode.E_FORMAT
    assert "inputs[0]" in error.details["field_path"]


def test_bool_is_not_accepted_as_integer():
    text = '{"format_version": true, "entries": []}'
    error = _err(text)
    assert error.code == ErrorCode.E_FORMAT
    assert error.details["field_path"] == "$.format_version"

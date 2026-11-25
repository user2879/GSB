from __future__ import annotations

import copy

import pytest

from shard_diag.errors import DiagnosticError, ErrorCode
from shard_diag.spec import validate_spec

from conftest import make_spec_values


def _raises_spec(values, *, db_id=None, key_column=None, field):
    with pytest.raises(DiagnosticError) as caught:
        validate_spec(values)
    error = caught.value
    assert error.code == ErrorCode.SPEC.value
    assert error.field == field
    if db_id is not None:
        assert error.db_id == db_id
    if key_column is not None:
        assert error.key_column == key_column
    return error


def test_valid_spec_normalizes_ranges_and_versions():
    spec = validate_spec(make_spec_values(2))
    assert spec.format_version == 1
    assert spec.table_versions() == {"audit": (1, 2, 3), "orders": (1, 2, 3)}
    orders_v2 = spec.rule("orders", 2)
    assert orders_v2.boundaries == (0, 80, 220, 450, 800)
    assert orders_v2.shard_for_key(79) == "orders:s0"
    assert orders_v2.shard_for_key(80) == "orders:s1"
    assert spec.previous_rule("orders", 2).route_version == 1
    assert spec.previous_rule("orders", 1) is None


def test_duplicate_database_id_reports_first_invalid_database():
    values = make_spec_values()
    values["databases"][1]["id"] = "db-a"
    error = _raises_spec(values, db_id="db-a", field="id")
    assert "duplicated" in error.message


def test_duplicate_port_is_E_SPEC():
    values = make_spec_values()
    values["databases"][1]["port"] = values["databases"][0]["port"]
    _raises_spec(values, db_id="db-b", field="port")


@pytest.mark.parametrize("field", ["lease_limit", "concurrency"])
def test_non_positive_quota_fields(field):
    values = make_spec_values()
    values["databases"][0][field] = 0
    _raises_spec(values, db_id="db-a", field=field)


def test_non_loopback_host_rejected():
    values = make_spec_values()
    values["databases"][0]["host"] = "0.0.0.0"
    _raises_spec(values, db_id="db-a", field="host")


def test_non_increasing_boundaries_rejected():
    values = make_spec_values()
    values["tables"][1]["boundaries"] = [0, 80, 80, 450, 800]
    _raises_spec(values, key_column="order_id", field="boundaries")


def test_gapped_or_overlapping_segment_ranges_rejected():
    values = make_spec_values()
    values["tables"][1]["boundaries"] = [[0, 80], [90, 220], [220, 450], [450, 800]]
    error = _raises_spec(values, key_column="order_id", field="boundaries")
    assert "gap" in error.message
    values = make_spec_values()
    values["tables"][1]["boundaries"] = [[0, 80], [70, 220], [220, 450], [450, 800]]
    error = _raises_spec(values, key_column="order_id", field="boundaries")
    assert "overlap" in error.message


def test_duplicate_event_sequence_rejected():
    values = make_spec_values()
    values["events"][1]["seq"] = 1
    _raises_spec(values, field="seq")


def test_descending_event_sequence_rejected():
    values = make_spec_values()
    values["events"][2]["seq"] = 0
    _raises_spec(values, field="seq")


def test_negative_event_sequence_rejected():
    values = make_spec_values()
    values["events"][0]["seq"] = -1
    _raises_spec(values, db_id="db-a", key_column="order_id", field="seq")


def test_event_route_key_outside_bound_rule_is_spec_error():
    values = make_spec_values()
    values["events"][0]["route_key"] = 999
    error = _raises_spec(values, db_id="db-a", key_column="order_id", field="route_key")
    assert error.details["event_seq"] == 1


def test_duplicate_table_version_rejected():
    values = make_spec_values()
    values["tables"].append(copy.deepcopy(values["tables"][1]))
    _raises_spec(values, key_column="order_id", field="route_version")


def test_database_must_have_rule_for_all_declared_tables_at_its_version():
    values = make_spec_values()
    values["tables"] = [table for table in values["tables"] if not (table["name"] == "audit" and table["route_version"] == 2)]
    _raises_spec(values, db_id="db-a", field="route_version")

import json

import pytest

from shardbench.config_loader import load_config
from shardbench.errors import BizError, E_CONFIG

SAMPLE = "samples/sharding_config.json"


def load_doc(tmp_path, mutate):
    with open(SAMPLE, encoding="utf-8") as fh:
        doc = json.load(fh)
    mutate(doc)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def expect_econfig(path, field_prefix):
    with pytest.raises(BizError) as ei:
        load_config(str(path))
    assert ei.value.code == E_CONFIG
    assert ei.value.field.startswith(field_prefix), ei.value.field
    return ei.value


def test_valid_sample_loads():
    cfg = load_config(SAMPLE)
    assert cfg.config_version == 1
    assert set(cfg.units) == {
        "orders_by_user", "orders_by_amount", "users_by_id", "order_user_join"}
    assert cfg.units["orders_by_user"].total_slots == 4


def test_invalid_json_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"config_version": 1, oops', encoding="utf-8")
    expect_econfig(path, "$json")


def test_duplicate_table_rejected(tmp_path):
    path = load_doc(tmp_path, lambda d: d["logical_tables"].append(
        {"name": "ORDERS ", "columns": ["x"]}))
    expect_econfig(path, "logical_tables[2].name")


def test_shard_key_must_exist(tmp_path):
    def mutate(d):
        d["routing_units"][0]["shard_keys"] = ["user_id", "no_such_col"]
    path = load_doc(tmp_path, mutate)
    expect_econfig(path, "routing_units[0].shard_keys[1]")


def test_nonpositive_db_count_rejected(tmp_path):
    def mutate(d):
        d["routing_units"][0]["physical_dbs"] = 0
    path = load_doc(tmp_path, mutate)
    expect_econfig(path, "routing_units[0].physical_dbs")


def test_modulus_inconsistent_rejected(tmp_path):
    def mutate(d):
        d["routing_units"][0]["algorithm"]["modulus"] = 7
    path = load_doc(tmp_path, mutate)
    expect_econfig(path, "routing_units[0].algorithm.modulus")


def test_range_boundaries_must_be_monotonic(tmp_path):
    def mutate(d):
        d["routing_units"][1]["algorithm"]["boundaries"] = [100, 90, 1000]
    path = load_doc(tmp_path, mutate)
    expect_econfig(path, "routing_units[1].algorithm.boundaries[1]")


def test_range_boundary_count_must_match_slots(tmp_path):
    def mutate(d):
        d["routing_units"][1]["algorithm"]["boundaries"] = [100, 500]
    path = load_doc(tmp_path, mutate)
    expect_econfig(path, "routing_units[1].algorithm.boundaries")


def test_dependency_cycle_rejected(tmp_path):
    def mutate(d):
        d["routing_units"][0]["depends_on"] = ["order_user_join"]
    path = load_doc(tmp_path, mutate)
    expect_econfig(path, "routing_units")


def test_unknown_dependency_rejected(tmp_path):
    def mutate(d):
        d["routing_units"][3]["depends_on"] = ["ghost_unit"]
    path = load_doc(tmp_path, mutate)
    expect_econfig(path, "routing_units.order_user_join.depends_on")


def test_first_violation_wins(tmp_path):
    def mutate(d):
        d["logical_tables"].append({"name": "orders", "columns": ["x"]})
        d["routing_units"][0]["physical_dbs"] = -1
    path = load_doc(tmp_path, mutate)
    err = expect_econfig(path, "logical_tables")
    assert "duplicate" in err.message

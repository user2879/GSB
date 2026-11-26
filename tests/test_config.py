import json
import os

import pytest

from edge_cache.config import load_config_dir
from edge_cache.errors import ConfigError, E_CONFIG_INVALID


def _write(config_dir, name, payload):
    with open(os.path.join(config_dir, name), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _load_sample(tmp_path):
    base = os.path.join(os.path.dirname(__file__), "..", "samples", "config")
    for name in os.listdir(base):
        with open(os.path.join(base, name), "r", encoding="utf-8") as src:
            with open(os.path.join(tmp_path, name), "w", encoding="utf-8") as dst:
                dst.write(src.read())


def test_sample_config_loads(tmp_path):
    _load_sample(tmp_path)
    config = load_config_dir(str(tmp_path))
    assert len(config.nodes) == 3
    assert {quota.version for quota in config.quotas} == {"q1", "q2", "q3"}
    assert len(config.shards) == 5
    assert config.source_hash


def test_duplicate_node_id_rejects_whole_batch(tmp_path):
    _load_sample(tmp_path)
    with open(os.path.join(tmp_path, "nodes.json"), "r", encoding="utf-8") as handle:
        nodes = json.load(handle)
    nodes["nodes"][1]["node_id"] = nodes["nodes"][0]["node_id"]
    _write(tmp_path, "nodes.json", nodes)
    with pytest.raises(ConfigError) as exc:
        load_config_dir(str(tmp_path))
    assert all(failure.code == E_CONFIG_INVALID for failure in exc.value.failures)
    assert any("duplicate node_id" in failure.message for failure in exc.value.failures)


def test_duplicate_port_rejected(tmp_path):
    _load_sample(tmp_path)
    with open(os.path.join(tmp_path, "nodes.json"), "r", encoding="utf-8") as handle:
        nodes = json.load(handle)
    nodes["nodes"][1]["port"] = nodes["nodes"][0]["port"]
    _write(tmp_path, "nodes.json", nodes)
    with pytest.raises(ConfigError) as exc:
        load_config_dir(str(tmp_path))
    assert any("duplicate loopback port" in failure.message for failure in exc.value.failures)


def test_lease_limit_above_global_concurrency_rejected(tmp_path):
    _load_sample(tmp_path)
    with open(os.path.join(tmp_path, "nodes.json"), "r", encoding="utf-8") as handle:
        nodes = json.load(handle)
    nodes["nodes"][0]["lease_limit"] = nodes["global_concurrency"] + 1
    _write(tmp_path, "nodes.json", nodes)
    with pytest.raises(ConfigError) as exc:
        load_config_dir(str(tmp_path))
    assert any("lease_limit" in failure.message for failure in exc.value.failures)


def test_quota_chain_must_be_monotonic_and_linked(tmp_path):
    _load_sample(tmp_path)
    quotas = [
        {"version": "q1", "parent": None, "seq": 100},
        {"version": "q2", "parent": "q1", "seq": 90},
    ]
    _write(tmp_path, "quotas.json", quotas)
    with pytest.raises(ConfigError) as exc:
        load_config_dir(str(tmp_path))
    assert any("strictly increasing" in failure.message for failure in exc.value.failures)

    quotas = [
        {"version": "q1", "parent": None, "seq": 100},
        {"version": "q2", "parent": "missing", "seq": 200},
    ]
    _write(tmp_path, "quotas.json", quotas)
    with pytest.raises(ConfigError) as exc:
        load_config_dir(str(tmp_path))
    assert any("existing parent" in failure.message for failure in exc.value.failures)


def test_upstream_missing_and_cycle_rejected(tmp_path):
    _load_sample(tmp_path)
    with open(os.path.join(tmp_path, "shards.json"), "r", encoding="utf-8") as handle:
        shards = json.load(handle)
    shards[0]["upstream"] = ["ghost"]
    _write(tmp_path, "shards.json", shards)
    with pytest.raises(ConfigError) as exc:
        load_config_dir(str(tmp_path))
    assert any("upstream shard does not exist" in failure.message for failure in exc.value.failures)

    _load_sample(tmp_path)
    with open(os.path.join(tmp_path, "shards.json"), "r", encoding="utf-8") as handle:
        shards = json.load(handle)
    shards[0]["upstream"] = ["s2"]
    _write(tmp_path, "shards.json", shards)
    with pytest.raises(ConfigError) as exc:
        load_config_dir(str(tmp_path))
    assert any("cycle" in failure.message for failure in exc.value.failures)


def test_enum_and_positive_integer_enforcement(tmp_path):
    _load_sample(tmp_path)
    with open(os.path.join(tmp_path, "nodes.json"), "r", encoding="utf-8") as handle:
        nodes = json.load(handle)
    nodes["nodes"][0]["fixed_point"] = 3
    _write(tmp_path, "nodes.json", nodes)
    with pytest.raises(ConfigError) as exc:
        load_config_dir(str(tmp_path))
    assert any("fixed_point" in failure.message for failure in exc.value.failures)

    _load_sample(tmp_path)
    with open(os.path.join(tmp_path, "shards.json"), "r", encoding="utf-8") as handle:
        shards = json.load(handle)
    shards[0]["size_units"] = 0
    _write(tmp_path, "shards.json", shards)
    with pytest.raises(ConfigError) as exc:
        load_config_dir(str(tmp_path))
    assert any("size_units" in failure.message for failure in exc.value.failures)

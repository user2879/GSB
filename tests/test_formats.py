from __future__ import annotations

import copy
import json

import pytest

from ckptmeta.canonical import load_json
from ckptmeta.errors import InteropError
from ckptmeta.formats import decode_manifest, decode_manifest_data


def _mutate(path, callback):
    raw = load_json(path)
    callback(raw)
    return raw


def test_versions_decode(fixtures):
    v1 = decode_manifest(fixtures.v1_target)
    v2 = decode_manifest(fixtures.v2_target)
    v3 = decode_manifest(fixtures.direct_target)
    assert (v1.version, v2.version, v3.version) == (1, 2, 3)
    assert all(shard.momentum_digest is None for shard in v1.shards)
    assert all(shard.momentum_digest for shard in v2.shards)
    assert all(shard.data_cursor for shard in v3.shards)
    assert v1.rng_state is None and v2.rng_state is not None
    assert v3.parent_manifest is not None


def test_unknown_version_rejected_at_format_version(fixtures):
    raw = _mutate(fixtures.direct_target, lambda value: value.__setitem__("format_version", 9))
    with pytest.raises(InteropError) as raised:
        decode_manifest_data(raw)
    assert raised.value.code == "E_FORMAT"
    assert raised.value.path == "format_version"


def test_boolean_version_is_not_accepted_as_v1(fixtures):
    raw = _mutate(fixtures.v1_target, lambda value: value.__setitem__("format_version", True))
    with pytest.raises(InteropError) as raised:
        decode_manifest_data(raw)
    assert raised.value.code == "E_FORMAT"
    assert raised.value.path == "format_version"


def test_missing_field_rejected_with_path(fixtures):
    raw = _mutate(fixtures.v2_target, lambda value: value.pop("rng_state"))
    with pytest.raises(InteropError) as raised:
        decode_manifest_data(raw)
    assert raised.value.code == "E_FORMAT"
    assert raised.value.path == "rng_state"


def test_invalid_digest_rejected_with_shard_path(fixtures):
    raw = _mutate(
        fixtures.v1_target,
        lambda value: value["shards"][1].__setitem__("parameter_digest", "not-hex"),
    )
    with pytest.raises(InteropError) as raised:
        decode_manifest_data(raw)
    assert raised.value.code == "E_FORMAT"
    assert raised.value.path == "shards[1].parameter_digest"
    assert "shards[1].parameter_digest" in raised.value.details["field"]


def test_shard_count_mismatch_rejected(fixtures):
    raw = _mutate(fixtures.v1_target, lambda value: value.__setitem__("shard_count", 5))
    with pytest.raises(InteropError) as raised:
        decode_manifest_data(raw)
    assert raised.value.code == "E_FORMAT"
    assert raised.value.path == "shards"


def test_parent_field_missing_is_format_error(fixtures):
    raw = _mutate(fixtures.direct_target, lambda value: value.pop("parent_manifest"))
    with pytest.raises(InteropError) as raised:
        decode_manifest_data(raw)
    assert raised.value.code == "E_FORMAT"
    assert raised.value.path == "parent_manifest"


def test_null_parent_for_non_genesis_is_chain_error(fixtures):
    raw = _mutate(fixtures.direct_target, lambda value: value.__setitem__("parent_manifest", None))
    with pytest.raises(InteropError) as raised:
        decode_manifest_data(raw)
    assert raised.value.code == "E_CHAIN"
    assert raised.value.path == "parent_manifest"


def test_unknown_field_is_rejected(fixtures):
    raw = _mutate(fixtures.v1_target, lambda value: value.__setitem__("surprise", 1))
    with pytest.raises(InteropError) as raised:
        decode_manifest_data(raw)
    assert raised.value.code == "E_FORMAT"
    assert raised.value.path == "surprise"

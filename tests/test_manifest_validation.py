"""Requirement 1: strong manifest/registry validation with E_MANIFEST."""

import pytest

from config_regression.errors import E_MANIFEST, ManifestError
from config_regression.model import (
    load_manifest,
    load_registry,
    assert_version_monotonic,
)


def _registry(node_ids=("node-a", "node-b")):
    return load_registry({"nodes": [
        {"node_id": node_id, "host": "127.0.0.1", "port": 0}
        for node_id in node_ids]})


def _item(key, value=1.0, targets=("node-a",), derived_from=None):
    return {"key": key, "value": value, "targets": list(targets),
            "derived_from": derived_from or []}


def test_valid_manifest_loads_and_topological_order():
    registry = _registry()
    data = {"version": 2, "predecessor_version": 1, "items": [
        _item("child", derived_from=[{"key": "parent", "coefficient": 1.0}]),
        _item("parent")]}
    manifest = load_manifest(data, registry)
    assert manifest.order.index("parent") < manifest.order.index("child")


def test_illegal_key_boolean_value_and_unknown_target_are_listed():
    registry = _registry()
    data = {"version": 1, "items": [
        {"key": "1bad", "value": 1, "targets": ["node-a"]},
        {"key": "ok", "value": True, "targets": ["ghost"]}]}
    with pytest.raises(ManifestError) as exc:
        load_manifest(data, registry)
    assert exc.value.code == E_MANIFEST
    subjects = {v["subject"] for v in exc.value.details()}
    assert any("1bad" in s for s in subjects)
    assert any("ok.value" in s for s in subjects)
    assert any("ghost" in s for s in subjects)
    # stable ordering of violations
    assert exc.value.details() == sorted(exc.value.details(),
                                         key=lambda d: d["subject"])


def test_missing_derivation_target_and_cycle_are_rejected():
    registry = _registry()
    data = {"version": 1, "items": [
        _item("a", derived_from=[{"key": "missing", "coefficient": 1.0}]),
        _item("x", derived_from=[{"key": "y", "coefficient": 1.0}]),
        _item("y", derived_from=[{"key": "x", "coefficient": 1.0}])]}
    with pytest.raises(ManifestError) as exc:
        load_manifest(data, registry)
    reasons = " ".join(v["reason"] for v in exc.value.details())
    assert "does not exist" in reasons
    assert "cycle" in reasons


def test_version_must_increase_monotonically():
    registry = _registry()
    data = {"version": 3, "predecessor_version": 3,
            "items": [_item("k")]}
    with pytest.raises(ManifestError):
        load_manifest(data, registry)
    base = load_manifest({"version": 1, "items": [_item("k")]}, registry)
    cand = load_manifest({"version": 1, "items": [_item("k")]}, registry)
    with pytest.raises(ManifestError):
        assert_version_monotonic(base, cand)


def test_registry_rejects_bad_port_and_duplicate_node():
    with pytest.raises(ManifestError):
        load_registry({"nodes": [
            {"node_id": "n1", "port": 70000},
            {"node_id": "n1", "port": 0}]})


def test_unsupported_numeric_type_and_strategy_rejected():
    registry = _registry()
    data = {"version": 1, "items": [
        {"key": "k", "value": "1.0", "targets": ["node-a"],
         "strategy": "mystery"}]}
    with pytest.raises(ManifestError) as exc:
        load_manifest(data, registry)
    subjects = {v["subject"] for v in exc.value.details()}
    assert any(s.endswith("k.value") for s in subjects)
    assert any(s.endswith("k.strategy") for s in subjects)

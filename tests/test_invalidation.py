from __future__ import annotations

import copy

from tunnel_interop.fragmenter import build_bundle
from tunnel_interop.invalidation import evolve_bundle
from tunnel_interop.spec import load_spec_mapping


def _bumped(data):
    data["versions"]["declaration"] += 1
    return load_spec_mapping(data, source_path="<evolved>")


def test_route_priority_and_endpoint_mtu_only_invalidate_dependent_message(spec, spec_data):
    old_bundle = build_bundle(spec)
    changed = copy.deepcopy(spec_data)
    changed["versions"]["declaration"] += 1
    changed["versions"]["endpoint_set"] += 1
    changed["versions"]["route_table"] += 1
    changed["endpoints"][0]["mtu"] -= 20
    changed["routes"][0]["priority"] += 1
    new_spec = load_spec_mapping(changed, source_path="<evolved>")
    result = evolve_bundle(old_bundle, old_spec=spec, new_spec=new_spec)
    assert result.invalid_reassemblies == ("tun-0",)
    assert {item["message_key"] for item in result.invalid_fragments} == {"tun-0"}
    assert all(result.unchanged_checks.values())
    old_other = old_bundle.message_map["tun-1"]
    new_other = result.bundle.message_map["tun-1"]
    assert [fragment.wire for fragment in old_other.fragments] == [fragment.wire for fragment in new_other.fragments]
    assert [fragment.fragment_version for fragment in old_other.fragments] == [
        fragment.fragment_version for fragment in new_other.fragments
    ]
    assert any(step.reason == "ENDPOINT_CHANGED" for step in result.steps)
    assert any(step.reason == "ROUTE_CHANGED" for step in result.steps)


def test_fragment_size_change_invalidates_all_fragments_and_reassemblies(spec, spec_data):
    old_bundle = build_bundle(spec)
    changed = copy.deepcopy(spec_data)
    changed["versions"]["declaration"] += 1
    changed["versions"]["fragment_policy"] += 1
    changed["fragment_policy"]["fragment_size"] = 128
    new_spec = load_spec_mapping(changed, source_path="<frags>")
    result = evolve_bundle(old_bundle, old_spec=spec, new_spec=new_spec)
    assert result.invalid_reassemblies == ("tun-0", "tun-1")
    assert {item["message_key"] for item in result.invalid_fragments} == {"tun-0", "tun-1"}
    assert len(result.bundle.all_fragments) > len(old_bundle.all_fragments)


def test_reassembly_timeout_change_invalidates_reassembly_not_fragment_wire(spec, spec_data):
    old_bundle = build_bundle(spec)
    changed = copy.deepcopy(spec_data)
    changed["versions"]["declaration"] += 1
    changed["versions"]["fragment_policy"] += 1
    changed["fragment_policy"]["reassembly_timeout_ms"] += 1
    new_spec = load_spec_mapping(changed, source_path="<timeout>")
    result = evolve_bundle(old_bundle, old_spec=spec, new_spec=new_spec)
    assert result.invalid_fragments == ()
    assert result.invalid_reassemblies == ("tun-0", "tun-1")
    assert all(result.unchanged_checks.values())
    for key in ("tun-0", "tun-1"):
        assert result.bundle.message_map[key].reassembly.reassembly_version == 2


def test_added_and_removed_routes_are_listed_stably(spec, spec_data):
    old_bundle = build_bundle(spec)
    changed = copy.deepcopy(spec_data)
    changed["versions"]["declaration"] += 1
    changed["versions"]["route_table"] += 1
    changed["routes"] = changed["routes"][:1]
    changed["routes"][0]["tunnel_id"] = "tun-renamed"
    new_spec = load_spec_mapping(changed, source_path="<routes>")
    result = evolve_bundle(old_bundle, old_spec=spec, new_spec=new_spec)
    assert result.added_messages == ("tun-renamed",)
    assert result.removed_messages == ("tun-0", "tun-1")

from __future__ import annotations

import copy

import pytest

from tunnel_interop.errors import E_REF, E_SPEC, TunnelError
from tunnel_interop.spec import load_spec_mapping, validate_versions_monotonic


def assert_error(code: str, path: str, function) -> TunnelError:
    with pytest.raises(TunnelError) as captured:
        function()
    assert captured.value.code == code
    assert captured.value.path == path
    return captured.value


def test_valid_spec_loads(spec):
    assert len(spec.endpoints) == 2
    assert len(spec.routes) == 2
    assert spec.effective_payload_limit == 1320


def test_endpoint_id_must_be_unique(spec_data):
    spec_data["endpoints"][1]["id"] = spec_data["endpoints"][0]["id"]
    assert_error(E_SPEC, "$.endpoints[1].id", lambda: load_spec_mapping(spec_data))


@pytest.mark.parametrize("port", [0, 65536])
def test_port_range(spec_data, port):
    spec_data["endpoints"][0]["port"] = port
    assert_error(E_SPEC, "$.endpoints[0].port", lambda: load_spec_mapping(spec_data))


def test_ports_must_be_globally_unique(spec_data):
    spec_data["endpoints"][1]["port"] = spec_data["endpoints"][0]["port"]
    assert_error(E_SPEC, "$.endpoints[1].port", lambda: load_spec_mapping(spec_data))


@pytest.mark.parametrize(
    ("field", "value", "suffix"),
    [
        ("mtu", 0, "mtu"),
        ("mtu", True, "mtu"),
        ("encapsulation_overhead", 0, "encapsulation_overhead"),
        ("replay_window", 0, "replay_window"),
        ("seed", "bad", "seed"),
    ],
)
def test_endpoint_integer_fields(spec_data, field, value, suffix):
    spec_data["endpoints"][0][field] = value
    assert_error(E_SPEC, f"$.endpoints[0].{suffix}", lambda: load_spec_mapping(spec_data))


def test_overhead_must_be_smaller_than_mtu(spec_data):
    spec_data["endpoints"][0]["encapsulation_overhead"] = spec_data["endpoints"][0]["mtu"]
    assert_error(
        E_SPEC,
        "$.endpoints[0].encapsulation_overhead",
        lambda: load_spec_mapping(spec_data),
    )


def test_route_prefix_must_be_valid(spec_data):
    spec_data["routes"][0]["prefix"] = "not-a-prefix"
    assert_error(E_SPEC, "$.routes[0].prefix", lambda: load_spec_mapping(spec_data))


def test_route_next_hop_must_exist(spec_data):
    spec_data["routes"][0]["next_hop"] = "missing"
    assert_error(E_SPEC, "$.routes[0].next_hop", lambda: load_spec_mapping(spec_data))


def test_route_reference_error_code_is_injectable(spec_data):
    spec_data["routes"][0]["next_hop"] = "missing"
    assert_error(
        E_REF,
        "$.routes[0].next_hop",
        lambda: load_spec_mapping(spec_data, reference_error_code=E_REF),
    )


def test_tunnel_id_must_be_unique(spec_data):
    spec_data["routes"][1]["tunnel_id"] = spec_data["routes"][0]["tunnel_id"]
    assert_error(E_SPEC, "$.routes[1].tunnel_id", lambda: load_spec_mapping(spec_data))


def test_fragment_size_must_fit_effective_payload(spec_data):
    spec_data["fragment_policy"]["fragment_size"] = 1321
    assert_error(
        E_SPEC,
        "$.fragment_policy.fragment_size",
        lambda: load_spec_mapping(spec_data),
    )


@pytest.mark.parametrize("field", ["max_retries", "reassembly_timeout_ms"])
def test_fragment_nonnegative_fields(spec_data, field):
    spec_data["fragment_policy"][field] = -1
    assert_error(E_SPEC, f"$.fragment_policy.{field}", lambda: load_spec_mapping(spec_data))


def test_quota_must_not_exceed_global_limit(spec_data):
    spec_data["runtime"]["per_tunnel_inflight_quota"] = spec_data["runtime"]["global_inflight_limit"] + 1
    assert_error(
        E_SPEC,
        "$.runtime.per_tunnel_inflight_quota",
        lambda: load_spec_mapping(spec_data),
    )


def test_first_violation_is_reported(spec_data):
    spec_data["endpoints"][0]["port"] = 0
    spec_data["endpoints"][0]["mtu"] = 0
    assert_error(E_SPEC, "$.endpoints[0].port", lambda: load_spec_mapping(spec_data))


def test_changed_content_requires_version_increment(spec, spec_data):
    changed = copy.deepcopy(spec_data)
    changed["routes"][0]["priority"] += 1
    new = load_spec_mapping(changed)
    assert_error(E_SPEC, "$.versions.declaration", lambda: validate_versions_monotonic(spec, new))


def test_versions_must_be_monotonic(spec, spec_data):
    old_data = copy.deepcopy(spec_data)
    old_data["versions"]["declaration"] = 2
    old = load_spec_mapping(old_data)
    new = load_spec_mapping(spec_data)
    assert_error(E_SPEC, "$.versions.declaration", lambda: validate_versions_monotonic(old, new))

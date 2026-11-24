from __future__ import annotations

import copy
import dataclasses

import pytest

from tunnel_interop.errors import E_FIELD, TunnelError
from tunnel_interop.fragmenter import build_bundle, fragment_message, reassemble_fragments
from tunnel_interop.samples import generate_messages, generate_payload, ordered_routes
from tunnel_interop.spec import load_spec_mapping
from tunnel_interop.util import sha256_hex


def test_sample_generation_is_exactly_recomputable(spec):
    for plan in generate_messages(spec):
        assert generate_payload(plan.rng_seed, len(plan.payload)) == plan.payload
        assert plan.payload_sha256 == sha256_hex(plan.payload)
        assert len(plan.payload) == spec.samples.bytes_per_message


def test_order_of_endpoints_and_routes_does_not_change_wire(spec, spec_data):
    reordered = copy.deepcopy(spec_data)
    reordered["endpoints"] = list(reversed(reordered["endpoints"]))
    reordered["routes"] = list(reversed(reordered["routes"]))
    other = load_spec_mapping(reordered, source_path="<reordered>")
    first = build_bundle(spec)
    second = build_bundle(other)
    assert [(m.plan.message_key, m.plan.message_id) for m in first.messages] == [
        (m.plan.message_key, m.plan.message_id) for m in second.messages
    ]
    assert [(f.fragment_index, f.fragment_offset, f.wire_sha256) for f in first.all_fragments] == [
        (f.fragment_index, f.fragment_offset, f.wire_sha256) for f in second.all_fragments
    ]


def test_fragment_offsets_lengths_and_metadata(spec):
    bundle = build_bundle(spec)
    for message in bundle.messages:
        expected_offsets = list(range(0, len(message.plan.payload), message.plan.fragment_size))
        assert [fragment.fragment_offset for fragment in message.fragments] == expected_offsets
        assert [fragment.fragment_index for fragment in message.fragments] == list(range(len(expected_offsets)))
        for fragment in message.fragments:
            assert fragment.fragment_length <= spec.fragment_policy.fragment_size
            assert fragment.total_fragments == len(message.fragments)
            assert fragment.normalized.source_format == bundle.target_format
            assert fragment.format_variants.keys() == {"f1", "f2", "f3"}
        assert message.reassembly.payload == message.plan.payload


def test_reverse_arrival_reassembles(spec):
    message = build_bundle(spec).messages[0]
    fragments = [item.normalized for item in reversed(message.fragments)]
    result = reassemble_fragments(
        fragments,
        expected_message_key=message.plan.message_key,
        expected_sha256=message.plan.payload_sha256,
    )
    assert result.payload == message.plan.payload
    assert result.arrival_indices == tuple(reversed([item.fragment_index for item in message.fragments]))


def test_duplicate_arrival_reassembles_and_records_duplicates(spec):
    message = build_bundle(spec).messages[0]
    fragments = []
    for item in message.fragments:
        fragments.extend([item.normalized, item.normalized])
    result = reassemble_fragments(
        fragments,
        expected_message_key=message.plan.message_key,
        expected_sha256=message.plan.payload_sha256,
    )
    assert result.payload == message.plan.payload
    assert len(result.duplicate_indices) == len(message.fragments)


def test_conflicting_duplicate_is_rejected(spec):
    message = build_bundle(spec).messages[0]
    original = message.fragments[0].normalized
    conflict = dataclasses.replace(original, payload=b"x" * len(original.payload))
    with pytest.raises(TunnelError) as captured:
        reassemble_fragments(
            [original, conflict, *[item.normalized for item in message.fragments[1:]]],
            expected_message_key=message.plan.message_key,
        )
    assert captured.value.code == E_FIELD
    assert "conflicting" in captured.value.message


def test_incomplete_fragment_set_is_rejected(spec):
    message = build_bundle(spec).messages[0]
    with pytest.raises(TunnelError) as captured:
        reassemble_fragments(
            [item.normalized for item in message.fragments[:-1]],
            expected_message_key=message.plan.message_key,
        )
    assert captured.value.code == E_FIELD
    assert captured.value.path == "reassembly.fragments"

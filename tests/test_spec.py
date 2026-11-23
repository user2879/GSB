from __future__ import annotations

import copy

import pytest

from ossdiag.errors import SpecError
from ossdiag.models import load_spec_data

from conftest import KiB, MiB, object_doc, spec_doc


def test_valid_spec_is_fully_loaded() -> None:
    spec = load_spec_data(spec_doc(objects=[object_doc(key="a", total_bytes=5 * KiB, shard_size=64 * KiB)]))
    item = spec.objects[0]
    assert item.shard_count == 1
    assert item.shard_seeds == (100,)
    assert item.batch_size == 64 * KiB


@pytest.mark.parametrize(
    ("mutator", "field"),
    [
        (lambda doc: doc["objects"][0].__setitem__("total_bytes", 0), "total_bytes"),
        (lambda doc: doc["objects"][0].__setitem__("total_bytes", "10"), "total_bytes"),
        (lambda doc: doc["objects"][0].__setitem__("shard_size", 12345), "shard_size"),
        (lambda doc: doc["objects"][0].__setitem__("shard_seeds", [1]), "shard_seeds"),
        (lambda doc: doc["objects"][0].__setitem__("shard_seeds", [1, True]), "shard_seeds[1]"),
        (lambda doc: doc["objects"][0].__setitem__("object_key", ""), "object_key"),
        (
            lambda doc: doc["objects"][0]["lease_policy"].__setitem__("renew_interval_ms", 500),
            "lease_policy.renew_interval_ms",
        ),
        (
            lambda doc: doc["objects"][0]["lease_policy"].__setitem__("lease_ms", 0),
            "lease_policy.lease_ms",
        ),
        (lambda doc: doc["objects"][0].__setitem__("batch_size", 12345), "batch_size"),
        (lambda doc: doc["objects"][0].__setitem__("concurrency_limit", 0), "concurrency_limit"),
    ],
)
def test_e_spec_rejects_first_bad_field(mutator, field: str) -> None:
    doc = spec_doc(objects=[object_doc(total_bytes=128 * KiB, shard_size=64 * KiB)])
    mutator(doc)
    with pytest.raises(SpecError) as caught:
        load_spec_data(doc)
    assert caught.value.code == "E_SPEC"
    assert caught.value.field == field
    assert caught.value.object_key == ("$objects[0]" if field == "object_key" else "object/a.bin")


def test_duplicate_object_key_is_rejected_at_whole_document_level() -> None:
    doc = spec_doc(
        objects=[
            object_doc(key="duplicate", total_bytes=64 * KiB),
            object_doc(key="duplicate", total_bytes=128 * KiB),
        ]
    )
    with pytest.raises(SpecError) as caught:
        load_spec_data(doc)
    assert caught.value.field == "object_key"
    assert caught.value.object_key == "duplicate"


def test_text_seed_and_shard_count_disagreement_are_rejected() -> None:
    doc = spec_doc(objects=[object_doc(total_bytes=128 * KiB, shard_size=64 * KiB)])
    doc["objects"][0]["shard_count"] = 1
    with pytest.raises(SpecError) as caught:
        load_spec_data(doc)
    assert caught.value.field == "shard_count"

    doc = spec_doc(objects=[object_doc(total_bytes=128 * KiB, shard_size=64 * KiB)])
    doc["objects"][0]["shard_seeds"][0] = "seed"
    with pytest.raises(SpecError) as caught:
        load_spec_data(doc)
    assert caught.value.field == "shard_seeds[0]"


def test_first_invalid_object_wins() -> None:
    doc = spec_doc(
        objects=[
            object_doc(key="first", total_bytes=0),
            object_doc(key="second", shard_size=999),
        ]
    )
    with pytest.raises(SpecError) as caught:
        load_spec_data(doc)
    assert caught.value.object_key == "first"
    assert caught.value.field == "total_bytes"


def test_session_limits_are_validated() -> None:
    doc = spec_doc(scale_exponent=10)
    with pytest.raises(SpecError) as caught:
        load_spec_data(doc)
    assert caught.value.object_key == "$session"
    assert caught.value.field == "scale_exponent"

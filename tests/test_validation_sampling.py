from __future__ import annotations

import copy

import pytest

from topic_ledger.errors import BusinessError, E_INVALID_INPUT
from topic_ledger.samplegen import build_sample_bundle
from topic_ledger.sampling import compute_shard_receipt
from topic_ledger.validation import (
    validate_document_stream,
    validate_inference,
    validate_topology,
)


def test_generated_bundle_is_valid() -> None:
    bundle = build_sample_bundle()
    topology = validate_topology(bundle["topology"])
    vocab_size = len(bundle["vocab"])
    inference = validate_inference(bundle["inference"], vocab_size=vocab_size)
    documents = validate_document_stream(
        {"vocab": bundle["vocab"], "documents": bundle["documents"]},
        shard_ids=topology["shards"],
        term_ids=set(),
    )
    assert inference["topic_count"] <= vocab_size
    assert len(documents) == len(bundle["documents"])


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda b: b["topology"]["nodes"][1].__setitem__("port", "bad"), "topology.nodes[1].port"),
        (lambda b: b["topology"]["nodes"][1].__setitem__("node_id", "node-a"), "topology.nodes[1].node_id"),
        (lambda b: b["topology"]["nodes"][0].__setitem__("port", b["topology"]["nodes"][1]["port"]), "topology.nodes[1].port"),
        (lambda b: b["inference"].__setitem__("topic_count", 99), "inference.topic_count"),
        (lambda b: b["inference"].__setitem__("sample_rounds", -1), "inference.sample_rounds"),
    ],
)
def test_validation_reports_first_invalid_field(mutate, field) -> None:
    bundle = build_sample_bundle()
    mutate(bundle)
    with pytest.raises(BusinessError) as exc_info:
        if field.startswith("topology"):
            validate_topology(bundle["topology"])
        else:
            validate_inference(bundle["inference"], vocab_size=len(bundle["vocab"]))
    error = exc_info.value
    assert error.code == E_INVALID_INPUT
    assert error.field == field


def test_document_revision_and_vocab_are_validated() -> None:
    bundle = build_sample_bundle()
    sample = {"vocab": bundle["vocab"], "documents": copy.deepcopy(bundle["documents"])}
    sample["documents"][0]["revision"] = 2
    sample["documents"][1]["doc_id"] = sample["documents"][0]["doc_id"]
    # Duplicate identity is rejected before revision semantics in a static set.
    with pytest.raises(BusinessError) as exc_info:
        validate_document_stream(sample, shard_ids={s for s in bundle["topology"]["shards"]}, term_ids=set())
    assert exc_info.value.code == E_INVALID_INPUT
    assert exc_info.value.field.endswith("doc_id")

    bad_revision = copy.deepcopy(bundle)
    bad_revision["documents"][0]["revision"] = 0
    with pytest.raises(BusinessError) as exc_info:
        validate_document_stream(
            {"vocab": bad_revision["vocab"], "documents": bad_revision["documents"]},
            shard_ids=set(bad_revision["topology"]["shards"]),
            term_ids=set(),
        )
    assert exc_info.value.field == "sample.documents[0].revision"


def test_fixed_point_sampler_is_deterministic_and_closed() -> None:
    bundle = build_sample_bundle()
    request = {
        "node_id": "node-a",
        "shard_id": "shard-00",
        "term_counts": [[0, 4], [2, 3], [7, 1]],
        "document_refs": [{"doc_id": "doc-000", "revision": 1}],
        "config": bundle["inference"],
        "vocab_size": len(bundle["vocab"]),
        "monotonic_clock_ms": 12345,
    }
    first = compute_shard_receipt(request)
    second = compute_shard_receipt(copy.deepcopy(request))
    assert first == second
    assert first["monotonic_ms"] == 12345
    counts = first["soft_summary"]["topic_counts"]
    assert sum(counts) == first["total_tokens"] == 8
    assert sum(first["soft_summary"]["soft_weights_fixed"]) == bundle["inference"]["fixed_point"]["scale"]
    pairs = [(entry["term_id"], entry["topic_id"]) for entry in first["term_topic_counts"]]
    assert pairs == sorted(pairs)
    assert len(first["cache_keys"]["rounds"]) == bundle["inference"]["sample_rounds"]

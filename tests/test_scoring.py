from __future__ import annotations

from gsb_index.indexing import compute_shard_summary
from gsb_index.models import (
    CompressionProfile,
    Document,
    Query,
    ScoringProfile,
    ShardSpec,
    Vocabulary,
)
from gsb_index.scoring import recall_shard


def _summary(bit_width: int, quantum_step: int) -> dict:
    vocabulary = Vocabulary(("index", "shard"), frozenset())
    shard = ShardSpec("shard-s", 19001, 64, 1, 7, 1)
    profile = CompressionProfile(4, (1.0, 1.0, 1.0, 1.0), "zigzag-varint-v1")
    documents = [
        Document(
            "doc-z",
            1,
            "shard-s",
            {"title": tuple(["index"] * 100), "body": ()},
            {"title": False, "body": True},
        ),
        Document(
            "doc-a",
            1,
            "shard-s",
            {"title": tuple(["index"] * 20), "body": ()},
            {"title": False, "body": True},
        ),
        Document(
            "doc-b",
            1,
            "shard-s",
            {"title": ("index",), "body": ("shard",)},
            {"title": False, "body": True},
        ),
    ]
    return compute_shard_summary(shard, profile, vocabulary, documents, 1)


def test_bit_width_changes_intermediate_scores_and_order():
    summary = _summary(bit_width=8, quantum_step=1)
    query = Query("q", ("index",), top_k=2)
    low = recall_shard(
        summary,
        query,
        ScoringProfile(
            bit_width=8,
            fraction_bits=6,
            quantum_step=1,
            title_weight=5,
            body_weight=1,
            tf_saturation_k=1,
        ),
    )
    high = recall_shard(
        summary,
        query,
        ScoringProfile(
            bit_width=32,
            fraction_bits=12,
            quantum_step=1,
            title_weight=5,
            body_weight=1,
            tf_saturation_k=1,
        ),
    )
    low_selected = [item["doc_id"] for item in low["selected"]]
    high_selected = [item["doc_id"] for item in high["selected"]]
    assert low["ranked"][0]["quantized_score"] == (1 << 8) - 1
    assert [item["raw_score"] for item in low["ranked"][:2]] == [255, 255]
    assert low_selected[0] == "doc-a"
    assert high_selected[0] == "doc-z"
    assert low_selected != high_selected
    assert high["ranked"][0]["score_components"][0]["title_tf_fp"] != low["ranked"][0]["score_components"][0]["title_tf_fp"]


def test_quantization_boundary_and_tie_breaks():
    summary = _summary(bit_width=32, quantum_step=1)
    query = Query("q", ("index",), top_k=2)
    unquantized = recall_shard(
        summary,
        query,
        ScoringProfile(bit_width=32, fraction_bits=12, quantum_step=1, title_weight=5, body_weight=1, tf_saturation_k=1),
    )
    coarse = recall_shard(
        summary,
        query,
        ScoringProfile(bit_width=32, fraction_bits=12, quantum_step=1000, title_weight=5, body_weight=1, tf_saturation_k=1),
    )
    assert [item["quantized_score"] for item in unquantized["selected"]] != [
        item["quantized_score"] for item in coarse["selected"]
    ]
    assert any(item["boundary"] for item in coarse["ranked"])
    # Equal contents produce equal quantized scores; doc_id breaks the tie ascending.
    tied_summary = compute_shard_summary(
        ShardSpec("shard-t", 19001, 64, 1, 7, 1),
        CompressionProfile(4, (1.0, 1.0, 1.0, 1.0), "plain-int-v1"),
        Vocabulary(("index",), frozenset()),
        [
            Document("doc-zzz", 2, "shard-t", {"title": ("index",), "body": ()}, {"title": False, "body": True}),
            Document("doc-aaa", 1, "shard-t", {"title": ("index",), "body": ()}, {"title": False, "body": True}),
        ],
        1,
    )
    tied = recall_shard(
        tied_summary,
        Query("q", ("index",), top_k=2),
        ScoringProfile(bit_width=32, fraction_bits=12, quantum_step=1),
    )
    assert [item["doc_id"] for item in tied["selected"]] == ["doc-aaa", "doc-zzz"]


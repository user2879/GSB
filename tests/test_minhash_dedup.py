from __future__ import annotations

from corpus_workbench.config import DedupConfig
from corpus_workbench.dedup import deduplicate
from corpus_workbench.hashing import compute_signature, signature_cache_key
from corpus_workbench.models import Document, SignatureRecord, content_fingerprint


def _record(source: str, doc_id: str, text: str, config: DedupConfig) -> SignatureRecord:
    signature, cost = compute_signature(text, config)
    return SignatureRecord(
        source=source,
        doc_id=doc_id,
        fingerprint=content_fingerprint(text),
        shingle_length=config.shingle_length,
        signature_dim=config.signature_dim,
        point_cost=cost,
        signature=signature,
        cache_key=signature_cache_key(text, config),
    )


def test_minhash_groups_and_weighted_representative(tmp_path):
    config = DedupConfig(
        shingle_length=3,
        signature_dim=256,
        jaccard_threshold=0.5,
        representative_rule="source_weight_desc_doc_id_asc",
    )
    low = _record(
        "low",
        "a-0001",
        "alpha beta gamma delta epsilon zeta eta theta iota",
        config,
    )
    high = _record(
        "high",
        "z-9999",
        "alpha beta gamma delta epsilon zeta eta theta kappa",
        config,
    )
    other = _record(
        "high",
        "a-other",
        "one two three four five six seven eight nine ten",
        config,
    )
    result = deduplicate([low, high, other], {"low": 1.0, "high": 2.0}, config)

    assert result.input_document_count == 3
    assert result.output_document_count == 2
    assert len(result.groups) == 1
    group = result.groups[0]
    assert group["representative"]["doc_id"] == "z-9999"
    assert len(group["removed"]) == 1
    assert group["removed"][0]["doc_id"] == "a-0001"
    assert group["removed"][0]["representative_id"] == "z-9999"
    assert group["removed"][0]["similarity"] >= 0.5
    assert {item["doc_id"] for item in result.retained_documents} == {"z-9999", "a-other"}


def test_representative_tie_breaks_by_doc_id():
    config = DedupConfig(shingle_length=2, signature_dim=128, jaccard_threshold=0.4)
    left = _record("same", "b-0002", "red green blue yellow black white", config)
    right = _record("same", "a-0001", "red green blue yellow black gray", config)
    result = deduplicate([left, right], {"same": 1.0}, config)
    assert result.groups[0]["representative"]["doc_id"] == "a-0001"


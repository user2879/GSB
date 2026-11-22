from __future__ import annotations

from iidx.compression import (
    block_fingerprint,
    compare_decoded_block,
    decode_block,
    encode_block,
    encode_uvarint,
    read_uvarint,
)


def _block():
    terms = [
        {
            "term": "alpha",
            "doc_freq": 3,
            "postings": [
                {"doc_id": "doc-001", "positions": [0, 7, 19]},
                {"doc_id": "doc-003", "positions": [2]},
                {"doc_id": "doc-008", "positions": [4, 12]},
            ],
        },
        {
            "term": "beta",
            "doc_freq": 1,
            "postings": [{"doc_id": "doc-003", "positions": [1, 10]}],
        },
        {
            "term": "gamma",
            "doc_freq": 2,
            "postings": [
                {"doc_id": "doc-001", "positions": [5]},
                {"doc_id": "doc-008", "positions": [0, 3]},
            ],
        },
    ]
    return {
        "block_no": 7,
        "shard": "shard-a",
        "local_block_no": 2,
        "terms": terms,
        "doc_count": 3,
        "position_count": 11,
        "fingerprint": block_fingerprint(terms),
    }


def test_uvarint_roundtrip() -> None:
    for value in [0, 1, 2, 127, 128, 255, 300, 16_384, 2**32, 2**63 - 1]:
        encoded = encode_uvarint(value)
        decoded, offset = read_uvarint(encoded, 0)
        assert decoded == value
        assert offset == len(encoded)


def test_two_tiers_have_different_lengths_and_exact_roundtrip() -> None:
    block = _block()
    tier1, stats1 = encode_block(block, 1, low_frequency_threshold=1)
    tier2, stats2 = encode_block(block, 2, low_frequency_threshold=1)
    assert len(tier1) != len(tier2)
    assert stats2["filtered_terms"] == 1
    assert stats2["filtered_term_names"] == ["beta"]
    decoded1, decoded_stats1 = decode_block(tier1)
    decoded2, decoded_stats2 = decode_block(tier2)
    assert compare_decoded_block(block, decoded1)[0]
    assert compare_decoded_block(block, decoded2)[0]
    assert decoded_stats1["decoded_checksum"] == decoded_stats2["decoded_checksum"]
    assert decoded_stats1["fingerprint_matches"] and decoded_stats2["fingerprint_matches"]
    assert len(tier2) == len(tier1) + 1


def test_decode_mismatch_reports_first_location() -> None:
    block = _block()
    encoded, _ = encode_block(block, 1, low_frequency_threshold=1)
    decoded, _ = decode_block(encoded)
    decoded["terms"][0]["postings"][1]["positions"] = [99]
    ok, location, expected, actual = compare_decoded_block(block, decoded)
    assert not ok
    assert location is not None
    assert location.startswith("$.terms[0].postings[1].positions")
    assert expected == 2
    assert actual == 99

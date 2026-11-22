from __future__ import annotations

from decimal import Decimal

from instruction_factory.dedupe import dedupe_task


class FakeTokenizer:
    fingerprint = "fake-tokenizer-v1"

    def __init__(self, mapping: dict[str, set[str]]):
        self.mapping = mapping

    def tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(sorted(self.mapping[text]))


def _candidate(candidate_id: str, seq: int, score: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "sample_seq": seq,
        "total_fixed": score,
        "cache_key": f"score-{candidate_id}",
    }


def test_threshold_equal_is_inclusive_and_ties_use_candidate_id() -> None:
    candidate_ids = ["cand-a-000001", "cand-a-000002", "cand-a-000003"]
    mapping = {
        candidate_ids[0]: {"a", "b", "c", "d"},
        candidate_ids[1]: {"a", "b", "c", "e"},
        candidate_ids[2]: {"x", "y"},
    }
    scored = [
        _candidate(candidate_ids[0], 1, 90),
        _candidate(candidate_ids[1], 2, 90),
        _candidate(candidate_ids[2], 3, 95),
    ]
    sampling = {candidate_id: {"instruction": candidate_id} for candidate_id in candidate_ids}
    artifact = dedupe_task(
        task_type="a",
        scored_candidates=scored,
        sampling_by_id=sampling,
        threshold=Decimal("0.6"),
        tokenizer=FakeTokenizer(mapping),  # type: ignore[arg-type]
    )

    first_group = artifact["groups"][0]
    assert first_group["representative_candidate_id"] == candidate_ids[0]
    assert [member["candidate_id"] for member in first_group["member_candidates"]] == candidate_ids[:2]
    removed = artifact["removed_candidates"][0]
    assert removed["candidate_id"] == candidate_ids[1]
    assert removed["representative_candidate_id"] == candidate_ids[0]
    assert removed["similarity"] == "0.6"
    assert removed["threshold_equal"] is True
    assert artifact["equal_threshold_pairs"] == [
        {"left_candidate_id": candidate_ids[0], "right_candidate_id": candidate_ids[1]}
    ]


def test_representative_uses_highest_fixed_total() -> None:
    candidate_ids = ["cand-b-000001", "cand-b-000002"]
    mapping = {
        candidate_ids[0]: {"a", "b", "c"},
        candidate_ids[1]: {"a", "b", "c"},
    }
    scored = [_candidate(candidate_ids[0], 1, 80), _candidate(candidate_ids[1], 2, 81)]
    sampling = {candidate_id: {"instruction": candidate_id} for candidate_id in candidate_ids}
    artifact = dedupe_task(
        task_type="b",
        scored_candidates=scored,
        sampling_by_id=sampling,
        threshold=Decimal("1"),
        tokenizer=FakeTokenizer(mapping),  # type: ignore[arg-type]
    )
    assert artifact["retained_candidate_ids"] == [candidate_ids[1]]
    assert artifact["removed_candidates"][0]["representative_candidate_id"] == candidate_ids[1]


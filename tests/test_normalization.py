from __future__ import annotations

import unittest

from mmledger.jsonutil import MISSING
from mmledger.models import MergeRules, R3, Receipt, RequestSpec
from mmledger.normalization import batch_input_hash, normalize_batch, resolve_ownership


MERGE_RULES = MergeRules(
    worker_sets={"active": ("w1", "w2")},
    parameter_version_sets={"production": (2, 3, 4)},
    active_worker_set="active",
    active_parameter_version_set="production",
    worker_priority=("w1", "w2"),
)


def request(sequence: int, batch_id: str, worker_id: str = "w1") -> RequestSpec:
    return RequestSpec(
        sequence=sequence,
        request_id="q1",
        kind="image_text",
        arrival_ms=sequence,
        batch_id=batch_id,
        worker_id=worker_id,
        input_seed=100 + sequence,
    )


def delivery(
    batch_id: str,
    *,
    worker_id: str,
    parameter_version: int,
    duration_ms: int,
    stable_rank: int = 1,
    uid: str,
    overwritten_ref: str | None = None,
) -> Receipt:
    return Receipt(
        batch_id=batch_id,
        request_id="q1",
        output_digest=f"digest-{batch_id}-{worker_id}",
        parameter_version=parameter_version,
        duration_ms=duration_ms,
        worker_id=worker_id,
        stable_rank=stable_rank,
        overwritten_ref=overwritten_ref,
        parameter_digest=f"pd-{parameter_version}",
        format_version=R3,
        source_version=R3,
        uid=uid,
        source_file=f"{batch_id}.json",
    )


def resolve(*specs: tuple[RequestSpec, list[Receipt]]):
    batches = []
    requests = []
    for req, receipts in specs:
        requests.append(req)
        batch, failures = normalize_batch(
            req.batch_id,
            [req],
            receipts,
            merge_rules=MERGE_RULES,
            input_hash=batch_input_hash([req], receipts),
        )
        assert not failures
        batches.append(batch)
    result = resolve_ownership(batches, requests, MERGE_RULES)
    assert not result.failures
    assert len(result.decisions) == 1
    return result.decisions[0]


class NormalizationTests(unittest.TestCase):
    def test_no_overwrite_uses_parameter_version_then_batch_id(self) -> None:
        decision = resolve(
            (
                request(1, "batch-a"),
                [delivery("batch-a", worker_id="w1", parameter_version=2, duration_ms=1, uid="a#0")],
            ),
            (
                request(2, "batch-b"),
                [delivery("batch-b", worker_id="w1", parameter_version=4, duration_ms=20, uid="b#0")],
            ),
        )
        self.assertEqual(decision.batch_id, "batch-b")
        self.assertEqual(decision.basis["rule"], "parameter_version_desc+batch_id_asc")
        self.assertEqual(len(decision.candidates), 2)

    def test_overwrite_reference_beats_parameter_version(self) -> None:
        decision = resolve(
            (
                request(1, "batch-a"),
                [delivery("batch-a", worker_id="w1", parameter_version=9, duration_ms=1, uid="a#0")],
            ),
            (
                request(2, "batch-b"),
                [
                    delivery(
                        "batch-b",
                        worker_id="w1",
                        parameter_version=2,
                        duration_ms=20,
                        uid="b#0",
                        overwritten_ref="batch-a:q1:w1:1",
                    )
                ],
            ),
        )
        self.assertEqual(decision.batch_id, "batch-b")
        self.assertEqual(decision.basis["rule"], "overwrite_reference+lowest_duration_ms")
        self.assertEqual(decision.basis["overwritten_uids"], ["a#0"])

    def test_same_batch_rank_conflict_uses_worker_id_and_records_both(self) -> None:
        decision = resolve(
            (
                request(1, "batch-a"),
                [
                    delivery("batch-a", worker_id="w2", parameter_version=3, duration_ms=5, stable_rank=2, uid="w2#0"),
                    delivery("batch-a", worker_id="w1", parameter_version=3, duration_ms=9, stable_rank=1, uid="w1#0"),
                ],
            )
        )
        self.assertEqual(decision.worker_id, "w1")
        self.assertEqual(len(decision.rank_conflicts), 1)
        conflict = decision.rank_conflicts[0]
        self.assertEqual((conflict.worker_a, conflict.worker_b), ("w1", "w2"))
        self.assertEqual(conflict.winner_worker, "w1")


if __name__ == "__main__":
    unittest.main()

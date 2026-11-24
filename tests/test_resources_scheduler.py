from __future__ import annotations

import unittest

from mmledger.errors import E_NORMALIZE_BUDGET_PENDING, E_WORKER_QUOTA_EXCEEDED
from mmledger.jsonutil import MISSING
from mmledger.models import (
    BatchNormalized,
    ModelSpec,
    OwnershipDecision,
    WorkerSpec,
)
from mmledger.resources import compute_batch_resource, compute_invalidation, quality_summary
from mmledger.scheduler import BatchPlan, schedule_batches


def batch(batch_id: str, worker_id: str, input_hash: str, param_version: int = 3) -> BatchNormalized:
    return BatchNormalized(
        batch_id=batch_id,
        worker_id=worker_id,
        param_version=param_version,
        receipts=(),
        candidates=(),
        conflicts=(),
        input_hash=input_hash,
    )


def decision(request_id: str, batch_id: str, kind: str, duration) -> OwnershipDecision:
    return OwnershipDecision(
        request_id=request_id,
        request_sequence=1,
        kind=kind,
        batch_id=batch_id,
        worker_id="w1",
        authoritative_uid=f"{request_id}#0",
        candidates=(),
        basis={"rule": "test"},
        duration_ms=duration,
        parameter_version=3,
        output_digest="digest",
    )


class ResourceTests(unittest.TestCase):
    def test_fixed_point_macs_utilization_and_quality(self) -> None:
        model = ModelSpec(input_dim=4, hidden_dim=2, quantization_bits=8)
        item = batch("b1", "w1", "hash")
        decisions = [
            decision("q1", "b1", "image_text", 10),
            decision("q2", "b1", "image_only", MISSING),
        ]
        resource = compute_batch_resource(item, decisions, model=model, batch_size=4)
        self.assertEqual(resource.request_count, 2)
        self.assertEqual(resource.macs, 2 * model.macs_per_request)
        self.assertEqual(resource.quantized_macs, (resource.macs * 8 + 7) // 8)
        self.assertEqual(resource.duration_ms, 10)
        self.assertEqual(resource.duration_observed_count, 1)
        self.assertEqual(resource.duration_missing_count, 1)
        self.assertEqual(resource.utilization_ppm, 50_000)
        quality = quality_summary(decisions, [resource])
        self.assertEqual(quality["kinds"]["image_text"]["count"], 1)
        self.assertEqual(quality["kinds"]["image_only"]["count"], 1)
        self.assertEqual(quality["kinds"]["image_only"]["missing_duration_count"], 1)
        self.assertEqual(quality["total_authoritative_requests"], 2)

    def test_parameter_version_change_invalidates_only_linked_batches(self) -> None:
        previous = {
            "worker_parameter_versions": {"w1": 3, "w2": 4},
            "merge_rules_hash": "merge",
            "model_hash": "model",
            "batch_size": 3,
            "batches": {
                "b1": {"worker_id": "w1", "worker_parameter_version": 3, "input_hash": "h1"},
                "b2": {"worker_id": "w2", "worker_parameter_version": 4, "input_hash": "h2"},
            },
        }
        batches = [batch("b1", "w1", "h1"), batch("b2", "w2", "h2")]
        invalidation = compute_invalidation(
            workers=[
                WorkerSpec("w1", "127.0.0.1", 19001, 5, "pd5"),
                WorkerSpec("w2", "127.0.0.1", 19002, 4, "pd4"),
            ],
            batches=batches,
            previous_cache=previous,
            merge_rules_hash="merge",
            model_hash="model",
            batch_size=3,
        )
        self.assertEqual(invalidation["normalization_invalidated_batches"], ["b1"])
        self.assertEqual(invalidation["resource_invalidated_batches"], ["b1"])
        self.assertEqual(invalidation["reused_batches"], ["b2"])
        self.assertEqual(invalidation["changed_workers"][0]["worker_id"], "w1")

    def test_batch_size_change_invalidates_resources_not_batch_normalization(self) -> None:
        previous = {
            "worker_parameter_versions": {"w1": 3},
            "merge_rules_hash": "merge",
            "model_hash": "model",
            "batch_size": 3,
            "batches": {
                "b1": {"worker_id": "w1", "worker_parameter_version": 3, "input_hash": "h1"},
            },
        }
        invalidation = compute_invalidation(
            workers=[WorkerSpec("w1", "127.0.0.1", 19001, 3, "pd3")],
            batches=[batch("b1", "w1", "h1")],
            previous_cache=previous,
            merge_rules_hash="merge",
            model_hash="model",
            batch_size=4,
        )
        self.assertEqual(invalidation["normalization_invalidated_batches"], [])
        self.assertEqual(invalidation["resource_invalidated_batches"], ["b1"])
        self.assertTrue(invalidation["resource_global_invalidated"])


class SchedulerTests(unittest.TestCase):
    def test_budget_pending_then_previous_pending_has_priority(self) -> None:
        batches = [
            BatchPlan("a", "w1", ("q1", "q2")),
            BatchPlan("b", "w1", ("q3",)),
            BatchPlan("c", "w2", ("q4", "q5")),
            BatchPlan("d", "w2", ("q6",)),
        ]
        first = schedule_batches(
            batches,
            budget=3,
            worker_quotas={"w1": 10, "w2": 10},
        )
        self.assertEqual([ticket.batch_id for ticket in first.tickets], ["a", "b"])
        self.assertEqual(first.pending_batches, ("c", "d"))
        self.assertEqual(first.consumed_budget, 3)
        self.assertEqual({failure.code for failure in first.failures}, {E_NORMALIZE_BUDGET_PENDING})

        second = schedule_batches(
            batches,
            budget=3,
            worker_quotas={"w1": 10, "w2": 10},
            previous_pending=["c", "d"],
        )
        self.assertEqual([ticket.batch_id for ticket in second.tickets], ["c", "d"])
        self.assertTrue(all(ticket.priority_pending for ticket in second.tickets))
        deduction_trace = [item for item in second.trace if item["event"] == "deduct"]
        self.assertEqual([item["budget_remaining"] for item in deduction_trace], [2, 1, 0])

    def test_worker_quota_blocks_batch_without_deducting_budget(self) -> None:
        result = schedule_batches(
            [
                BatchPlan("a", "w1", ("q1", "q2")),
                BatchPlan("b", "w1", ("q3",)),
                BatchPlan("c", "w2", ("q4",)),
            ],
            budget=10,
            worker_quotas={"w1": 2, "w2": 10},
        )
        self.assertEqual([ticket.batch_id for ticket in result.tickets], ["a", "c"])
        self.assertEqual(result.quota_blocked_batches, ("b",))
        self.assertEqual(result.consumed_budget, 3)
        self.assertEqual(result.failures[0].code, E_WORKER_QUOTA_EXCEEDED)


if __name__ == "__main__":
    unittest.main()

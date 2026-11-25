"""端到端编排：规划选优 → 分片 → 回环调度 → 账本原子提交 → 回执归一。"""

import os
from typing import Any, Dict, List, Optional

from .cost import CostEstimator, choose_plan
from .invalidation import PlanVersionState
from .ledger import Ledger
from .loader import load_query, load_session, load_weights
from .planner import enumerate_plans
from .receipts import encode_receipt, load_receipts
from .scheduler import Scheduler
from .sharding import split_batches
from .storage import load_generated_specs


class Pipeline:
    def __init__(
        self,
        workspace: str,
        executor_count: int = 2,
        max_in_flight: int = 2,
        per_executor_slots: int = 1,
        shard_rows: int = 100,
        declared_ports: Optional[List[int]] = None,
    ):
        self.workspace = workspace
        self.data_dir = os.path.join(workspace, "data")
        self.config_dir = os.path.join(workspace, "configs")
        self.receipt_dir = os.path.join(workspace, "receipts")
        self.ledger_dir = os.path.join(workspace, "ledger")
        self.snapshot_dir = os.path.join(workspace, "snapshots")
        os.makedirs(self.receipt_dir, exist_ok=True)
        os.makedirs(self.ledger_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.groups = load_generated_specs(self.data_dir)
        self.query = load_query(os.path.join(self.config_dir, "query.json"), self.groups)
        self.weights = load_weights(os.path.join(self.config_dir, "weights.json"))
        self.session = load_session(os.path.join(self.config_dir, "session.json"))
        self.executor_count = executor_count
        self.max_in_flight = max_in_flight
        self.per_executor_slots = per_executor_slots
        self.shard_rows = shard_rows
        self.declared_ports = declared_ports

    def plan(self):
        plans = enumerate_plans(self.query)
        estimator = CostEstimator(self.groups, self.weights, self.session)
        evaluations = [estimator.evaluate(plan) for plan in plans]
        decision = choose_plan(evaluations)
        return plans, evaluations, decision

    def run(
        self,
        plan_version: int = 1,
        prior_state: Optional[PlanVersionState] = None,
        ledger: Optional[Ledger] = None,
        crash_after: Optional[int] = None,
    ) -> Dict[str, Any]:
        plans, evaluations, decision = self.plan()
        chosen_id = decision["chosen_plan_id"]
        chosen_plan = next(plan for plan in plans if plan["plan_id"] == chosen_id)
        cost_summary = next(item for item in evaluations if item["plan_id"] == chosen_id)
        group_by_id = {group.bg_id: group for group in self.groups}
        driver = group_by_id[chosen_plan["body"]["order"][0]]
        batches = split_batches(chosen_plan, driver, self.shard_rows)

        requests = []
        for batch in batches:
            requests.append(
                {
                    "type": "shard",
                    "query_path": os.path.join(self.config_dir, "query.json"),
                    "plan": chosen_plan,
                    "plan_version": plan_version,
                    "executor_id": "executor-placeholder",
                    "cost_vector": cost_summary["vector"],
                    "batch": batch,
                }
            )

        own_ledger = ledger is None
        if own_ledger:
            ledger = Ledger(
                os.path.join(self.ledger_dir, "coljoin.db"), self.snapshot_dir
            )
        ledger.init_cursor(plan_version, len(batches))
        receipts: List[Dict[str, Any]] = []
        waits = []
        batch_by_id = {batch["shard_id"]: batch for batch in batches}
        ordinal_by_id = {batch["shard_id"]: batch["ordinal"] for batch in batches}
        committed_count = 0
        try:
            with Scheduler(
                self.data_dir,
                self.executor_count,
                self.max_in_flight,
                self.per_executor_slots,
                declared_ports=self.declared_ports,
            ) as scheduler:
                for index, request in enumerate(requests):
                    request["executor_id"] = scheduler.handles[
                        index % self.executor_count
                    ].executor_id
                report = scheduler.run(requests)
                waits = [
                    {
                        "shard_id": event.shard_id,
                        "reason": event.reason,
                        "wait_start_ms": event.wait_start_ms,
                        "wait_end_ms": event.wait_end_ms,
                        "waited_ms": event.waited_ms,
                    }
                    for event in report.waits
                ]
                # 回执按 shard_id 排序后逐个提交，使崩溃点精确落在提交边界：
                # crash_after 之后的分片不会留下任何 committed 记录。
                for receipt in report.receipts:
                    if crash_after is not None and committed_count >= crash_after:
                        raise SystemExit(255)
                    shard_id = receipt["shard_id"]
                    ledger.commit_batch(
                        plan_version=plan_version,
                        plan_id=chosen_id,
                        ordinal=ordinal_by_id[shard_id],
                        executor_id=receipt["executor_id"],
                        receipt=receipt,
                        cost_summary=cost_summary,
                        shard_digest_value=batch_by_id[shard_id]["digest"],
                    )
                    committed_count += 1
                    receipts.append(receipt)
        finally:
            if own_ledger:
                ledger.close()

        self._write_receipt_versions(receipts, cost_summary["vector"], plan_version)
        return {
            "plan_version": plan_version,
            "chosen_plan_id": chosen_id,
            "decision": decision,
            "cost_summary": cost_summary,
            "batches": batches,
            "receipts": receipts,
            "waits": waits,
        }

    def _write_receipt_versions(self, receipts, cost_vector, plan_version):
        q1_path = os.path.join(self.receipt_dir, f"receipts-q1-v{plan_version}.jsonl")
        q2_path = os.path.join(self.receipt_dir, f"receipts-q2-v{plan_version}.jsonl")
        q3_path = os.path.join(self.receipt_dir, f"receipts-q3-v{plan_version}.jsonl")
        q1_records = [
            {
                "shard_id": item["shard_id"],
                "aggregate": item["aggregate"],
            }
            for item in receipts
        ]
        q2_records = [
            {
                "shard_id": item["shard_id"],
                "aggregate": item["aggregate"],
                "plan_version": plan_version,
                "elapsed_ms": item["elapsed_ms"],
            }
            for item in receipts
        ]
        q3_records = [
            {
                "shard_id": item["shard_id"],
                "aggregate": item["aggregate"],
                "plan_version": plan_version,
                "elapsed_ms": item["elapsed_ms"],
                "executor_id": item["executor_id"],
                "cost_vector": item["cost_vector"],
                "clock_start_ms": item["clock_start_ms"],
                "clock_end_ms": item["clock_end_ms"],
            }
            for item in receipts
        ]
        for path, version, records in (
            (q1_path, "q1", q1_records),
            (q2_path, "q2", q2_records),
            (q3_path, "q3", q3_records),
        ):
            with open(path, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(encode_receipt(version, record) + "\n")

    def normalize_receipts(self, version: str, plan_version: int):
        path = os.path.join(
            self.receipt_dir, f"receipts-{version}-v{plan_version}.jsonl"
        )
        plans, evaluations, _ = self.plan()
        chosen_cost = evaluations[0]
        return load_receipts(path, chosen_cost["vector"])

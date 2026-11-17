"""Recovery-time replay verification against the committed ledger prefix."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .engine import Simulation
from .errors import BusinessError, E_REPLAY
from .ledger import Ledger
from .spec import ClusterSpec


REPLAY_FIELDS = (
    "observation",
    "window_stats",
    "rule_evaluations",
    "selected_rule_id",
    "expected_replicas",
    "attempted_replicas",
    "budget_target_replicas",
    "final_replicas",
    "decision_type",
    "cooldown_suppressed",
    "cooldown_code",
    "cooldown_trigger_minute",
    "cooldown_message",
)


@dataclass(frozen=True)
class ReplayResult:
    run_id: str
    verified_minutes: int
    verified_decisions: int
    decision_digest: str
    ledger_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "business_code": "OK",
            "run_id": self.run_id,
            "verified_minutes": self.verified_minutes,
            "verified_decisions": self.verified_decisions,
            "decision_digest": self.decision_digest,
            "ledger_status": self.ledger_status,
        }


def verify_spec_against_ledger(spec: ClusterSpec, run: dict[str, Any]) -> None:
    if run["declaration_version"] != spec.declaration_version:
        raise BusinessError(
            E_REPLAY,
            "declaration version differs from the ledger and cannot be replayed safely",
            "declaration_version",
            {
                "ledger_declaration_version": run["declaration_version"],
                "current_declaration_version": spec.declaration_version,
                "basis": "the ledger is bound to the declaration version at run creation",
            },
        )
    ledger_policy = run["policy"]
    current_policy = spec.policy_payload()
    for field in (
        "declaration_version",
        "global_replica_budget",
        "aggregation_threshold",
        "window_size",
    ):
        if ledger_policy.get(field) != current_policy.get(field):
            raise BusinessError(
                E_REPLAY,
                "policy parameter differs from the ledger and cannot be reproduced",
                field,
                {
                    "ledger_value": ledger_policy.get(field),
                    "current_value": current_policy.get(field),
                    "basis": "policy parameters are part of the reproducibility contract",
                },
            )
    if run["spec_sha256"] != spec.sha256():
        raise BusinessError(
            E_REPLAY,
            "declaration content differs from the ledger and cannot be replayed safely",
            "spec_sha256",
            {
                "ledger_spec_sha256": run["spec_sha256"],
                "current_spec_sha256": spec.sha256(),
                "basis": "declaration_version and policy match, but observations or rules changed",
            },
        )


def _replay_error(
    header: dict[str, Any],
    workload_id: str | None,
    field: str,
    ledger_value: Any,
    replay_value: Any,
) -> BusinessError:
    return BusinessError(
        E_REPLAY,
        "replay diverged from the committed ledger",
        field,
        {
            "minute_index": header["minute_index"],
            "minute": header["minute"],
            "workload_id": workload_id,
            "field": field,
            "ledger_value": ledger_value,
            "replay_value": replay_value,
            "basis": "first difference in minute, workload-id and replay-field order",
        },
    )


def replay_ledger(
    ledger: Ledger,
    spec: ClusterSpec,
    run_id: str,
    *,
    limit: int | None = None,
) -> ReplayResult:
    run = ledger.get_run(run_id)
    if run is None:
        from .errors import ledger_error

        raise ledger_error("run does not exist", "run_id", run_id=run_id)
    verify_spec_against_ledger(spec, run)
    committed = ledger.load_committed(run_id, limit=limit)
    simulation = Simulation(spec, requested_path=run["path"])
    verified_minutes = 0
    verified_decisions = 0
    for header in committed:
        if header["minute_index"] != simulation.index:
            raise _replay_error(header, None, "minute_index", header["minute_index"], simulation.index)
        try:
            replayed = simulation.step()
        except StopIteration:
            raise _replay_error(header, None, "minute_index", header["minute_index"], None)
        if replayed.minute != header["minute"]:
            raise _replay_error(header, None, "minute", header["minute"], replayed.minute)
        if replayed.budget_used != header["budget_used"]:
            raise _replay_error(
                header, None, "budget_used", header["budget_used"], replayed.budget_used
            )
        ledger_decisions = {item["workload_id"]: item for item in header["decisions"]}
        replay_decisions = {item.workload_id: item for item in replayed.decisions}
        if sorted(ledger_decisions) != sorted(replay_decisions):
            raise _replay_error(
                header,
                None,
                "workload_set",
                sorted(ledger_decisions),
                sorted(replay_decisions),
            )
        for workload_id in sorted(ledger_decisions):
            ledger_item = ledger_decisions[workload_id]
            replay_item = replay_decisions[workload_id]
            ledger_core = {
                "observation": ledger_item["observation"],
                "window_stats": ledger_item["window_stats"],
                "rule_evaluations": ledger_item["rule_evaluations"],
                "selected_rule_id": ledger_item["selected_rule_id"],
                "expected_replicas": ledger_item["expected_replicas"],
                "attempted_replicas": ledger_item["attempted_replicas"],
                "budget_target_replicas": ledger_item["budget_target_replicas"],
                "final_replicas": ledger_item["final_replicas"],
                "decision_type": ledger_item["decision_type"],
                "cooldown_suppressed": ledger_item["cooldown_suppressed"],
                "cooldown_code": ledger_item["cooldown_code"],
                "cooldown_trigger_minute": ledger_item["cooldown_trigger_minute"],
                "cooldown_message": ledger_item["cooldown_message"],
            }
            replay_core = replay_item.replay_core()
            for field in REPLAY_FIELDS:
                if ledger_core[field] != replay_core[field]:
                    raise _replay_error(
                        header,
                        workload_id,
                        field,
                        ledger_core[field],
                        replay_core[field],
                    )
            verified_decisions += 1
        verified_minutes += 1

    payload = [
        {
            "minute_index": header["minute_index"],
            "minute": header["minute"],
            "budget_used": header["budget_used"],
            "decisions": [
                json.loads(json.dumps(item["decision"], sort_keys=True))
                for item in header["decisions"]
            ],
        }
        for header in committed
    ]
    import hashlib

    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return ReplayResult(
        run_id=run_id,
        verified_minutes=verified_minutes,
        verified_decisions=verified_decisions,
        decision_digest=digest,
        ledger_status=run["status"],
    )

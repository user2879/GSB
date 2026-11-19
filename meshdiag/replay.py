from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json, sha256_json
from .circuit import CircuitBreakerRegistry
from .engine import Engine
from .errors import fail_replay
from .models import (
    AttemptTrace,
    RequestResult,
    RequestSpec,
    instance_fingerprint,
)
from .snapshot import rebuild_revision_from_snapshot, result_rows


@dataclass
class ReplayReport:
    snapshot_id: str
    revision_version: int
    requests_compared: int
    ledger_requests_compared: int
    ledger_attempts_compared: int
    instance_set_fingerprint: str
    consistency: str = "REPLAY_MATCH"
    clock_fields_excluded: tuple[str, ...] = (
        "started_mono_ms",
        "ended_mono_ms",
        "elapsed_ms",
        "budget_consumed_ms",
        "wall_elapsed_ms",
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "revision_version": self.revision_version,
            "requests_compared": self.requests_compared,
            "ledger_requests_compared": self.ledger_requests_compared,
            "ledger_attempts_compared": self.ledger_attempts_compared,
            "instance_set_fingerprint": self.instance_set_fingerprint,
            "consistency": self.consistency,
            "clock_fields_excluded": list(self.clock_fields_excluded),
        }


def replay_snapshot(
    engine: Engine,
    snapshot_path: str | Path,
    *,
    current_revision=None,
) -> ReplayReport:
    payload = engine.snapshots.load(snapshot_path)
    revision, snapshot_services = rebuild_revision_from_snapshot(payload)
    if payload.get("status") != "completed":
        unknown = len(payload.get("in_progress", []))
        fail_replay(
            "快照不是完整会话，不能进行结论重放",
            field_name="snapshot.status",
            expected="completed",
            actual=payload.get("status"),
            basis=f"快照仍有 {unknown} 个未完成请求，需先重跑",
        )
    if current_revision is None:
        current_revision = engine.load_workspace_revision()
    if (
        current_revision.revision_version != revision.revision_version
        or current_revision.digest != revision.digest
    ):
        fail_replay(
            "修订版本或内容摘要不一致，无法按记录版本重放",
            field_name="revision_version",
            expected=revision.revision_version,
            actual=current_revision.revision_version,
            basis=(
                f"快照 r{revision.revision_version}/{revision.digest[:12]}，"
                f"当前 r{current_revision.revision_version}/{current_revision.digest[:12]}"
            ),
        )
    if (
        current_revision.dependency_graph.version
        != payload.get("dependency_graph_version")
    ):
        fail_replay(
            "依赖图版本不一致",
            field_name="dependency_graph_version",
            expected=payload.get("dependency_graph_version"),
            actual=current_revision.dependency_graph.version,
            basis="规则依赖关系已变化，不能复现同一治理计划",
        )
    current_fp = instance_fingerprint(engine.workspace.services)
    if current_fp != payload.get("instance_fingerprint"):
        fail_replay(
            "本地上游实例集合与快照不一致，无法复现实例选择",
            field_name="instance_set",
            expected=payload.get("instance_fingerprint"),
            actual=current_fp,
            basis="实例 id、回环地址、端口或服务并发配额发生变化",
        )
    request_rows = sorted(
        payload.get("requests", []),
        key=lambda row: (int(row.get("index", 0)), str(row.get("id", ""))),
    )
    requests = tuple(
        RequestSpec(
            request_id=str(row["id"]),
            method=str(row.get("method", "GET")),
            path=str(row.get("path", "/")),
            index=int(row.get("index", 0)),
        )
        for row in request_rows
    )
    replay_breakers = CircuitBreakerRegistry()
    replay_breakers.restore(payload.get("initial_circuit_breakers", []))
    actual_results, _ = engine.execute_sequence(
        revision=revision,
        services=engine.workspace.services,
        requests=requests,
        global_concurrency=int(payload.get("global_concurrency", engine.workspace.global_concurrency)),
        breakers=replay_breakers,
        snapshot=None,
        progress=None,
    )
    expected_results = result_rows(payload)
    expected_map = {result.request_id: result for result in expected_results}
    actual_map = {result.request_id: result for result in actual_results}
    for expected in expected_results:
        request_id = expected.request_id
        actual = actual_map.get(request_id)
        if actual is None:
            fail_replay(
                "重放缺少请求结果",
                request_id=request_id,
                rule_id=expected.rule_id,
                field_name="request_result",
                expected="present",
                actual="missing",
                basis="请求序列未产生对应结果",
            )
        for field_name, expected_value, actual_value in (
            ("outcome", expected.outcome, actual.outcome),
            ("instance_sequence", list(expected.instance_sequence), list(actual.instance_sequence)),
            ("attempt_count", expected.attempt_count, actual.attempt_count),
            ("rule_id", expected.rule_id, actual.rule_id),
            ("service_id", expected.service_id, actual.service_id),
        ):
            if expected_value != actual_value:
                fail_replay(
                    "重放首个不一致请求",
                    request_id=request_id,
                    rule_id=expected.rule_id,
                    field_name=field_name,
                    expected=expected_value,
                    actual=actual_value,
                    basis="逐请求结局、实例序列或尝试次数与快照不一致",
                )
        expected_traces = sorted(expected.traces, key=lambda item: item.attempt_no)
        actual_traces = sorted(actual.traces, key=lambda item: item.attempt_no)
        if len(expected_traces) != len(actual_traces):
            fail_replay(
                "重放尝试轨迹数量不一致",
                request_id=request_id,
                rule_id=expected.rule_id,
                field_name="attempt_count",
                expected=len(expected_traces),
                actual=len(actual_traces),
                basis="逐尝试账本不一致",
            )
        for expected_trace, actual_trace in zip(expected_traces, actual_traces):
            for field_name, expected_value, actual_value in (
                ("attempt_no", expected_trace.attempt_no, actual_trace.attempt_no),
                ("instance_id", expected_trace.instance_id, actual_trace.instance_id),
                ("outcome", expected_trace.outcome, actual_trace.outcome),
                ("reason", expected_trace.reason, actual_trace.reason),
                ("status_code", expected_trace.status_code, actual_trace.status_code),
            ):
                if expected_value != actual_value:
                    fail_replay(
                        "重放账本首个不一致字段",
                        request_id=request_id,
                        rule_id=expected_trace.rule_id,
                        field_name=f"attempt[{expected_trace.attempt_no}].{field_name}",
                        expected=expected_value,
                        actual=actual_value,
                        basis="尝试轨迹与快照账本不一致",
                    )
    ledger_rows = payload.get("ledger_rows", {})
    if not isinstance(ledger_rows, dict) or not ledger_rows.get("requests"):
        fail_replay(
            "快照缺少完整账本行，无法完成逐项账本比较",
            field_name="ledger_rows",
            expected="non-empty",
            actual="missing",
            basis="快照未包含完整提交轮次的 SQLite 账本视图",
        )
    expected_ledger_requests = {
        str(row["request_id"]): row
        for row in ledger_rows.get("requests", [])
        if isinstance(row, dict) and "request_id" in row
    }
    expected_ledger_attempts = {}
    for row in ledger_rows.get("attempts", []):
        if isinstance(row, dict):
            key = (str(row.get("request_id")), int(row.get("attempt_no", 0)))
            expected_ledger_attempts[key] = row
    ledger_request_count = 0
    ledger_attempt_count = 0
    for actual in sorted(actual_results, key=lambda item: (item.request_index, item.request_id)):
        request_id = actual.request_id
        expected_row = expected_ledger_requests.get(request_id)
        if expected_row is None:
            continue
        ledger_request_count += 1
        for field_name, expected_value, actual_value in (
            ("rule_id", expected_row.get("rule_id"), actual.rule_id),
            ("service_id", expected_row.get("service_id"), actual.service_id),
            ("outcome", expected_row.get("outcome"), actual.outcome),
            ("reason", expected_row.get("reason"), actual.reason),
            ("attempt_count", expected_row.get("attempt_count"), actual.attempt_count),
            (
                "instance_sequence",
                json.loads(expected_row.get("instance_sequence_json", "[]")),
                list(actual.instance_sequence),
            ),
        ):
            if expected_value != actual_value:
                fail_replay(
                    "重放账本请求行不一致",
                    request_id=request_id,
                    rule_id=actual.rule_id,
                    field_name=f"ledger.requests.{field_name}",
                    expected=expected_value,
                    actual=actual_value,
                    basis="SQLite 账本快照与重放结果不一致",
                )
        for trace in sorted(actual.traces, key=lambda item: item.attempt_no):
            expected_attempt = expected_ledger_attempts.get((request_id, trace.attempt_no))
            if expected_attempt is None:
                fail_replay(
                    "重放账本缺少尝试行",
                    request_id=request_id,
                    rule_id=actual.rule_id,
                    field_name=f"ledger.attempts[{trace.attempt_no}]",
                    expected="present",
                    actual="missing",
                    basis="逐尝试账本不一致",
                )
            ledger_attempt_count += 1
            for field_name, expected_value, actual_value in (
                ("rule_id", expected_attempt.get("rule_id"), trace.rule_id),
                ("service_id", expected_attempt.get("service_id"), trace.service_id),
                ("instance_id", expected_attempt.get("instance_id"), trace.instance_id),
                ("outcome", expected_attempt.get("outcome"), trace.outcome),
                ("reason", expected_attempt.get("reason"), trace.reason),
                ("status_code", expected_attempt.get("status_code"), trace.status_code),
            ):
                if expected_value != actual_value:
                    fail_replay(
                        "重放账本尝试行不一致",
                        request_id=request_id,
                        rule_id=trace.rule_id,
                        field_name=(
                            f"ledger.attempts[{trace.attempt_no}].{field_name}"
                        ),
                        expected=expected_value,
                        actual=actual_value,
                        basis="SQLite 账本快照与重放尝试轨迹不一致",
                    )
    return ReplayReport(
        snapshot_id=str(payload["snapshot_id"]),
        revision_version=revision.revision_version,
        requests_compared=len(expected_results),
        ledger_requests_compared=ledger_request_count,
        ledger_attempts_compared=ledger_attempt_count,
        instance_set_fingerprint=current_fp,
    )

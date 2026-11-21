from __future__ import annotations

from typing import Any, Sequence

from .canonical import digest_obj
from .errors import FORMAT_VERSION, BusinessError
from .fixed import build_error_attribution, quantize
from .generator import generate_session
from .ledger import Ledger, affected_window_keys
from .logging_utils import JsonlLogger
from .model import (
    Event,
    LateDecision,
    RuleSpec,
    SessionSpec,
    WatermarkTrace,
    WindowAccumulator,
    WindowKey,
    WindowResult,
)


def floor_window_start(event_time_ms: int, width_ms: int) -> int:
    return (event_time_ms // width_ms) * width_ms


def make_view_id(spec: SessionSpec) -> str:
    return f"v{spec.rules_version}-{spec.spec_digest[:16]}"


def _window_key(rule: RuleSpec, partition_key: str, timestamp: int) -> WindowKey:
    start = floor_window_start(timestamp, rule.width_ms)
    return WindowKey(
        rule_id=rule.rule_id,
        rule_version=rule.rule_version,
        partition_key=partition_key,
        window_start_ms=start,
        window_end_ms=start + rule.width_ms,
    )


def _key_sort_value(key: WindowKey) -> tuple[Any, ...]:
    return (key.partition_key, key.window_end_ms, key.window_start_ms, key.rule_id, key.rule_version)


class Engine:
    def __init__(
        self,
        spec: SessionSpec,
        *,
        session_id: str | None = None,
        ledger: Ledger | None = None,
        logger: JsonlLogger | None = None,
        view_id: str | None = None,
    ) -> None:
        self.spec = spec
        self.session_id = session_id or f"session-{spec.spec_digest[:20]}"
        self.view_id = view_id or make_view_id(spec)
        self.ledger = ledger
        self.logger = logger
        self.events: list[Event] = []
        self.source_digests: dict[str, str] = {}
        self._processed_arrivals: set[int] = set()
        self.watermarks: dict[str, int] = {}
        self.open_windows: dict[WindowKey, WindowAccumulator] = {}
        self.finalized_windows: dict[WindowKey, WindowResult] = {}
        self.unknown_windows: set[WindowKey] = set()
        self.traces: list[WatermarkTrace] = []
        self.decisions: list[LateDecision] = []
        self.event_cursor = 0
        self.batch_index = 0
        self.last_batch: dict[str, Any] | None = None
        self.ledger_content_digest = ""
        self._rule_by_id = {rule.rule_id: rule for rule in spec.rules}
        if ledger is not None:
            self._register_or_validate_session()

    def _register_or_validate_session(self) -> None:
        assert self.ledger is not None
        self.events, generations = generate_session(self.spec)
        self.source_digests = {
            source_id: generation.digest for source_id, generation in generations.items()
        }
        try:
            runtime = self.ledger.get_runtime(self.session_id)
        except KeyError:
            self.ledger.register_session(
                self.session_id, self.spec, self.view_id, self.source_digests
            )
            self.ledger_content_digest = self.ledger.get_view_manifest(
                self.session_id, self.view_id
            )["content_digest"]
            return
        if runtime["current_view_id"] != self.view_id:
            raise BusinessError(
                code="E_SNAPSHOT",
                message="session is already bound to a different active view",
                field="view_id",
                details={
                    "session_id": self.session_id,
                    "expected_view_id": runtime["current_view_id"],
                    "actual_view_id": self.view_id,
                },
            )
        if int(runtime["event_cursor"]) > 0:
            raise BusinessError(
                code="E_SNAPSHOT",
                message="session already has progress; restore it from a snapshot instead of creating a fresh engine",
                field="session_id",
                details={
                    "session_id": self.session_id,
                    "view_id": self.view_id,
                    "event_cursor": runtime["event_cursor"],
                },
            )

    def _rule_lateness(self, key: WindowKey) -> int:
        rule = self._rule_by_id.get(key.rule_id)
        if rule is None or rule.rule_version != key.rule_version:
            raise BusinessError(
                code="E_SPEC",
                message="window state references an unknown rule version",
                field="rule_id",
                details={"rule_id": key.rule_id, "rule_version": key.rule_version},
            )
        return rule.lateness_ms

    def _decision_id(self, rule_id: str, event: Event) -> str:
        return digest_obj(
            {
                "session_id": self.session_id,
                "view_id": self.view_id,
                "rule_id": rule_id,
                "event_id": event.event_id,
                "arrival_no": event.arrival_no,
            }
        )

    def _trace_id(self, event: Event) -> str:
        return digest_obj(
            {
                "session_id": self.session_id,
                "view_id": self.view_id,
                "event_id": event.event_id,
                "arrival_no": event.arrival_no,
            }
        )

    def _new_accumulator(self, key: WindowKey) -> WindowAccumulator:
        accumulator = self.open_windows.get(key)
        if accumulator is None:
            accumulator = WindowAccumulator(key=key)
            self.open_windows[key] = accumulator
            if self.ledger is not None:
                self.ledger.stage_window(self.session_id, self.view_id, key)
        return accumulator

    def run_all(self) -> dict[str, Any]:
        if not self.events:
            self.events, generations = generate_session(self.spec)
            self.source_digests = {
                source_id: generation.digest for source_id, generation in generations.items()
            }
        return self.run_events(self.events)

    def run_events(self, events: Sequence[Event], *, canonicalize: bool = True) -> dict[str, Any]:
        ordered = sorted(events, key=lambda event: (event.arrival_no, event.event_id)) if canonicalize else list(events)
        if canonicalize and len({event.arrival_no for event in ordered}) != len(ordered):
            raise BusinessError(
                code="E_ORDER",
                message="arrival_no values must be unique for canonical ingestion",
                field="arrival_no",
            )
        for event in ordered:
            if event.arrival_no in self._processed_arrivals:
                continue
            self.ingest(event)
        return self.summary()

    def ingest(self, event: Event) -> dict[str, Any]:
        watermark_before = self.watermarks.get(event.partition_key, event.event_time_ms)
        backward = watermark_before - event.event_time_ms
        if backward > self.spec.order_backward_limit_ms:
            raise BusinessError(
                code="E_ORDER",
                message="event timestamp regressed beyond the configured order guard",
                field="event_time_ms",
                details={
                    "partition_key": event.partition_key,
                    "event_id": event.event_id,
                    "event_time_ms": event.event_time_ms,
                    "watermark_before_ms": watermark_before,
                    "backward_ms": backward,
                    "limit_ms": self.spec.order_backward_limit_ms,
                },
            )

        was_late = event.event_time_ms < watermark_before
        event_decisions: list[LateDecision] = []
        accepted_rules: list[str] = []

        for rule in self.spec.rules:
            if event.partition_key not in rule.partition_keys:
                continue
            key = _window_key(rule, event.partition_key, event.event_time_ms)
            fixed_value, quantization_error = quantize(event.measure, self.spec.fixed_scale)
            accumulator = self.open_windows.get(key)
            finalized_result = self.finalized_windows.get(key)
            budget_count_before = self.spec.lateness_budget_count - (
                finalized_result.late_accepted_count
                if finalized_result is not None
                else accumulator.late_accepted_count if accumulator is not None else 0
            )
            budget_amount_before = self.spec.lateness_budget_amount - (
                finalized_result.late_accepted_abs_amount
                if finalized_result is not None
                else accumulator.late_accepted_abs_amount if accumulator is not None else 0
            )
            cumulative_late_abs = (
                finalized_result.late_accepted_abs_amount
                if finalized_result is not None
                else accumulator.late_accepted_abs_amount if accumulator is not None else 0
            )
            decision = "DROP"
            reason = ""
            accepted = False

            if key in self.unknown_windows:
                reason = "WINDOW_STATE_UNKNOWN"
            elif key in self.finalized_windows or key.window_end_ms <= watermark_before - rule.lateness_ms:
                reason = "WINDOW_ALREADY_FINALIZED"
            elif not was_late:
                decision = "ACCEPT"
                reason = "ON_TIME"
                accepted = True
            else:
                late_abs = abs(fixed_value)
                if budget_count_before <= 0:
                    reason = "LATE_BUDGET_COUNT_EXHAUSTED"
                elif late_abs > budget_amount_before:
                    reason = "LATE_BUDGET_AMOUNT_EXCEEDED"
                else:
                    decision = "ACCEPT"
                    reason = "LATE_WITHIN_BUDGET"
                    accepted = True

            if accepted:
                accumulator = self._new_accumulator(key)
                accumulator.count += 1
                accumulator.sum_exact += fixed_value
                accumulator.quantization_error_sum += quantization_error
                accumulator.max_single_quantization_error = max(
                    accumulator.max_single_quantization_error, quantization_error
                )
                if was_late:
                    accumulator.late_accepted_count += 1
                    accumulator.late_accepted_abs_amount += abs(fixed_value)
                    accumulator.accepted_late_events.append(event.arrival_no)
                if event.event_time_ms % rule.width_ms == 0:
                    accumulator.boundary_abs_amount += abs(fixed_value)
                    accumulator.boundary_events.append(event.arrival_no)
                accepted_rules.append(rule.rule_id)

            if accepted and was_late:
                budget_count_after = budget_count_before - 1
                budget_amount_after = budget_amount_before - abs(fixed_value)
                cumulative_late_abs += abs(fixed_value)
            else:
                budget_count_after = budget_count_before
                budget_amount_after = budget_amount_before

            event_decisions.append(
                LateDecision(
                    decision_id=self._decision_id(rule.rule_id, event),
                    rule_id=rule.rule_id,
                    rule_version=rule.rule_version,
                    partition_key=event.partition_key,
                    event_id=event.event_id,
                    arrival_no=event.arrival_no,
                    event_time_ms=event.event_time_ms,
                    window_start_ms=key.window_start_ms,
                    window_end_ms=key.window_end_ms,
                    decision=decision,
                    reason=reason,
                    fixed_value=fixed_value,
                    budget_count_before=budget_count_before,
                    budget_count_after=budget_count_after,
                    budget_amount_before=budget_amount_before,
                    budget_amount_after=budget_amount_after,
                    cumulative_late_abs_amount=cumulative_late_abs,
                    was_late=was_late,
                )
            )

        watermark_after = max(watermark_before, event.event_time_ms)
        self.watermarks[event.partition_key] = watermark_after
        finalizable = self._finalizable_windows(event.partition_key, watermark_after)
        new_cursor = self.event_cursor + 1
        trace = WatermarkTrace(
            trace_id=self._trace_id(event),
            event_id=event.event_id,
            arrival_no=event.arrival_no,
            partition_key=event.partition_key,
            event_time_ms=event.event_time_ms,
            watermark_before_ms=watermark_before,
            watermark_after_ms=watermark_after,
            was_late=was_late,
            finalized_count=len(finalizable),
        )
        self.traces.append(trace)
        self.decisions.extend(event_decisions)
        self._processed_arrivals.add(event.arrival_no)
        self.event_cursor = new_cursor
        if self.ledger is not None:
            self.ledger.record_event(
                self.session_id,
                self.view_id,
                trace,
                event_decisions,
                self.event_cursor,
            )

        for key in finalizable:
            self._finalize_window(key)

        if self.logger is not None:
            self.logger.info(
                "OK",
                "event_ingested",
                "event adjudicated and watermark advanced",
                {
                    "event_id": event.event_id,
                    "arrival_no": event.arrival_no,
                    "partition_key": event.partition_key,
                    "watermark_before_ms": watermark_before,
                    "watermark_after_ms": watermark_after,
                    "was_late": was_late,
                    "accepted_rules": accepted_rules,
                    "finalized_count": len(finalizable),
                },
            )
        return {
            "event_id": event.event_id,
            "watermark_before_ms": watermark_before,
            "watermark_after_ms": watermark_after,
            "was_late": was_late,
            "accepted_rules": accepted_rules,
            "finalized_count": len(finalizable),
        }

    def _finalizable_windows(self, partition_key: str, watermark: int) -> list[WindowKey]:
        result: list[WindowKey] = []
        for key in self.open_windows:
            if key.partition_key != partition_key:
                continue
            if key.window_end_ms <= watermark - self._rule_lateness(key):
                result.append(key)
        result.sort(key=lambda key: (key.window_end_ms, key.window_start_ms, key.rule_id, key.rule_version))
        return result

    def _finalize_window(self, key: WindowKey) -> WindowResult:
        accumulator = self.open_windows.get(key)
        if accumulator is None:
            raise KeyError(key)
        if accumulator.count <= 0:
            del self.open_windows[key]
            raise BusinessError(
                code="E_INTERNAL",
                message="attempted to finalize an empty window",
                details=key.to_dict(),
            )
        attribution, sum_fixed, mean_fixed = build_error_attribution(
            count=accumulator.count,
            sum_exact=accumulator.sum_exact,
            quantization_error_sum=accumulator.quantization_error_sum,
            max_single_quantization_error=accumulator.max_single_quantization_error,
            late_accepted_abs_amount=accumulator.late_accepted_abs_amount,
            boundary_abs_amount=accumulator.boundary_abs_amount,
            threshold_bp=self.spec.dominant_threshold_bp,
        )
        result = WindowResult(
            key=key,
            count=accumulator.count,
            sum_fixed=sum_fixed,
            mean_fixed=mean_fixed,
            attribution=attribution,
            sum_exact=accumulator.sum_exact,
            late_accepted_count=accumulator.late_accepted_count,
            late_accepted_abs_amount=accumulator.late_accepted_abs_amount,
            boundary_abs_amount=accumulator.boundary_abs_amount,
        )
        pairs = self._ambiguity_pairs_for(result)
        self.finalized_windows[key] = result
        del self.open_windows[key]
        if self.ledger is not None:
            batch = self.ledger.commit_window_finalization(
                session_id=self.session_id,
                view_id=self.view_id,
                result=result,
                event_cursor=self.event_cursor,
                ambiguity_pairs=pairs,
            )
            self.last_batch = batch
            self.batch_index = int(batch["batch_no"])
            self.ledger_content_digest = str(batch["content_digest"])
        return result

    def _ambiguity_pairs_for(self, result: WindowResult) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        for other in self.finalized_windows.values():
            combined_bound = result.attribution.mean_bound + other.attribution.mean_bound
            difference = abs(result.mean_fixed - other.mean_fixed)
            if difference > combined_bound:
                continue
            if result.key.sort_key() <= other.key.sort_key():
                key_a, key_b = result.key, other.key
                mean_a, mean_b = result.mean_fixed, other.mean_fixed
            else:
                key_a, key_b = other.key, result.key
                mean_a, mean_b = other.mean_fixed, result.mean_fixed
            winner = key_a if key_a.sort_key() <= key_b.sort_key() else key_b
            pairs.append(
                {
                    "key_a": key_a.to_dict(),
                    "key_b": key_b.to_dict(),
                    "left_mean_fixed": mean_a,
                    "right_mean_fixed": mean_b,
                    "combined_mean_bound": combined_bound,
                    "stable_winner": winner.to_dict(),
                    "error_affected": True,
                }
            )
        pairs.sort(
            key=lambda pair: (
                pair["key_a"]["partition_key"],
                pair["key_a"]["window_start_ms"],
                pair["key_a"]["rule_id"],
                pair["key_b"]["partition_key"],
                pair["key_b"]["window_start_ms"],
                pair["key_b"]["rule_id"],
            )
        )
        return pairs

    def stable_ordered_results(self) -> list[WindowResult]:
        return sorted(
            self.finalized_windows.values(),
            key=lambda result: (
                result.mean_fixed,
                result.key.partition_key,
                result.key.window_end_ms,
                result.key.window_start_ms,
                result.key.rule_id,
                result.key.rule_version,
            ),
        )

    def ambiguity_report(self) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        results = sorted(self.finalized_windows.values(), key=lambda item: _key_sort_value(item.key))
        for left_index, left in enumerate(results):
            for right in results[left_index + 1 :]:
                combined = left.attribution.mean_bound + right.attribution.mean_bound
                if abs(left.mean_fixed - right.mean_fixed) > combined:
                    continue
                winner = left.key if left.key.sort_key() <= right.key.sort_key() else right.key
                pairs.append(
                    {
                        "key_a": left.key.to_dict(),
                        "key_b": right.key.to_dict(),
                        "left_mean_fixed": left.mean_fixed,
                        "right_mean_fixed": right.mean_fixed,
                        "combined_mean_bound": combined,
                        "stable_winner": winner.to_dict(),
                        "error_affected": True,
                    }
                )
        return pairs

    def stable_ambiguity_groups(self) -> list[dict[str, Any]]:
        results = sorted(self.finalized_windows.values(), key=lambda result: _key_sort_value(result.key))
        if not results:
            return []
        parent = list(range(len(results)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left_index, left in enumerate(results):
            for right_index in range(left_index + 1, len(results)):
                right = results[right_index]
                combined = left.attribution.mean_bound + right.attribution.mean_bound
                if abs(left.mean_fixed - right.mean_fixed) <= combined:
                    union(left_index, right_index)
        components: dict[int, list[WindowResult]] = {}
        for index, result in enumerate(results):
            components.setdefault(find(index), []).append(result)
        groups: list[dict[str, Any]] = []
        for members in components.values():
            ordered = sorted(members, key=lambda result: result.key.sort_key())
            groups.append(
                {
                    "error_affected": len(ordered) > 1,
                    "stable_order": [result.key.to_dict() for result in ordered],
                    "members": [
                        {
                            "key": result.key.to_dict(),
                            "mean_fixed": result.mean_fixed,
                            "mean_bound": result.attribution.mean_bound,
                        }
                        for result in ordered
                    ],
                }
            )
        groups.sort(
            key=lambda group: (
                min(member["mean_fixed"] for member in group["members"]),
                group["stable_order"][0]["partition_key"],
                group["stable_order"][0]["window_start_ms"],
                group["stable_order"][0]["rule_id"],
            )
        )
        return groups

    def evolve_spec(self, new_spec: SessionSpec, *, reason: str) -> dict[str, Any]:
        assert self.ledger is not None
        comparison_spec = new_spec.with_rules_version(self.spec.rules_version)
        if comparison_spec.to_dict() == self.spec.to_dict():
            raise BusinessError(
                code="E_SPEC",
                message="evolved specification is identical to the active specification",
                field="rules",
            )
        next_version = max(self.spec.rules_version + 1, new_spec.rules_version)
        next_spec = new_spec.with_rules_version(next_version)
        old_view_id = self.view_id
        new_view_id = make_view_id(next_spec)
        affected_windows = affected_window_keys(self.ledger.all_windows(self.session_id, old_view_id))
        drop_keys: set[tuple[str, str, int, int, int]] = set()
        for decision in self.ledger.current_decisions(self.session_id, old_view_id):
            if decision["decision"] == "DROP":
                drop_keys.add(
                    (
                        decision["rule_id"],
                        decision["partition_key"],
                        decision["window_start_ms"],
                        decision["window_end_ms"],
                        decision["rule_version"],
                    )
                )
        for rule_id, partition_key, start, end, version in sorted(drop_keys):
            affected_windows.append(
                {
                    "rule_id": rule_id,
                    "rule_version": version,
                    "partition_key": partition_key,
                    "window_start_ms": start,
                    "window_end_ms": end,
                }
            )
        seen_affected: set[tuple[Any, ...]] = set()
        deduplicated: list[dict[str, Any]] = []
        for item in affected_windows:
            identity = (
                item["rule_id"],
                item["rule_version"],
                item["partition_key"],
                item["window_start_ms"],
                item["window_end_ms"],
            )
            if identity in seen_affected:
                continue
            seen_affected.add(identity)
            deduplicated.append(item)
        affected_windows = deduplicated
        affected_windows.sort(
            key=lambda item: (
                item["partition_key"],
                item["window_start_ms"],
                item["window_end_ms"],
                item["rule_id"],
                item["rule_version"],
            )
        )
        affected_partitions = sorted({item["partition_key"] for item in affected_windows})
        _, generations = generate_session(next_spec)
        event_digests = {
            source_id: generation.digest for source_id, generation in generations.items()
        }
        digest = self.ledger.evolve_session(
            session_id=self.session_id,
            old_view_id=old_view_id,
            new_view_id=new_view_id,
            new_spec=next_spec,
            reason=reason,
            affected=affected_windows,
            event_digests=event_digests,
        )
        self.spec = next_spec
        self.view_id = new_view_id
        self.events = []
        self.source_digests = event_digests
        self._processed_arrivals.clear()
        self.watermarks.clear()
        self.open_windows.clear()
        self.finalized_windows.clear()
        self.unknown_windows.clear()
        self.traces.clear()
        self.decisions.clear()
        self.event_cursor = 0
        self.batch_index = 0
        self.last_batch = None
        self.ledger_content_digest = self.ledger.get_view_manifest(
            self.session_id, new_view_id
        )["content_digest"]
        self._rule_by_id = {rule.rule_id: rule for rule in next_spec.rules}
        return {
            "old_view_id": old_view_id,
            "new_view_id": new_view_id,
            "rules_version": next_spec.rules_version,
            "affected_windows": affected_windows,
            "affected_partitions": affected_partitions,
            "staleness_digest": digest,
        }

    def drop_decisions(self) -> list[LateDecision]:
        return [decision for decision in self.decisions if decision.decision == "DROP"]

    def accepted_late_decisions(self) -> list[LateDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.decision == "ACCEPT" and decision.was_late
        ]

    def summary(self) -> dict[str, Any]:
        results = sorted(self.finalized_windows.values(), key=lambda result: _key_sort_value(result.key))
        open_windows = sorted(self.open_windows.values(), key=lambda acc: _key_sort_value(acc.key))
        drops = sorted(
            self.drop_decisions(),
            key=lambda decision: (
                decision.partition_key,
                decision.window_start_ms,
                decision.rule_id,
                decision.arrival_no,
            ),
        )
        late_accepts = sorted(
            self.accepted_late_decisions(),
            key=lambda decision: (decision.arrival_no, decision.partition_key, decision.rule_id),
        )
        return {
            "format_version": FORMAT_VERSION,
            "session_id": self.session_id,
            "view_id": self.view_id,
            "rules_version": self.spec.rules_version,
            "fixed_scale": self.spec.fixed_scale,
            "event_cursor": self.event_cursor,
            "watermarks": dict(sorted(self.watermarks.items())),
            "finalized_windows": [result.to_dict() for result in results],
            "finalized_window_keys": [result.key.to_dict() for result in results],
            "open_window_count": len(self.open_windows),
            "unknown_window_count": len(self.unknown_windows),
            "drop_count": len(drops),
            "drop_decisions": [decision.to_dict() for decision in drops],
            "late_accept_count": len(late_accepts),
            "late_accept_decisions": [decision.to_dict() for decision in late_accepts],
            "ambiguity_pairs": self.ambiguity_report(),
            "ambiguity_groups": self.stable_ambiguity_groups(),
            "last_batch": self.last_batch,
            "ledger_content_digest": self.ledger_content_digest,
            "trace_count": len(self.traces),
            "decision_count": len(self.decisions),
        }

    def snapshot_state(self) -> dict[str, Any]:
        open_windows = []
        for accumulator in sorted(self.open_windows.values(), key=lambda item: _key_sort_value(item.key)):
            open_windows.append(
                {
                    "key": accumulator.key.to_dict(),
                    "count": accumulator.count,
                    "sum_exact": accumulator.sum_exact,
                    "quantization_error_sum": accumulator.quantization_error_sum,
                    "max_single_quantization_error": accumulator.max_single_quantization_error,
                    "late_accepted_count": accumulator.late_accepted_count,
                    "late_accepted_abs_amount": accumulator.late_accepted_abs_amount,
                    "boundary_abs_amount": accumulator.boundary_abs_amount,
                    "accepted_late_events": list(accumulator.accepted_late_events),
                    "boundary_events": list(accumulator.boundary_events),
                }
            )
        return {
            "format_version": FORMAT_VERSION,
            "session_id": self.session_id,
            "view_id": self.view_id,
            "spec": self.spec.to_dict(),
            "spec_digest": self.spec.spec_digest,
            "rules_version": self.spec.rules_version,
            "fixed_scale": self.spec.fixed_scale,
            "event_cursor": self.event_cursor,
            "processed_arrivals": sorted(self._processed_arrivals),
            "watermarks": dict(sorted(self.watermarks.items())),
            "open_windows": open_windows,
            "finalized_windows": [
                result.to_dict()
                for result in sorted(self.finalized_windows.values(), key=lambda item: _key_sort_value(item.key))
            ],
            "unknown_windows": [
                key.to_dict() for key in sorted(self.unknown_windows, key=_key_sort_value)
            ],
            "batch_index": self.batch_index,
            "last_batch": self.last_batch,
            "ledger_content_digest": self.ledger_content_digest,
            "source_digests": dict(sorted(self.source_digests.items())),
            "traces": [trace.to_dict() for trace in self.traces],
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    @classmethod
    def from_snapshot_state(
        cls,
        state: dict[str, Any],
        *,
        ledger: Ledger | None,
        logger: JsonlLogger | None = None,
        mark_pending_unknown: bool = True,
    ) -> tuple["Engine", list[dict[str, Any]]]:
        from .fixed import ErrorAttribution
        from .model import LateDecision, WatermarkTrace
        from .spec import parse_spec

        spec = parse_spec(state["spec"]).with_rules_version(int(state["rules_version"]))
        engine = cls(
            spec,
            session_id=state["session_id"],
            ledger=None,
            logger=logger,
            view_id=state["view_id"],
        )
        engine.ledger = ledger
        engine.event_cursor = int(state["event_cursor"])
        engine._processed_arrivals = set(int(value) for value in state.get("processed_arrivals", []))
        engine.watermarks = {str(key): int(value) for key, value in state.get("watermarks", {}).items()}
        engine.source_digests = dict(state.get("source_digests", {}))
        engine.batch_index = int(state.get("batch_index", 0))
        engine.last_batch = state.get("last_batch")
        engine.ledger_content_digest = state.get("ledger_content_digest", "")
        engine.traces = [WatermarkTrace(**row) for row in state.get("traces", [])]
        engine.decisions = [LateDecision(**row) for row in state.get("decisions", [])]
        for key_data in state.get("unknown_windows", []):
            engine.unknown_windows.add(WindowKey(**key_data))
        for row in state.get("finalized_windows", []):
            key = WindowKey(
                rule_id=row["rule_id"],
                rule_version=row["rule_version"],
                partition_key=row["partition_key"],
                window_start_ms=row["window_start_ms"],
                window_end_ms=row["window_end_ms"],
            )
            attribution_data = row["attribution"]
            attribution = ErrorAttribution(**attribution_data)
            engine.finalized_windows[key] = WindowResult(
                key=key,
                count=row["count"],
                sum_fixed=row["sum_fixed"],
                mean_fixed=row["mean_fixed"],
                attribution=attribution,
                sum_exact=row.get("sum_exact", row["sum_fixed"]),
                late_accepted_count=row.get("late_accepted_count", 0),
                late_accepted_abs_amount=row.get("late_accepted_abs_amount", 0),
                boundary_abs_amount=row.get("boundary_abs_amount", 0),
                sum_saturation_error=row.get("sum_saturation_error", 0),
                mean_saturation_error=row.get("mean_saturation_error", 0),
                status=row.get("status", "COMMITTED"),
            )
        pending_keys: list[dict[str, Any]] = []
        for row in state.get("open_windows", []):
            key = WindowKey(**row["key"])
            if mark_pending_unknown:
                engine.unknown_windows.add(key)
                pending_keys.append(key.to_dict())
                continue
            engine.open_windows[key] = WindowAccumulator(
                key=key,
                count=row["count"],
                sum_exact=row["sum_exact"],
                quantization_error_sum=row["quantization_error_sum"],
                max_single_quantization_error=row["max_single_quantization_error"],
                late_accepted_count=row["late_accepted_count"],
                late_accepted_abs_amount=row["late_accepted_abs_amount"],
                boundary_abs_amount=row["boundary_abs_amount"],
                accepted_late_events=list(row.get("accepted_late_events", [])),
                boundary_events=list(row.get("boundary_events", [])),
            )
        unknown_rows: list[dict[str, Any]] = []
        if ledger is not None and mark_pending_unknown:
            unknown_rows = ledger.mark_pending_unknown(state["session_id"], state["view_id"])
        return engine, unknown_rows

    def remaining_events(self) -> list[Event]:
        if not self.events:
            self.events, generations = generate_session(self.spec)
            self.source_digests = {
                source_id: generation.digest for source_id, generation in generations.items()
            }
        ordered = sorted(self.events, key=lambda event: (event.arrival_no, event.event_id))
        return [event for event in ordered if event.arrival_no not in self._processed_arrivals]

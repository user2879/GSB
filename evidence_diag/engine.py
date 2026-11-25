from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Callable, Iterable, Mapping, Sequence

from .fingerprint import (
    cache_key,
    channel_parameter_fingerprint,
    evidence_content_fingerprint,
    evidence_set_fingerprint,
    evidence_weight_fingerprint,
    topic_assignment_fingerprint,
    turn_content_fingerprint,
    turn_query_fingerprint,
)
from .models import (
    AppConfig,
    Evidence,
    FixedPointConfig,
    Session,
    TopicConfig,
    Turn,
)
from .stages import (
    compute_boundary,
    compute_confidence,
    compute_dedupe,
    compute_evidence_attribution,
    compute_recall,
    compute_topics,
)
from .util import sha256_json
from .validation import ERROR_CODE, issue, validate_config, validate_session


@dataclass
class EngineState:
    config: AppConfig
    session: Session
    command_cursor: int = 0
    version: int = 1
    failures: list[dict[str, Any]] | None = None
    last_result: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []


@dataclass
class ComputeOutcome:
    result: dict[str, Any]
    execution: dict[str, Any]


def config_fingerprint(config: AppConfig) -> str:
    return sha256_json(config.to_dict())


def evidence_library_fingerprint(session: Session) -> str:
    return sha256_json(
        [
            {
                "evidence_id": item.evidence_id,
                "text_fingerprint": evidence_content_fingerprint(item),
                "weight_fingerprint": evidence_weight_fingerprint(item),
            }
            for item in sorted(session.evidence, key=lambda item: item.evidence_id)
        ]
    )


def _config_hash(config: AppConfig) -> str:
    return config_fingerprint(config)


def _evidence_library_fingerprint(session: Session) -> str:
    return evidence_library_fingerprint(session)


def _topic_segment_signature(
    topic: Mapping[str, Any], turns_by_id: Mapping[int, Turn]
) -> str:
    return sha256_json(
        {
            "start_turn_id": topic["start_turn_id"],
            "end_turn_id": topic["end_turn_id"],
            "turn_queries": [
                turns_by_id[turn_id].query for turn_id in topic["turn_ids"]
            ],
        }
    )


def _evidence_attribution_signature(attribution: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "evidence_id": attribution["evidence_id"],
            "evidence_fingerprint": attribution["evidence_fingerprint"],
            "citing_turns": attribution["citing_turns"],
            "topic_ids": attribution["topic_ids"],
        }
    )


class DiagnosticEngine:
    def __init__(
        self,
        config: AppConfig,
        session: Session,
        *,
        command_cursor: int = 0,
        version: int = 1,
        failures: Iterable[Mapping[str, Any]] | None = None,
        last_result: Mapping[str, Any] | None = None,
        caches: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.session = session
        self.command_cursor = command_cursor
        self.version = version
        self.failures = [dict(item) for item in (failures or [])]
        self.last_result = dict(last_result) if last_result is not None else None
        cache_values = caches or {}
        self.caches: dict[str, dict[str, Any]] = {
            "recall": dict(cache_values.get("recall", {})),
            "dedupe": dict(cache_values.get("dedupe", {})),
            "boundary": dict(cache_values.get("boundary", {})),
            "confidence": dict(cache_values.get("confidence", {})),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DiagnosticEngine":
        return cls(
            AppConfig.from_dict(state["config"]),
            Session.from_dict(state["session"]),
            command_cursor=int(state.get("command_cursor", 0)),
            version=int(state.get("version", 1)),
            failures=state.get("failures", []),
            last_result=state.get("last_result"),
            caches=state.get("caches", {}),
        )

    def export_state(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "session": self.session.to_dict(),
            "command_cursor": self.command_cursor,
            "version": self.version,
            "failures": self.failures,
            "last_result": self.last_result,
            "caches": self.caches,
        }

    def _validate_current(self) -> list[dict[str, str]]:
        issues = validate_config(self.config)
        issues.extend(validate_session(self.config, self.session))
        return issues

    def _record_failure(
        self, seq: int, errors: Sequence[Mapping[str, Any]], op: str
    ) -> None:
        self.failures.append(
            {
                "seq": seq,
                "op": op,
                "code": ERROR_CODE,
                "isolated": True,
                "errors": [dict(error) for error in errors],
            }
        )

    def apply_command(self, command: Mapping[str, Any]) -> dict[str, Any]:
        seq = command.get("seq")
        op = str(command.get("op", ""))
        if seq != self.command_cursor + 1:
            self._record_failure(
                int(seq) if isinstance(seq, int) else self.command_cursor + 1,
                [
                    issue(
                        "seq",
                        f"expected command seq {self.command_cursor + 1}",
                    )
                ],
                op,
            )
            self.command_cursor = (
                int(seq) if isinstance(seq, int) else self.command_cursor + 1
            )
            self.version += 1
            return {"valid": False, "op": op, "seq": seq, "rollback": False}

        old_config = self.config
        old_session = self.session
        try:
            self._mutate(command)
            errors = self._validate_current()
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            errors = [issue(op or "command", str(exc))]
        if errors:
            self.config = old_config
            self.session = old_session
            self._record_failure(seq, errors, op)
            self.command_cursor = seq
            self.version += 1
            return {
                "valid": False,
                "op": op,
                "seq": seq,
                "rollback": True,
                "errors": errors,
            }
        self.command_cursor = seq
        self.version += 1
        return {"valid": True, "op": op, "seq": seq}

    def _mutate(self, command: Mapping[str, Any]) -> None:
        op = command.get("op")
        if op == "add_turn":
            raw = command.get("turn")
            if not isinstance(raw, Mapping):
                raise ValueError("add_turn requires an object field 'turn'")
            candidate = Turn.from_dict(raw)
            next_id = len(self.session.turns) + 1
            candidate = replace(candidate, turn_id=next_id)
            self.session = replace(
                self.session,
                turns=self.session.turns + (candidate,),
            )
            return
        if op == "replace_turn_query":
            turn = self._turn(int(command["turn_id"]))
            query = command["query"]
            if not isinstance(query, str):
                raise ValueError("query must be a string")
            self._replace_turn(replace(turn, query=query))
            return
        if op == "delete_turn":
            turn_id = int(command["turn_id"])
            self._turn(turn_id)
            remaining = [
                turn for turn in self.session.turns if turn.turn_id != turn_id
            ]
            if not remaining:
                raise ValueError("cannot delete the final remaining turn")
            reindexed = tuple(
                replace(turn, turn_id=index)
                for index, turn in enumerate(remaining, start=1)
            )
            self.session = replace(self.session, turns=reindexed)
            return
        if op == "set_topic_threshold":
            value = command.get("value_bp")
            self.config = replace(
                self.config,
                topic=TopicConfig(switch_threshold_bp=value),
            )
            return
        if op == "set_channel_weight":
            channel = str(command["channel"])
            weight = command["weight"]
            weights = dict(self.config.channel_weights)
            weights[channel] = Decimal(str(weight))
            self.config = replace(self.config, channel_weights=weights)
            return
        if op == "set_evidence_weight":
            evidence_id = str(command["evidence_id"])
            weight = Decimal(str(command["weight"]))
            self._replace_evidence(
                evidence_id, weight=weight, text=None, require=True
            )
            return
        if op == "set_evidence_text":
            evidence_id = str(command["evidence_id"])
            text = command["text"]
            if not isinstance(text, str):
                raise ValueError("text must be a string")
            self._replace_evidence(
                evidence_id, weight=None, text=text, require=True
            )
            return
        if op == "upsert_evidence":
            raw = command.get("evidence")
            if not isinstance(raw, Mapping):
                raise ValueError("upsert_evidence requires an object 'evidence'")
            candidate = Evidence.from_dict(raw)
            items = {
                item.evidence_id: item for item in self.session.evidence
            }
            items[candidate.evidence_id] = candidate
            self.session = replace(
                self.session,
                evidence=tuple(items[key] for key in sorted(items)),
            )
            return
        if op == "set_channel_declaration":
            turn = self._turn(int(command["turn_id"]))
            channel = str(command["channel"])
            values = command.get("evidence_ids")
            if not isinstance(values, list):
                raise ValueError("evidence_ids must be a list")
            channels = dict(turn.channels)
            channels[channel] = tuple(str(value) for value in values)
            self._replace_turn(replace(turn, channels=channels))
            return
        if op == "set_fixed_point":
            fixed = FixedPointConfig.from_dict(command)
            self.config = self.config.with_fixed_point(fixed)
            return
        if op == "set_budget":
            value = command.get("value")
            limits = replace(self.config.limits, evidence_budget=value)
            self.config = self.config.with_limits(limits)
            return
        if op == "set_max_workers":
            value = command.get("value")
            limits = replace(self.config.limits, max_workers=value)
            self.config = self.config.with_limits(limits)
            return
        if op == "set_answer_threshold":
            value = Decimal(str(command["value"]))
            confidence = replace(
                self.config.confidence, answer_threshold=value
            )
            self.config = replace(self.config, confidence=confidence)
            return
        if op == "set_turn_decay":
            value = Decimal(str(command["value"]))
            confidence = replace(self.config.confidence, turn_decay=value)
            self.config = replace(self.config, confidence=confidence)
            return
        if op == "set_cross_topic_penalty":
            value = Decimal(str(command["value"]))
            confidence = replace(
                self.config.confidence, cross_topic_penalty=value
            )
            self.config = replace(self.config, confidence=confidence)
            return
        raise ValueError(f"unsupported command op {op!r}")

    def _turn(self, turn_id: int) -> Turn:
        for turn in self.session.turns:
            if turn.turn_id == turn_id:
                return turn
        raise KeyError(f"unknown turn_id {turn_id}")

    def _replace_turn(self, candidate: Turn) -> None:
        self.session = replace(
            self.session,
            turns=tuple(
                candidate if turn.turn_id == candidate.turn_id else turn
                for turn in self.session.turns
            ),
        )

    def _replace_evidence(
        self,
        evidence_id: str,
        *,
        weight: Decimal | None,
        text: str | None,
        require: bool,
    ) -> None:
        found = False
        evidence_items: list[Evidence] = []
        for item in self.session.evidence:
            if item.evidence_id == evidence_id:
                found = True
                evidence_items.append(
                    replace(
                        item,
                        weight=item.weight if weight is None else weight,
                        text=item.text if text is None else text,
                    )
                )
            else:
                evidence_items.append(item)
        if require and not found:
            raise KeyError(f"unknown evidence_id {evidence_id}")
        self.session = replace(self.session, evidence=tuple(evidence_items))

    def invalid_result_after_failure(self) -> dict[str, Any]:
        if self.last_result is None:
            raise RuntimeError("no successful computation to annotate")
        result = dict(self.last_result)
        result["version"] = self.version
        result["command_cursor"] = self.command_cursor
        result["failures"] = [dict(item) for item in self.failures]
        result["invalidation"] = {
            "command_cursor": self.command_cursor,
            "reason": "command_rejected_no_state_change",
            "invalidated_turns": [],
            "reused_turns": [
                turn["turn_id"] for turn in self.last_result.get("turns", [])
            ],
            "invalidated_topic_ids": [],
            "reused_topic_ids": [
                topic["topic_id"] for topic in self.last_result.get("topics", [])
            ],
            "invalidated_evidence_ids": [],
            "reused_evidence_ids": [
                item["evidence_id"]
                for item in self.last_result.get("evidence_attribution", [])
            ],
            "reasons_by_turn": {},
        }
        result["config_hash"] = _config_hash(self.config)
        result["evidence_library_fingerprint"] = _evidence_library_fingerprint(
            self.session
        )
        self.last_result = result
        return result

    def _submit_stage(
        self,
        stage: str,
        tasks: Sequence[tuple[str, str, Callable[[], Any], bool]],
        cache: dict[str, Any],
        force_recompute: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run stable-keyed tasks and return payloads plus execution counters."""
        payloads: dict[str, Any] = {}
        hits: list[str] = []
        misses: list[str] = []
        sort_keys_by_cache_key: dict[str, list[str]] = {}
        task_by_cache_key: dict[str, tuple[Callable[[], Any], bool]] = {}
        for sort_key, cache_key_value, fn, cacheable in sorted(tasks):
            sort_keys_by_cache_key.setdefault(cache_key_value, []).append(sort_key)
            task_by_cache_key.setdefault(cache_key_value, (fn, cacheable))
            if cacheable and not force_recompute and cache_key_value in cache:
                for key in sort_keys_by_cache_key[cache_key_value]:
                    payloads[key] = cache[cache_key_value]
                hits.append(cache_key_value)
            else:
                misses.append(cache_key_value)

        miss_keys = sorted(set(misses))

        def run_one(key: str) -> tuple[str, Any, bool]:
            fn, cacheable = task_by_cache_key[key]
            return key, fn(), cacheable

        if miss_keys:
            with ThreadPoolExecutor(
                max_workers=self.config.limits.max_workers,
                thread_name_prefix=f"evidence-{stage}",
            ) as executor:
                futures = [
                    executor.submit(run_one, key) for key in miss_keys
                ]
                for future in futures:
                    key, payload, cacheable = future.result()
                    for sort_key in sort_keys_by_cache_key[key]:
                        payloads[sort_key] = payload
                    if cacheable:
                        cache[key] = payload
        execution = {
            "stage": stage,
            "max_workers": self.config.limits.max_workers,
            "task_count": len(tasks),
            "cache_hits": len(hits),
            "cache_misses": len(misses),
            "sorted_task_keys": [item[0] for item in sorted(tasks)],
        }
        return payloads, execution

    def compute(self, *, force_recompute: bool = False) -> ComputeOutcome:
        domain = self.config.fixed_point.domain()
        evidence_map = self.session.evidence_map()
        turns = self.session.turns
        previous = self.last_result or {}
        previous_turn_stage_keys = {
            int(turn["turn_id"]): turn.get("stage_keys", {})
            for turn in previous.get("turns", [])
        }
        previous_status = {
            int(turn["turn_id"]): turn.get("status", "OK")
            for turn in previous.get("turns", [])
        }
        previous_topics = {
            topic["topic_id"]: topic for topic in previous.get("topics", [])
        }
        previous_evidence_keys = {
            item["evidence_id"]: item.get("attribution_key")
            for item in previous.get("evidence_attribution", [])
        }
        execution_stages: list[dict[str, Any]] = []

        content_fps: dict[int, str] = {}
        query_fps: dict[int, str] = {}
        recall_keys: dict[int, str] = {}
        recall_tasks: list[tuple[str, str, Callable[[], Any], bool]] = []
        for turn in turns:
            content_fp = turn_content_fingerprint(turn, evidence_map)
            query_fp = turn_query_fingerprint(turn)
            content_fps[turn.turn_id] = content_fp
            query_fps[turn.turn_id] = query_fp
            key = cache_key(
                "recall",
                {
                    "turn_content_fingerprint": content_fp,
                    "query_fingerprint": query_fp,
                    "channel_parameters": channel_parameter_fingerprint(
                        self.config, turn.channels
                    ),
                    "domain": domain.fingerprint(),
                },
            )
            recall_keys[turn.turn_id] = key
            recall_tasks.append(
                (
                    f"{turn.turn_id:08d}:recall",
                    key,
                    lambda turn=turn: compute_recall(
                        turn, self.config, evidence_map, domain
                    ),
                    True,
                )
            )
        recall_payloads, execution = self._submit_stage(
            "recall", recall_tasks, self.caches["recall"], force_recompute
        )
        execution_stages.append(execution)

        dedupe_keys: dict[int, str] = {}
        dedupe_tasks: list[tuple[str, str, Callable[[], Any], bool]] = []
        for turn in turns:
            recall_payload = recall_payloads[f"{turn.turn_id:08d}:recall"]
            key = cache_key(
                "dedupe",
                {
                    "recall_fingerprint": sha256_json(recall_payload),
                    "domain": domain.fingerprint(),
                },
            )
            dedupe_keys[turn.turn_id] = key
            dedupe_tasks.append(
                (
                    f"{turn.turn_id:08d}:dedupe",
                    key,
                    lambda recall_payload=recall_payload: compute_dedupe(
                        recall_payload
                    ),
                    True,
                )
            )
        dedupe_payloads, execution = self._submit_stage(
            "dedupe", dedupe_tasks, self.caches["dedupe"], force_recompute
        )
        execution_stages.append(execution)
        dedupe_by_turn = {
            turn.turn_id: dedupe_payloads[f"{turn.turn_id:08d}:dedupe"]
            for turn in turns
        }

        boundary_keys: dict[tuple[int, int], str] = {}
        boundary_tasks: list[tuple[str, str, Callable[[], Any], bool]] = []
        for index in range(len(turns) - 1):
            left = turns[index]
            right = turns[index + 1]
            left_dedupe = dedupe_payloads[f"{left.turn_id:08d}:dedupe"]
            right_dedupe = dedupe_payloads[f"{right.turn_id:08d}:dedupe"]
            key = cache_key(
                "topic-boundary",
                {
                    "left_query_fingerprint": query_fps[left.turn_id],
                    "right_query_fingerprint": query_fps[right.turn_id],
                    "left_evidence_set": evidence_set_fingerprint(
                        left_dedupe, evidence_map
                    ),
                    "right_evidence_set": evidence_set_fingerprint(
                        right_dedupe, evidence_map
                    ),
                    "threshold_bp": self.config.topic.switch_threshold_bp,
                    "domain": domain.fingerprint(),
                },
            )
            boundary_keys[(left.turn_id, right.turn_id)] = key
            boundary_tasks.append(
                (
                    f"{left.turn_id:08d}:{right.turn_id:08d}:topic-boundary",
                    key,
                    lambda left=left, right=right, left_dedupe=left_dedupe, right_dedupe=right_dedupe: compute_boundary(
                        left, right, left_dedupe, right_dedupe, self.config
                    ),
                    True,
                )
            )
        boundary_payloads, execution = self._submit_stage(
            "topic-boundary",
            boundary_tasks,
            self.caches["boundary"],
            force_recompute,
        )
        execution_stages.append(execution)
        boundaries = [
            {
                **boundary_payloads[
                    f"{left_id:08d}:{right_id:08d}:topic-boundary"
                ],
                "left_turn_id": left_id,
                "right_turn_id": right_id,
            }
            for left_id, right_id in boundary_keys
        ]

        topics, turn_topics, turn_topic_records = compute_topics(
            turns, dedupe_by_turn, boundaries, evidence_map
        )
        evidence_attribution = compute_evidence_attribution(
            turns, dedupe_by_turn, turn_topics, evidence_map
        )
        attribution_keys: dict[str, str] = {}
        for evidence_id, attribution in evidence_attribution.items():
            attribution_keys[evidence_id] = cache_key(
                "evidence-attribution",
                {
                    "evidence_fingerprint": attribution["evidence_fingerprint"],
                    "citations": [
                        {
                            "turn_content_fingerprint": content_fps[turn_id],
                            "topic_id": turn_topics[turn_id],
                        }
                        for turn_id in attribution["citing_turns"]
                    ],
                    "domain": domain.fingerprint(),
                },
            )

        topic_stage_keys: dict[int, str] = {}
        for turn in turns:
            dedupe_payload = dedupe_payloads[f"{turn.turn_id:08d}:dedupe"]
            topic_record = turn_topic_records[turn.turn_id]
            topic_stage_keys[turn.turn_id] = topic_assignment_fingerprint(
                turn_content_fp=content_fps[turn.turn_id],
                topic_id=topics[
                    next(
                        index
                        for index, topic in enumerate(topics)
                        if topic["topic_id"] == turn_topics[turn.turn_id]
                    )
                ]["topic_id"],
                topic_start_content_fp=topic_record[
                    "topic_start_content_fingerprint"
                ],
                evidence_set_fp=evidence_set_fingerprint(
                    dedupe_payload, evidence_map
                ),
                evidence_fps=[
                    evidence_content_fingerprint(
                        evidence_map[item["evidence_id"]]
                    )
                    for item in dedupe_payload["evidence"]
                ],
            )

        confidence_keys: dict[int, str] = {}
        confidence_basis: dict[int, dict[str, Any]] = {}
        for turn in turns:
            dedupe_payload = dedupe_payloads[f"{turn.turn_id:08d}:dedupe"]
            evidence_ids = sorted(
                item["evidence_id"] for item in dedupe_payload["evidence"]
            )
            basis = {
                "turn_content_fingerprint": content_fps[turn.turn_id],
                "dedupe_fingerprint": sha256_json(dedupe_payload),
                "topic_assignment_fingerprint": topic_stage_keys[turn.turn_id],
                "evidence_parameters": [
                    {
                        "evidence_id": evidence_id,
                        "content": evidence_content_fingerprint(
                            evidence_map[evidence_id]
                        ),
                        "weight": evidence_weight_fingerprint(
                            evidence_map[evidence_id]
                        ),
                    }
                    for evidence_id in evidence_ids
                ],
                "attribution": [
                    {
                        "evidence_id": evidence_id,
                        "topic_ids": evidence_attribution[evidence_id][
                            "topic_ids"
                        ],
                        "cross_topic": evidence_attribution[evidence_id][
                            "cross_topic"
                        ],
                        "citing_turns": evidence_attribution[evidence_id][
                            "citing_turns"
                        ],
                    }
                    for evidence_id in evidence_ids
                ],
                "confidence_parameters": self.config.confidence.to_dict(),
                "topic": {
                    "topic_id": turn_topics[turn.turn_id],
                    "start_turn_id": turn_topic_records[turn.turn_id][
                        "start_turn_id"
                    ],
                    "end_turn_id": turn_topic_records[turn.turn_id][
                        "end_turn_id"
                    ],
                    "offset": turn_topic_records[turn.turn_id]["offset"],
                },
                "domain": domain.fingerprint(),
            }
            confidence_basis[turn.turn_id] = basis
            confidence_keys[turn.turn_id] = cache_key("confidence", basis)

        previous_successful_confidence_keys = {
            info.get("confidence")
            for turn_id, info in previous_turn_stage_keys.items()
            if previous_status.get(turn_id, "OK") == "OK"
        }
        pending_from_previous = {
            int(item["turn_id"])
            for item in previous.get("todo", [])
            if item.get("code") == "E_BUDGET_PENDING"
        }
        due_turns: list[int] = []
        cached_unreported_turns: set[int] = set()
        for turn in turns:
            confidence_key = confidence_keys[turn.turn_id]
            if confidence_key in previous_successful_confidence_keys:
                continue
            if confidence_key in self.caches["confidence"]:
                cached_unreported_turns.add(turn.turn_id)
                continue
            if turn.turn_id in pending_from_previous:
                due_turns.append(turn.turn_id)
                continue
            due_turns.append(turn.turn_id)
        due_turns.sort()
        remaining_budget = self.config.limits.evidence_budget
        scheduled_turns: set[int] = set()
        pending_entries: list[dict[str, Any]] = []
        cost_by_turn: dict[int, int] = {}
        for turn in turns:
            cost = max(1, len(dedupe_payloads[f"{turn.turn_id:08d}:dedupe"]["evidence"]))
            cost_by_turn[turn.turn_id] = cost
        for turn_id in due_turns:
            cost = cost_by_turn[turn_id]
            if cost <= remaining_budget:
                scheduled_turns.add(turn_id)
                remaining_budget -= cost
            else:
                pending_entries.append(
                    {
                        "turn_id": turn_id,
                        "stage": "confidence",
                        "code": "E_BUDGET_PENDING",
                        "cost": cost,
                        "confidence_key": confidence_keys[turn_id],
                    }
                )

        confidence_tasks: list[tuple[str, str, Callable[[], Any], bool]] = []
        for turn in turns:
            key = confidence_keys[turn.turn_id]
            turn_topic = turn_topic_records[turn.turn_id]
            dedupe_payload = dedupe_payloads[f"{turn.turn_id:08d}:dedupe"]
            if (
                turn.turn_id in scheduled_turns
                or key in previous_successful_confidence_keys
                or turn.turn_id in cached_unreported_turns
            ):
                confidence_tasks.append(
                    (
                        f"{turn.turn_id:08d}:confidence",
                        key,
                        lambda turn=turn, dedupe_payload=dedupe_payload, turn_topic=turn_topic: compute_confidence(
                            turn,
                            dedupe_payload,
                            self.config,
                            domain,
                            evidence_map,
                            evidence_attribution,
                            turn_topic,
                        ),
                        True,
                    )
                )
        confidence_payloads, execution = self._submit_stage(
            "confidence",
            confidence_tasks,
            self.caches["confidence"],
            force_recompute,
        )
        execution_stages.append(execution)

        turns_output: list[dict[str, Any]] = []
        for turn in turns:
            dedupe_payload = dedupe_payloads[f"{turn.turn_id:08d}:dedupe"]
            topic_record = turn_topic_records[turn.turn_id]
            reason_stages: list[str] = []
            old_keys = previous_turn_stage_keys.get(turn.turn_id, {})
            current_keys = {
                "recall": recall_keys[turn.turn_id],
                "dedupe": dedupe_keys[turn.turn_id],
                "topic": topic_stage_keys[turn.turn_id],
                "confidence": confidence_keys[turn.turn_id],
            }
            for stage in ("recall", "dedupe", "topic", "confidence"):
                if old_keys.get(stage) != current_keys[stage]:
                    reason_stages.append(stage)
            confidence_payload = confidence_payloads.get(
                f"{turn.turn_id:08d}:confidence"
            )
            if confidence_payload is None:
                status = "E_BUDGET_PENDING"
                confidence_output = {
                    "status": "E_BUDGET_PENDING",
                    "confidence_units": None,
                    "confidence": None,
                    "conclusion": "pending",
                    "conclusion_bool": None,
                    "citation_chain": [],
                    "accumulation_order": [],
                }
                if previous_status.get(turn.turn_id) != status:
                    reason_stages.append("budget")
            else:
                status = "OK"
                confidence_output = confidence_payload
            boundary_before = None
            if turn.turn_id > 1 and turn.turn_id - 2 < len(boundaries):
                boundary_before = boundaries[turn.turn_id - 2]
            turns_output.append(
                {
                    "turn_id": turn.turn_id,
                    "query": turn.query,
                    "query_fingerprint": query_fps[turn.turn_id],
                    "content_fingerprint": content_fps[turn.turn_id],
                    "topic_id": turn_topics[turn.turn_id],
                    "topic_record": topic_record,
                    "boundary_before": boundary_before,
                    "recall_key": recall_keys[turn.turn_id],
                    "dedupe_key": dedupe_keys[turn.turn_id],
                    "topic_key": topic_stage_keys[turn.turn_id],
                    "confidence_key": confidence_keys[turn.turn_id],
                    "stage_keys": current_keys,
                    "recall": recall_payloads[
                        f"{turn.turn_id:08d}:recall"
                    ],
                    "dedupe": dedupe_payload,
                    "confidence": confidence_output,
                    "status": status,
                }
            )

        topic_signatures = {
            topic["topic_id"]: _topic_segment_signature(topic, {
                turn.turn_id: turn for turn in turns
            })
            for topic in topics
        }
        invalidated_turns: list[int] = []
        reused_turns: list[int] = []
        reasons_by_turn: dict[str, list[str]] = {}
        for turn_output in turns_output:
            reasons = reasons_by_turn.get(str(turn_output["turn_id"]), [])
            old_keys = previous_turn_stage_keys.get(
                int(turn_output["turn_id"]), {}
            )
            for stage, current_key in turn_output["stage_keys"].items():
                if old_keys.get(stage) != current_key and stage not in reasons:
                    reasons.append(stage)
            if (
                previous_status.get(int(turn_output["turn_id"]))
                != turn_output["status"]
            ):
                reasons.append("status")
            if reasons:
                invalidated_turns.append(turn_output["turn_id"])
                reasons_by_turn[str(turn_output["turn_id"])] = sorted(set(reasons))
            else:
                reused_turns.append(turn_output["turn_id"])

        invalidated_topic_ids: set[str] = set()
        rebuilt_topic_ids: set[str] = set()
        for topic in topics:
            signature = topic_signatures[topic["topic_id"]]
            old = previous_topics.get(topic["topic_id"])
            if old is None or old.get("segment_signature") != signature:
                rebuilt_topic_ids.add(topic["topic_id"])
        for topic_id in previous_topics:
            if topic_id not in topic_signatures:
                invalidated_topic_ids.add(topic_id)
        for turn_output in turns_output:
            if turn_output["status"] != "OK" or turn_output["turn_id"] in invalidated_turns:
                invalidated_topic_ids.add(turn_output["topic_id"])
        invalidated_topic_ids.update(rebuilt_topic_ids)
        reused_topic_ids = [
            topic_id
            for topic_id in topic_signatures
            if topic_id not in invalidated_topic_ids
        ]
        for topic in topics:
            topic["segment_signature"] = topic_signatures[topic["topic_id"]]

        invalidated_evidence_ids: list[str] = []
        reused_evidence_ids: list[str] = []
        attribution_output: list[dict[str, Any]] = []
        for evidence_id in sorted(evidence_attribution):
            item = dict(evidence_attribution[evidence_id])
            item["attribution_key"] = attribution_keys[evidence_id]
            attribution_output.append(item)
            if previous_evidence_keys.get(evidence_id) != attribution_keys[evidence_id]:
                invalidated_evidence_ids.append(evidence_id)
            else:
                reused_evidence_ids.append(evidence_id)

        todo = sorted(pending_entries, key=lambda item: (item["turn_id"], item["code"]))
        result = {
            "schema_version": 1,
            "session_id": self.session.session_id,
            "version": self.version,
            "command_cursor": self.command_cursor,
            "config_hash": _config_hash(self.config),
            "evidence_library_fingerprint": _evidence_library_fingerprint(
                self.session
            ),
            "fixed_point": domain.describe(),
            "turn_count": len(turns),
            "topics": topics,
            "turns": turns_output,
            "evidence_attribution": attribution_output,
            "failures": [dict(item) for item in self.failures],
            "todo": todo,
            "config": self.config.to_dict(),
            "invalidation": {
                "command_cursor": self.command_cursor,
                "invalidated_turns": sorted(invalidated_turns),
                "reused_turns": sorted(reused_turns),
                "invalidated_topic_ids": sorted(invalidated_topic_ids),
                "reused_topic_ids": sorted(reused_topic_ids),
                "invalidated_evidence_ids": sorted(invalidated_evidence_ids),
                "reused_evidence_ids": sorted(reused_evidence_ids),
                "reasons_by_turn": reasons_by_turn,
            },
            "stage_versions": {
                "recall": sha256_json(
                    {
                        turn_id: recall_keys[turn_id]
                        for turn_id in sorted(recall_keys)
                    }
                ),
                "dedupe": sha256_json(
                    {
                        turn_id: dedupe_keys[turn_id]
                        for turn_id in sorted(dedupe_keys)
                    }
                ),
                "topic": sha256_json(
                    {
                        turn_id: topic_stage_keys[turn_id]
                        for turn_id in sorted(topic_stage_keys)
                    }
                ),
                "confidence": sha256_json(
                    {
                        turn_id: confidence_keys[turn_id]
                        for turn_id in sorted(confidence_keys)
                    }
                ),
            },
            "boundary_keys": [
                boundary_keys[key] for key in boundary_keys
            ],
        }
        self.last_result = result
        execution = {
            "mode": "full" if force_recompute else "incremental",
            "stages": execution_stages,
            "budget": {
                "limit": self.config.limits.evidence_budget,
                "remaining": remaining_budget,
                "scheduled_turns": sorted(scheduled_turns),
                "pending_turns": [item["turn_id"] for item in todo],
            },
        }
        return ComputeOutcome(result=result, execution=execution)


def run_sequence(
    config: AppConfig,
    session: Session,
    commands: Sequence[Mapping[str, Any]],
    *,
    mode: str = "incremental",
) -> dict[str, Any]:
    if mode not in ("incremental", "full"):
        raise ValueError("mode must be 'incremental' or 'full'")
    engine = DiagnosticEngine(config, session)
    initial = engine.compute(force_recompute=(mode == "full"))
    trace: list[dict[str, Any]] = [
        {
            "cursor": 0,
            "result": initial.result,
            "execution": initial.execution,
        }
    ]
    for command in commands:
        outcome = engine.apply_command(command)
        if outcome["valid"]:
            computed = engine.compute(force_recompute=(mode == "full"))
            result = computed.result
            execution = computed.execution
        else:
            result = engine.invalid_result_after_failure()
            execution = {
                "mode": mode,
                "stages": [],
                "command_rejected": True,
            }
        trace.append(
            {
                "cursor": engine.command_cursor,
                "result": result,
                "execution": execution,
            }
        )
    return {
        "mode": mode,
        "state": engine.export_state(),
        "trace": trace,
        "final_result": trace[-1]["result"],
    }

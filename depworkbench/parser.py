from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import sha256_json, stable_task_key
from .errors import ValidationError
from .models import (
    ACTION_CODES,
    ACTION_NAMES,
    Config,
    Model,
    QuantizedModel,
    Sentence,
)


INITIAL_STATE_TOKEN = "INITIAL_STATE"


@dataclass(frozen=True)
class ParseState:
    stack: tuple[int, ...]
    buffer_index: int
    heads: tuple[int, ...]
    relations: tuple[str | None, ...]
    children_count: tuple[int, ...]
    actions: tuple[str, ...]
    total_score: int
    finished: bool = False

    @property
    def action_codes(self) -> tuple[int, ...]:
        return tuple(ACTION_CODES[action] for action in self.actions)


@dataclass
class ParseOutcome:
    sentence_id: str
    mode: str
    steps: list[dict[str, Any]]
    final_state: ParseState
    completed: bool
    error_code: str | None = None
    error_message: str | None = None
    first_invalid_step: int | None = None
    invalidation_reason: str | None = None
    ledger: dict[str, Any] | None = None
    pruned_paths: list[dict[str, Any]] | None = None
    pruned_count_by_reason: dict[str, int] | None = None
    pruned_omitted_count: int = 0
    search_stats: dict[str, Any] | None = None
    cache: dict[str, Any] | None = None


def initial_state(token_count: int) -> ParseState:
    return ParseState(
        stack=(),
        buffer_index=0,
        heads=tuple([-1] * token_count),
        relations=tuple([None] * token_count),
        children_count=tuple([0] * token_count),
        actions=(),
        total_score=0,
        finished=False,
    )


def state_to_dict(state: ParseState) -> dict[str, Any]:
    return {
        "stack": list(state.stack),
        "buffer_index": state.buffer_index,
        "heads": list(state.heads),
        "relations": list(state.relations),
        "children_count": list(state.children_count),
        "actions": [ACTION_CODES[action] for action in state.actions],
        "action_names": list(state.actions),
        "total_score": state.total_score,
        "finished": state.finished,
    }


def state_from_dict(value: dict[str, Any]) -> ParseState:
    actions = value.get("action_names")
    if actions is None:
        actions = [ACTION_NAMES[int(code)] for code in value.get("actions", [])]
    return ParseState(
        stack=tuple(int(item) for item in value.get("stack", [])),
        buffer_index=int(value.get("buffer_index", 0)),
        heads=tuple(int(item) for item in value.get("heads", [])),
        relations=tuple(value.get("relations", [])),
        children_count=tuple(int(item) for item in value.get("children_count", [])),
        actions=tuple(str(item) for item in actions),
        total_score=int(value.get("total_score", 0)),
        finished=bool(value.get("finished", False)),
    )


def state_fingerprint(state: ParseState) -> str:
    return sha256_json(state_to_dict(state))


def _token_ref(index: int, sentence: Sentence) -> list[str]:
    return [sentence.words[index], sentence.pos_tags[index]]


def _context_slots(state: ParseState, sentence: Sentence) -> dict[str, int | None]:
    return {
        "s0": state.stack[-1] if state.stack else None,
        "s1": state.stack[-2] if len(state.stack) >= 2 else None,
        "b0": state.buffer_index if state.buffer_index < len(sentence.words) else None,
    }


def _feature_entries(
    action: str,
    state: ParseState,
    sentence: Sentence,
    model: Model,
    quantized: QuantizedModel,
) -> list[dict[str, Any]]:
    slots = _context_slots(state, sentence)
    feature_keys: list[str] = [f"a:{action}|bias"]
    if slots["s0"] is not None:
        index = int(slots["s0"])
        feature_keys.extend(
            (
                f"a:{action}|w:s0={sentence.words[index]}",
                f"a:{action}|p:s0={sentence.pos_tags[index]}",
            )
        )
    if slots["s1"] is not None:
        index = int(slots["s1"])
        feature_keys.extend(
            (
                f"a:{action}|w:s1={sentence.words[index]}",
                f"a:{action}|p:s1={sentence.pos_tags[index]}",
            )
        )
    if slots["b0"] is not None:
        index = int(slots["b0"])
        feature_keys.extend(
            (
                f"a:{action}|w:b0={sentence.words[index]}",
                f"a:{action}|p:b0={sentence.pos_tags[index]}",
            )
        )
    if slots["s0"] is not None and slots["b0"] is not None:
        s0_index = int(slots["s0"])
        b0_index = int(slots["b0"])
        feature_keys.append(
            f"a:{action}|pair={sentence.pos_tags[s0_index]}_{sentence.pos_tags[b0_index]}"
        )
    entries: list[dict[str, Any]] = []
    for key in feature_keys:
        bucket = model.feature_hashes.get(key)
        if bucket is None:
            continue
        weight_index = bucket % model.feature_dim
        entries.append(
            {
                "feature_key": key,
                "bucket": bucket,
                "weight_index": weight_index,
                "weight_value": quantized.weights[action][weight_index],
            }
        )
    return entries


def _action_score(entries: Iterable[dict[str, Any]]) -> int:
    return sum(int(entry["weight_value"]) for entry in entries)


def _effective_arc_distance(config: Config) -> int:
    if config.constraints.max_arc_distance is None:
        return config.max_dependency_distance
    return min(config.max_dependency_distance, config.constraints.max_arc_distance)


def _feasible_actions(
    state: ParseState,
    sentence: Sentence,
    config: Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    token_count = len(sentence.words)
    stack = state.stack
    buffer_index = state.buffer_index
    max_distance = _effective_arc_distance(config)
    feasible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    def exclude(action: str, reason: str, detail: str) -> None:
        excluded.append(
            {
                "action": action,
                "code": ACTION_CODES[action],
                "reason": reason,
                "detail": detail,
            }
        )

    shift_reason: tuple[str, str] | None = None
    if buffer_index >= token_count:
        shift_reason = ("E_SHIFT_BUFFER_EMPTY", "buffer has no unprocessed token")
    elif (
        config.constraints.max_stack_size is not None
        and len(stack) + 1 > config.constraints.max_stack_size
    ):
        shift_reason = (
            "E_SHIFT_STACK_LIMIT",
            f"stack size would exceed {config.constraints.max_stack_size}",
        )
    if shift_reason is None:
        feasible.append({"action": "SHIFT", "code": ACTION_CODES["SHIFT"]})
    else:
        exclude("SHIFT", *shift_reason)

    left_reason: tuple[str, str] | None = None
    if not stack:
        left_reason = ("E_LEFT_STACK_EMPTY", "stack is empty")
    elif buffer_index >= token_count:
        left_reason = ("E_LEFT_BUFFER_EMPTY", "buffer has no head token")
    else:
        dependent = stack[-1]
        head = buffer_index
        distance = abs(head - dependent)
        if distance > max_distance:
            left_reason = (
                "E_LEFT_DISTANCE_LIMIT",
                f"arc distance {distance} exceeds {max_distance}",
            )
        elif sentence.pos_tags[dependent] in config.constraints.forbid_left_dependent_pos:
            left_reason = (
                "E_LEFT_POS_FORBIDDEN",
                f"dependent POS {sentence.pos_tags[dependent]!r} is forbidden",
            )
        elif (
            config.constraints.max_dependents_per_head is not None
            and state.children_count[head] >= config.constraints.max_dependents_per_head
        ):
            left_reason = (
                "E_LEFT_MAX_DEPENDENTS",
                f"head already has {state.children_count[head]} dependents",
            )
    if left_reason is None:
        feasible.append({"action": "LEFT_ARC", "code": ACTION_CODES["LEFT_ARC"]})
    else:
        exclude("LEFT_ARC", *left_reason)

    right_reason: tuple[str, str] | None = None
    if len(stack) < 2:
        right_reason = ("E_RIGHT_STACK_TOO_SMALL", "right arc requires at least two stack tokens")
    else:
        head = stack[-2]
        dependent = stack[-1]
        distance = abs(head - dependent)
        if distance > max_distance:
            right_reason = (
                "E_RIGHT_DISTANCE_LIMIT",
                f"arc distance {distance} exceeds {max_distance}",
            )
        elif sentence.pos_tags[dependent] in config.constraints.forbid_right_dependent_pos:
            right_reason = (
                "E_RIGHT_POS_FORBIDDEN",
                f"dependent POS {sentence.pos_tags[dependent]!r} is forbidden",
            )
        elif (
            config.constraints.max_dependents_per_head is not None
            and state.children_count[head] >= config.constraints.max_dependents_per_head
        ):
            right_reason = (
                "E_RIGHT_MAX_DEPENDENTS",
                f"head already has {state.children_count[head]} dependents",
            )
    if right_reason is None:
        feasible.append({"action": "RIGHT_ARC", "code": ACTION_CODES["RIGHT_ARC"]})
    else:
        exclude("RIGHT_ARC", *right_reason)

    finish_reason: tuple[str, str] | None = None
    if buffer_index < token_count:
        finish_reason = ("E_FINISH_BUFFER_NOT_EMPTY", "buffer still contains tokens")
    elif config.constraints.finish_requires_single_root and len(stack) != 1:
        finish_reason = (
            "E_FINISH_STACK_NOT_SINGLE",
            f"finish requires one root but stack has {len(stack)} tokens",
        )
    elif not stack:
        finish_reason = ("E_FINISH_STACK_EMPTY", "cannot finish without a root")
    if finish_reason is None:
        feasible.append({"action": "FINISH", "code": ACTION_CODES["FINISH"]})
    else:
        exclude("FINISH", *finish_reason)

    feasible.sort(key=lambda item: int(item["code"]))
    excluded.sort(key=lambda item: int(item["code"]))
    return feasible, excluded


def _score_feasible_actions(
    state: ParseState,
    sentence: Sentence,
    model: Model,
    quantized: QuantizedModel,
    feasible: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    entry_by_action: dict[str, list[dict[str, Any]]] = {}
    for item in feasible:
        action = str(item["action"])
        entries = _feature_entries(action, state, sentence, model, quantized)
        entry_by_action[action] = entries
        scored.append(
            {
                "action": action,
                "code": int(item["code"]),
                "score": _action_score(entries),
            }
        )
    return scored, [
        {"action": action, "entries": entries}
        for action, entries in sorted(entry_by_action.items(), key=lambda pair: ACTION_CODES[pair[0]])
    ]


def _token_view(index: int, sentence: Sentence) -> dict[str, Any]:
    return {
        "position": index,
        "word": sentence.words[index],
        "pos": sentence.pos_tags[index],
    }


def _stack_view(state: ParseState, sentence: Sentence) -> list[dict[str, Any]]:
    return [_token_view(index, sentence) for index in state.stack]


def _buffer_view(state: ParseState, sentence: Sentence) -> list[dict[str, Any]]:
    return [
        _token_view(index, sentence)
        for index in range(state.buffer_index, len(sentence.words))
    ]


def _apply_action(
    state: ParseState,
    action: str,
    score: int,
    sentence: Sentence,
) -> ParseState:
    stack = list(state.stack)
    heads = list(state.heads)
    relations = list(state.relations)
    children = list(state.children_count)
    buffer_index = state.buffer_index
    total_score = state.total_score + score
    finished = state.finished
    if action == "SHIFT":
        stack.append(buffer_index)
        buffer_index += 1
    elif action == "LEFT_ARC":
        head = buffer_index
        dependent = stack.pop()
        heads[dependent] = head
        relations[dependent] = "L"
        children[head] += 1
    elif action == "RIGHT_ARC":
        head = stack[-2]
        dependent = stack.pop()
        heads[dependent] = head
        relations[dependent] = "R"
        children[head] += 1
    elif action == "FINISH":
        finished = True
    else:
        raise ValueError(f"unknown action {action}")
    return ParseState(
        stack=tuple(stack),
        buffer_index=buffer_index,
        heads=tuple(heads),
        relations=tuple(relations),
        children_count=tuple(children),
        actions=state.actions + (action,),
        total_score=total_score,
        finished=finished,
    )


def _feasible_fingerprint(
    scored: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> str:
    return sha256_json(
        {
            "feasible": [
                {"action": item["action"], "code": item["code"], "score": item["score"]}
                for item in scored
            ],
            "excluded": [
                {"action": item["action"], "reason": item["reason"]} for item in excluded
            ],
        }
    )


def _tie_actions(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not scored:
        return []
    best = max(int(item["score"]) for item in scored)
    return [item for item in scored if int(item["score"]) == best]


def _step_cache_metadata(
    state_before: ParseState,
    sentence: Sentence,
    feature_entries: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    config: Config,
    model: Model,
) -> tuple[dict[str, Any], dict[str, Any]]:
    slots = _context_slots(state_before, sentence)
    referenced_indexes: set[int] = {
        int(index) for index in slots.values() if index is not None
    }
    token_refs = {
        str(index): _token_ref(index, sentence) for index in sorted(referenced_indexes)
    }
    feature_hashes: dict[str, int] = {}
    weight_cells: dict[str, int] = {}
    for action_entry in feature_entries:
        action = str(action_entry["action"])
        for entry in action_entry["entries"]:
            feature_hashes[str(entry["feature_key"])] = int(entry["bucket"])
            key = f"{action}:{int(entry['weight_index'])}"
            weight_cells[key] = int(entry["weight_value"])
    metadata = {
        "token_refs": token_refs,
        "feature_hashes": feature_hashes,
        "weight_cells": weight_cells,
        "model_shape": [model.feature_dim, model.num_buckets],
        "quantization_fingerprint": config.quantization_fingerprint,
        "mode": config.mode,
        "max_transition_steps": config.max_transition_steps,
        "constraints_fingerprint": parse_constraints_fingerprint(config),
    }
    local_model = sha256_json(
        {
            "feature_hashes": feature_hashes,
            "weight_cells": weight_cells,
            "model_shape": [model.feature_dim, model.num_buckets],
        }
    )
    visible_sentence = sha256_json(
        {
            "sentence_id": sentence.sentence_id,
            "tokens": [[index, *token_refs[str(index)]] for index in sorted(referenced_indexes)],
        }
    )
    return metadata, {
        "visible_sentence_fingerprint": visible_sentence,
        "local_model_fingerprint": local_model,
    }


def _cache_key(
    state_before: ParseState,
    step_index: int,
    prior_step_key: str,
    feasible_fingerprint: str,
    fingerprints: dict[str, Any],
    config: Config,
) -> str:
    return sha256_json(
        {
            "step_index": step_index,
            "state": state_fingerprint(state_before),
            "prior_step_key": prior_step_key,
            "feasible_action_fingerprint": feasible_fingerprint,
            "visible_sentence_fingerprint": fingerprints["visible_sentence_fingerprint"],
            "local_model_fingerprint": fingerprints["local_model_fingerprint"],
            "quantization_fingerprint": config.quantization_fingerprint,
            "mode": config.mode,
        }
    )


def _build_step(
    state: ParseState,
    sentence: Sentence,
    model: Model,
    quantized: QuantizedModel,
    config: Config,
    selected_action: str,
    prior_step_key: str,
    step_index: int,
) -> tuple[dict[str, Any], ParseState]:
    feasible, excluded = _feasible_actions(state, sentence, config)
    scored, feature_entries = _score_feasible_actions(
        state, sentence, model, quantized, feasible
    )
    action_lookup = {str(item["action"]): item for item in scored}
    if selected_action not in action_lookup:
        raise ValidationError(
            "E_PARSE_DEAD_END",
            f"selected action {selected_action} is not feasible at step {step_index}",
        )
    selected = action_lookup[selected_action]
    selected_score = int(selected["score"])
    after = _apply_action(state, selected_action, selected_score, sentence)
    feasible_fp = _feasible_fingerprint(scored, excluded)
    metadata, fingerprints = _step_cache_metadata(
        state, sentence, feature_entries, excluded, config, model
    )
    key = _cache_key(state, step_index, prior_step_key, feasible_fp, fingerprints, config)
    ties = _tie_actions(scored)
    step = {
        "step_index": step_index,
        "task_key": stable_task_key(sentence.sentence_id, step_index, "transition"),
        "action": selected_action,
        "action_code": ACTION_CODES[selected_action],
        "score": selected_score,
        "total_score_after": after.total_score,
        "feasible_actions": scored,
        "excluded_actions": excluded,
        "tied_best_actions": [item["action"] for item in ties],
        "feasible_action_fingerprint": feasible_fp,
        "stack_before": _stack_view(state, sentence),
        "buffer_before": _buffer_view(state, sentence),
        "stack_after": _stack_view(after, sentence),
        "buffer_after": _buffer_view(after, sentence),
        "state_before": state_to_dict(state),
        "state_after": state_to_dict(after),
        "cache_key": key,
        "references_prior_step": prior_step_key,
        "cache": metadata,
        "cache_status": "RECOMPUTED",
    }
    return step, after


def _check_feature_declarations(sentence: Sentence, model: Model) -> None:
    for position, (word, pos) in enumerate(zip(sentence.words, sentence.pos_tags)):
        if f"w:{word}" not in model.feature_hashes:
            raise ValidationError(
                "E_FEATURE_UNDECLARED",
                f"word feature {word!r} at token {position} is not declared in feature_hashes",
            )
        if f"p:{pos}" not in model.feature_hashes:
            raise ValidationError(
                "E_FEATURE_UNDECLARED",
                f"POS feature {pos!r} at token {position} is not declared in feature_hashes",
            )


def _greedy_select(scored: list[dict[str, Any]]) -> dict[str, Any]:
    if not scored:
        raise ValidationError("E_PARSE_DEAD_END", "no feasible action")
    return sorted(scored, key=lambda item: (-int(item["score"]), int(item["code"])))[0]


def _empty_ledger(reason: str) -> dict[str, Any]:
    return {
        "first_invalid_step": 0,
        "reason": reason,
        "reused_steps": [],
        "recomputed_steps": [],
        "invalidated_steps": [],
        "reused_count": 0,
        "recomputed_count": 0,
        "invalidated_count": 0,
    }


def _first_invalid_step(
    old_cache: dict[str, Any],
    sentence: Sentence,
    model: Model,
    quantized: QuantizedModel,
    config: Config,
) -> tuple[int | None, str | None]:
    if config.mode == "exhaustive":
        old_environment = old_cache.get("exhaustive_environment_fingerprint")
        current_environment = _exhaustive_environment_fingerprint(sentence, model, config)
        if old_environment != current_environment:
            return 0, "EXHAUSTIVE_SEARCH_INPUT_CHANGED"
    if old_cache.get("mode") != config.mode:
        return 0, "MODE_CHANGED"
    if old_cache.get("quantization_fingerprint") != config.quantization_fingerprint:
        return 0, "QUANTIZATION_CHANGED"
    if old_cache.get("model_shape") != [model.feature_dim, model.num_buckets]:
        return 0, "MODEL_SHAPE_CHANGED"
    if old_cache.get("token_count") != len(sentence.words):
        return 0, "TOKEN_COUNT_CHANGED"
    constraints_changed = old_cache.get("constraints_fingerprint") != parse_constraints_fingerprint(config)
    for step in old_cache.get("steps", []):
        index = int(step["step_index"])
        if index >= config.max_transition_steps:
            return index, "MAX_TRANSITION_STEPS_CHANGED"
        metadata = step.get("cache", {})
        for position, values in metadata.get("token_refs", {}).items():
            token_index = int(position)
            if token_index >= len(sentence.words):
                return index, "TOKEN_CHANGED"
            if [sentence.words[token_index], sentence.pos_tags[token_index]] != values:
                return index, "TOKEN_CHANGED"
        for key, old_bucket in metadata.get("feature_hashes", {}).items():
            if model.feature_hashes.get(key) != old_bucket:
                return index, "FEATURE_HASH_CHANGED"
        for cell_key, old_value in metadata.get("weight_cells", {}).items():
            action, raw_index = cell_key.rsplit(":", 1)
            weight_index = int(raw_index)
            if weight_index >= model.feature_dim:
                return index, "MODEL_SHAPE_CHANGED"
            if quantized.weights[action][weight_index] != old_value:
                return index, "MODEL_WEIGHT_CHANGED"
        if constraints_changed:
            before = state_from_dict(step["state_before"])
            feasible, excluded = _feasible_actions(before, sentence, config)
            old_feasible = [
                [str(item["action"]), int(item["code"])]
                for item in step.get("feasible_actions", [])
            ]
            new_feasible = [[str(item["action"]), int(item["code"])] for item in feasible]
            old_excluded = {
                str(item["action"]): str(item["reason"])
                for item in step.get("excluded_actions", [])
            }
            new_excluded = {
                str(item["action"]): str(item["reason"]) for item in excluded
            }
            if old_feasible != new_feasible or old_excluded != new_excluded:
                return index, "CONSTRAINT_CHANGED"
    return None, None


def _reuse_outcome(
    sentence: Sentence,
    old_cache: dict[str, Any],
    reason: str | None = None,
) -> ParseOutcome:
    steps = copy.deepcopy(old_cache.get("steps", []))
    for step in steps:
        step["cache_status"] = "REUSED"
        _refresh_reused_step_views(step, sentence)
    ledger = {
        "first_invalid_step": None,
        "reason": reason or "NO_CHANGE",
        "reused_steps": [
            {"step_index": int(step["step_index"]), "action": step["action"], "cache_key": step["cache_key"]}
            for step in steps
        ],
        "recomputed_steps": [],
        "invalidated_steps": [],
        "reused_count": len(steps),
        "recomputed_count": 0,
        "invalidated_count": 0,
    }
    refreshed_cache = copy.deepcopy(old_cache)
    refreshed_cache["sentence_fingerprint"] = sentence.fingerprint
    refreshed_cache["steps"] = copy.deepcopy(steps)
    return ParseOutcome(
        sentence_id=sentence.sentence_id,
        mode=str(old_cache["mode"]),
        steps=steps,
        final_state=state_from_dict(old_cache["final_state"]),
        completed=bool(old_cache.get("completed", False)),
        error_code=old_cache.get("error_code"),
        error_message=old_cache.get("error_message"),
        ledger=ledger,
        pruned_paths=copy.deepcopy(old_cache.get("pruned_paths", [])),
        pruned_count_by_reason=copy.deepcopy(old_cache.get("pruned_count_by_reason", {})),
        pruned_omitted_count=int(old_cache.get("pruned_omitted_count", 0)),
        search_stats=copy.deepcopy(old_cache.get("search_stats", {})),
        cache=refreshed_cache,
    )


def _refresh_reused_step_views(step: dict[str, Any], sentence: Sentence) -> None:
    """Refresh presentation snapshots without recomputing any transition.

    A step's cache validity only depends on the tokens it references. The
    report nevertheless shows the full stack and remaining buffer, so a change
    in a not-yet-referenced token must be reflected in those display views.
    """

    before = state_from_dict(step["state_before"])
    after = state_from_dict(step["state_after"])
    step["stack_before"] = _stack_view(before, sentence)
    step["buffer_before"] = _buffer_view(before, sentence)
    step["stack_after"] = _stack_view(after, sentence)
    step["buffer_after"] = _buffer_view(after, sentence)


def parse_sentence(
    sentence: Sentence,
    model: Model,
    quantized: QuantizedModel,
    config: Config,
    old_cache: dict[str, Any] | None = None,
) -> ParseOutcome:
    _check_feature_declarations(sentence, model)
    start_step = 0
    invalidation_reason = "COLD_START"
    old_steps: list[dict[str, Any]] = []
    if old_cache is not None:
        first_invalid, reason = _first_invalid_step(
            old_cache, sentence, model, quantized, config
        )
        if first_invalid is None:
            if bool(old_cache.get("completed", False)):
                return _reuse_outcome(sentence, old_cache)
            start_step = len(old_cache.get("steps", []))
            invalidation_reason = "RESUME_CURSOR"
            old_steps = list(old_cache.get("steps", []))
        else:
            start_step = int(first_invalid)
            invalidation_reason = reason or "INVALIDATED"
            old_steps = list(old_cache.get("steps", []))

    if config.mode == "exhaustive" and old_cache is not None and start_step > 0:
        # A globally optimal path can change before the first locally changed
        # score. Correctness takes precedence over prefix reuse in this mode.
        start_step = 0
        invalidation_reason = "EXHAUSTIVE_GLOBAL_REOPTIMIZATION"

    reused_steps = copy.deepcopy(old_steps[:start_step])
    for step in reused_steps:
        step["cache_status"] = "REUSED"
        _refresh_reused_step_views(step, sentence)
    reused_count = len(reused_steps)
    invalidated_steps = [
        {
            "step_index": int(step["step_index"]),
            "action": str(step["action"]),
            "reason": invalidation_reason if index == start_step else "DOWNSTREAM_INVALIDATED",
        }
        for index, step in enumerate(old_steps[start_step:], start=start_step)
    ]
    if not old_steps:
        invalidation_reason = "COLD_START"

    if reused_steps:
        state = state_from_dict(reused_steps[-1]["state_after"])
        prior_key = str(reused_steps[-1]["cache_key"])
    else:
        state = initial_state(len(sentence.words))
        prior_key = INITIAL_STATE_TOKEN

    steps = reused_steps
    if config.mode == "greedy":
        outcome, steps = _parse_greedy_forward(
            sentence,
            model,
            quantized,
            config,
            state,
            steps,
            prior_key,
            start_step,
        )
    else:
        if reused_steps and not invalidated_steps:
            # This branch is normally handled by _reuse_outcome; keeping it
            # explicit makes an interrupted exhaustive cursor recoverable.
            outcome, steps = _parse_exhaustive(
                sentence, model, quantized, config, seed_steps=steps, seed_state=state
            )
        else:
            outcome, steps = _parse_exhaustive(sentence, model, quantized, config)

    recomputed_steps = [
        {"step_index": int(step["step_index"]), "action": step["action"], "cache_key": step["cache_key"]}
        for step in steps[start_step:]
    ]
    ledger = {
        "first_invalid_step": start_step if old_cache is not None else None,
        "reason": invalidation_reason if old_cache is not None else "COLD_START",
        "reused_steps": [
            {"step_index": int(step["step_index"]), "action": step["action"], "cache_key": step["cache_key"]}
            for step in steps[:reused_count]
        ],
        "recomputed_steps": recomputed_steps,
        "invalidated_steps": invalidated_steps,
        "reused_count": reused_count,
        "recomputed_count": len(recomputed_steps),
        "invalidated_count": len(invalidated_steps),
    }
    outcome = ParseOutcome(
        sentence_id=outcome.sentence_id,
        mode=outcome.mode,
        steps=steps,
        final_state=outcome.final_state,
        completed=outcome.completed,
        error_code=outcome.error_code,
        error_message=outcome.error_message,
        first_invalid_step=start_step if old_cache is not None else None,
        invalidation_reason=ledger["reason"],
        ledger=ledger,
        pruned_paths=outcome.pruned_paths,
        pruned_count_by_reason=outcome.pruned_count_by_reason,
        pruned_omitted_count=outcome.pruned_omitted_count,
        search_stats=outcome.search_stats,
        cache=None,
    )
    outcome.cache = _make_cache(outcome, sentence, model, config)
    return outcome


def _parse_greedy_forward(
    sentence: Sentence,
    model: Model,
    quantized: QuantizedModel,
    config: Config,
    state: ParseState,
    steps: list[dict[str, Any]],
    prior_key: str,
    start_step: int,
) -> tuple[ParseOutcome, list[dict[str, Any]]]:
    step_index = start_step
    while step_index < config.max_transition_steps:
        feasible, _excluded = _feasible_actions(state, sentence, config)
        if not feasible:
            return (
                ParseOutcome(
                    sentence_id=sentence.sentence_id,
                    mode=config.mode,
                    steps=steps,
                    final_state=state,
                    completed=False,
                    error_code="E_PARSE_DEAD_END",
                    error_message=f"no feasible action at transition step {step_index}",
                ),
                steps,
            )
        scored, _entries = _score_feasible_actions(
            state, sentence, model, quantized, feasible
        )
        selected = _greedy_select(scored)
        step, state = _build_step(
            state,
            sentence,
            model,
            quantized,
            config,
            str(selected["action"]),
            prior_key,
            step_index,
        )
        steps.append(step)
        prior_key = str(step["cache_key"])
        step_index += 1
        if state.finished:
            return (
                ParseOutcome(
                    sentence_id=sentence.sentence_id,
                    mode=config.mode,
                    steps=steps,
                    final_state=state,
                    completed=True,
                ),
                steps,
            )
    return (
        ParseOutcome(
            sentence_id=sentence.sentence_id,
            mode=config.mode,
            steps=steps,
            final_state=state,
            completed=False,
            error_code="E_MAX_TRANSITION_STEPS",
            error_message=f"maximum transition steps {config.max_transition_steps} exhausted",
        ),
        steps,
    )


def _parse_exhaustive(
    sentence: Sentence,
    model: Model,
    quantized: QuantizedModel,
    config: Config,
    *,
    seed_steps: list[dict[str, Any]] | None = None,
    seed_state: ParseState | None = None,
) -> tuple[ParseOutcome, list[dict[str, Any]]]:
    initial = seed_state if seed_state is not None else initial_state(len(sentence.words))
    best_score = -10**30
    best_codes: tuple[int, ...] | None = None
    complete_paths = 0
    nodes = 0
    aborted = False
    best_prefix: dict[tuple[Any, ...], tuple[int, tuple[int, ...]]] = {}
    pruned_records: list[dict[str, Any]] = []
    pruned_counts: dict[str, int] = {}
    pruned_omitted = 0

    def add_pruned(state: ParseState, reason: str, detail: str) -> None:
        nonlocal pruned_omitted
        pruned_counts[reason] = pruned_counts.get(reason, 0) + 1
        if len(pruned_records) >= config.max_pruned_records:
            pruned_omitted += 1
            return
        pruned_records.append(
            {
                "step_index": len(state.actions),
                "action_codes": list(state.action_codes),
                "action_names": list(state.actions),
                "stack": list(state.stack),
                "buffer_index": state.buffer_index,
                "prefix_score": state.total_score,
                "reason": reason,
                "detail": detail,
            }
        )

    def visit(state: ParseState) -> None:
        nonlocal nodes, complete_paths, best_score, best_codes, aborted
        if aborted:
            return
        nodes += 1
        if nodes > config.max_exhaustive_nodes:
            aborted = True
            add_pruned(state, "E_EXHAUSTIVE_NODE_LIMIT", f"node limit {config.max_exhaustive_nodes} reached")
            return
        # FINISH does not change stack/buffer/children, so the completion flag
        # must be part of the dominance key; otherwise the terminal state is
        # incorrectly dominated by the non-terminal state immediately before it.
        state_key = (
            state.stack,
            state.buffer_index,
            state.children_count,
            state.finished,
        )
        prior = best_prefix.get(state_key)
        current = (state.total_score, state.action_codes)
        if prior is not None:
            if prior[0] > current[0] or (prior[0] == current[0] and prior[1] <= current[1]):
                add_pruned(
                    state,
                    "E_DOMINATED_PARTIAL_PATH",
                    "an equal-or-better prefix reaches the same transition state",
                )
                return
        best_prefix[state_key] = current
        if state.finished:
            complete_paths += 1
            if best_codes is None or current[0] > best_score or (
                current[0] == best_score and current[1] < best_codes
            ):
                best_score = current[0]
                best_codes = current[1]
            return
        if len(state.actions) >= config.max_transition_steps:
            add_pruned(
                state,
                "E_MAX_TRANSITION_STEPS",
                f"maximum transition steps {config.max_transition_steps} reached before FINISH",
            )
            return
        feasible, _excluded = _feasible_actions(state, sentence, config)
        if not feasible:
            add_pruned(state, "E_PARSE_DEAD_END", "no feasible action")
            return
        scored, _entries = _score_feasible_actions(
            state, sentence, model, quantized, feasible
        )
        for item in sorted(scored, key=lambda row: int(row["code"])):
            if aborted:
                return
            next_state = _apply_action(state, str(item["action"]), int(item["score"]), sentence)
            visit(next_state)

    visit(initial)
    search_stats = {
        "nodes": nodes,
        "complete_paths": complete_paths,
        "dominated_states": len(best_prefix),
        "node_limit": config.max_exhaustive_nodes,
        "node_limit_hit": aborted,
    }
    if best_codes is None or aborted:
        error_code = "E_EXHAUSTIVE_NODE_LIMIT" if aborted else "E_PARSE_DEAD_END"
        outcome = ParseOutcome(
            sentence_id=sentence.sentence_id,
            mode=config.mode,
            steps=copy.deepcopy(seed_steps or []),
            final_state=initial,
            completed=False,
            error_code=error_code,
            error_message="exhaustive search found no complete path",
            pruned_paths=pruned_records,
            pruned_count_by_reason=pruned_counts,
            pruned_omitted_count=pruned_omitted,
            search_stats=search_stats,
        )
        return outcome, outcome.steps

    steps = copy.deepcopy(seed_steps or [])
    state = initial
    prior_key = str(steps[-1]["cache_key"]) if steps else INITIAL_STATE_TOKEN
    base_index = len(steps)
    for offset, code in enumerate(best_codes):
        action = ACTION_NAMES[code]
        step, state = _build_step(
            state,
            sentence,
            model,
            quantized,
            config,
            action,
            prior_key,
            base_index + offset,
        )
        steps.append(step)
        prior_key = str(step["cache_key"])
    outcome = ParseOutcome(
        sentence_id=sentence.sentence_id,
        mode=config.mode,
        steps=steps,
        final_state=state,
        completed=True,
        pruned_paths=pruned_records,
        pruned_count_by_reason=pruned_counts,
        pruned_omitted_count=pruned_omitted,
        search_stats=search_stats,
    )
    return outcome, steps


def _make_cache(outcome: ParseOutcome, sentence: Sentence, model: Model, config: Config) -> dict[str, Any]:
    cache_steps = copy.deepcopy(outcome.steps)
    return {
        "schema_version": 1,
        "sentence_id": sentence.sentence_id,
        "sentence_fingerprint": sentence.fingerprint,
        "token_count": len(sentence.words),
        "mode": outcome.mode,
        "model_shape": [model.feature_dim, model.num_buckets],
        "quantization_fingerprint": config.quantization_fingerprint,
        "constraints_fingerprint": parse_constraints_fingerprint(config),
        "exhaustive_environment_fingerprint": _exhaustive_environment_fingerprint(
            sentence, model, config
        ),
        "steps": cache_steps,
        "final_state": state_to_dict(outcome.final_state),
        "completed": outcome.completed,
        "error_code": outcome.error_code,
        "error_message": outcome.error_message,
        "pruned_paths": copy.deepcopy(outcome.pruned_paths or []),
        "pruned_count_by_reason": copy.deepcopy(outcome.pruned_count_by_reason or {}),
        "pruned_omitted_count": outcome.pruned_omitted_count,
        "search_stats": copy.deepcopy(outcome.search_stats or {}),
    }


def _exhaustive_environment_fingerprint(
    sentence: Sentence,
    model: Model,
    config: Config,
) -> str:
    return sha256_json(
        {
            "sentence_fingerprint": sentence.fingerprint,
            "model_fingerprint": model.fingerprint,
            "parse_config_fingerprint": config_parse_fingerprint(config),
        }
    )


def sentence_load_key(sentence: Sentence, model: Model, config: Config) -> str:
    return sha256_json(
        {
            "sentence_fingerprint": sentence.fingerprint,
            "model_fingerprint": model.fingerprint,
            "parse_config_fingerprint": config_parse_fingerprint(config),
            "quantization_fingerprint": config.quantization_fingerprint,
        }
    )


def config_parse_fingerprint(config: Config) -> str:
    """Fingerprint only fields that can affect parser products."""

    return sha256_json(
        {
            "mode": config.mode,
            "max_dependency_distance": config.max_dependency_distance,
            "max_transition_steps": config.max_transition_steps,
            "fixed_point_decimals": config.fixed_point_decimals,
            "quantization_level": config.quantization_level,
            "constraints": config.constraints.to_dict(),
            "max_exhaustive_nodes": config.max_exhaustive_nodes,
            "max_pruned_records": config.max_pruned_records,
        }
    )


def parse_constraints_fingerprint(config: Config) -> str:
    return sha256_json(
        {
            "max_dependency_distance": config.max_dependency_distance,
            "max_transition_steps": config.max_transition_steps,
            "constraints": config.constraints.to_dict(),
        }
    )


def step_summary_digest(steps: list[dict[str, Any]]) -> str:
    return sha256_json(
        [
            {
                "step": step["step_index"],
                "feasible": step["feasible_actions"],
                "excluded": step["excluded_actions"],
                "selected": step["action"],
                "score": step["score"],
            }
            for step in steps
        ]
    )

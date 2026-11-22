from __future__ import annotations

from typing import Any

from .canonical import sha256_json
from .models import Config, Sentence
from .parser import ParseOutcome, ParseState, state_to_dict


def build_arcs(outcome: ParseOutcome, sentence: Sentence) -> list[dict[str, Any]]:
    state = outcome.final_state
    arcs: list[dict[str, Any]] = []
    for dependent, head in enumerate(state.heads):
        if head < 0:
            continue
        relation = state.relations[dependent] or ""
        arcs.append(
            {
                "sentence_id": sentence.sentence_id,
                "head": head,
                "dependent": dependent,
                "relation": relation,
                "head_word": sentence.words[head],
                "head_pos": sentence.pos_tags[head],
                "dependent_word": sentence.words[dependent],
                "dependent_pos": sentence.pos_tags[dependent],
                "distance": abs(head - dependent),
            }
        )
    arcs.sort(key=lambda arc: (arc["head"], arc["dependent"]))
    return arcs


def build_flat_tree(outcome: ParseOutcome, sentence: Sentence) -> dict[str, Any]:
    state = outcome.final_state
    roots = [index for index, head in enumerate(state.heads) if head < 0]
    nodes: list[dict[str, Any]] = []
    children_by_head: dict[int, list[int]] = {}
    for dependent, head in enumerate(state.heads):
        if head >= 0:
            children_by_head.setdefault(head, []).append(dependent)
    for index, (word, pos) in enumerate(zip(sentence.words, sentence.pos_tags)):
        nodes.append(
            {
                "position": index,
                "word": word,
                "pos": pos,
                "head": state.heads[index],
                "relation": state.relations[index],
                "children": sorted(children_by_head.get(index, [])),
            }
        )
    return {
        "root_positions": roots,
        "root_position": roots[0] if len(roots) == 1 else None,
        "nodes": nodes,
    }


def build_nested_tree(
    outcome: ParseOutcome,
    sentence: Sentence,
    *,
    root_position: int | None,
) -> dict[str, Any] | None:
    if root_position is None:
        return None
    state = outcome.final_state
    children_by_head: dict[int, list[int]] = {}
    for dependent, head in enumerate(state.heads):
        if head >= 0:
            children_by_head.setdefault(head, []).append(dependent)

    def node(position: int, depth: int) -> dict[str, Any]:
        return {
            "position": position,
            "word": sentence.words[position],
            "pos": sentence.pos_tags[position],
            "relation": state.relations[position],
            "depth": depth,
            "children": [
                node(child, depth + 1)
                for child in sorted(children_by_head.get(position, []))
            ],
        }

    return node(root_position, 0)


def reference_metrics(
    outcome: ParseOutcome,
    sentence: Sentence,
) -> dict[str, Any]:
    if sentence.gold_heads is None:
        return {"available": False}
    predicted_heads = list(outcome.final_state.heads)
    predicted_relations = list(outcome.final_state.relations)
    gold_heads = list(sentence.gold_heads)
    if sentence.gold_relations is None:
        gold_relations: list[str] = []
        for position, head in enumerate(gold_heads):
            if head < 0:
                gold_relations.append("ROOT")
            elif head > position:
                gold_relations.append("L")
            else:
                gold_relations.append("R")
    else:
        gold_relations = list(sentence.gold_relations)
    token_count = len(sentence.words)
    head_matches = sum(1 for pred, gold in zip(predicted_heads, gold_heads) if pred == gold)
    las_matches = 0
    for position in range(token_count):
        if predicted_heads[position] == gold_heads[position] and predicted_relations[position] == gold_relations[position]:
            las_matches += 1
    return {
        "available": True,
        "tokens": token_count,
        "head_matches": head_matches,
        "las_matches": las_matches,
        "uas": round(head_matches / token_count, 6) if token_count else 0.0,
        "las": round(las_matches / token_count, 6) if token_count else 0.0,
        "exact_match": head_matches == token_count and las_matches == token_count,
        "gold_heads": gold_heads,
        "gold_relations": gold_relations,
    }


def build_sentence_products(
    outcome: ParseOutcome,
    sentence: Sentence,
    config: Config,
    sentence_load_key: str,
) -> dict[str, Any]:
    arcs = build_arcs(outcome, sentence)
    flat_tree = build_flat_tree(outcome, sentence)
    nested_tree = build_nested_tree(
        outcome,
        sentence,
        root_position=flat_tree["root_position"],
    )
    final_step_key = str(outcome.steps[-1]["cache_key"]) if outcome.steps else "NO_STEPS"
    arc_key = sha256_json(
        {
            "sentence_load_key": sentence_load_key,
            "final_step_key": final_step_key,
            "arcs": arcs,
            "quantization_fingerprint": config.quantization_fingerprint,
        }
    )
    tree_key = sha256_json(
        {
            "arc_key": arc_key,
            "flat_tree": flat_tree,
            "nested_tree": nested_tree,
        }
    )
    return {
        "arcs": arcs,
        "flat_tree": flat_tree,
        "nested_tree": nested_tree,
        "arc_key": arc_key,
        "tree_key": tree_key,
        "reference_metrics": reference_metrics(outcome, sentence),
        "final_state": state_to_dict(outcome.final_state),
        "final_step_key": final_step_key,
    }


def semantic_sentence_payload(sentence_report: dict[str, Any]) -> dict[str, Any]:
    """Fields that must be identical between incremental and full runs."""

    return {
        "sentence_id": sentence_report.get("sentence_id"),
        "status": sentence_report.get("status"),
        "action_sequence": sentence_report.get("action_sequence"),
        "action_names": sentence_report.get("action_names"),
        "steps": [
            {
                "step_index": step.get("step_index"),
                "action": step.get("action"),
                "score": step.get("score"),
                "total_score_after": step.get("total_score_after"),
                "feasible_actions": step.get("feasible_actions"),
                "excluded_actions": step.get("excluded_actions"),
                "stack_before": step.get("stack_before"),
                "buffer_before": step.get("buffer_before"),
                "stack_after": step.get("stack_after"),
                "buffer_after": step.get("buffer_after"),
            }
            for step in sentence_report.get("steps", [])
        ],
        "arcs": sentence_report.get("arcs"),
        "tree": sentence_report.get("tree"),
        "reference_metrics": sentence_report.get("reference_metrics"),
        "error_code": sentence_report.get("error_code"),
    }


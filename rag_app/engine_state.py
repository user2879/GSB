"""Mutable engine state and command-driven invalidation planning."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Command, Document, QueryTurn, RetrievalConfig
from .rules import RuleTable

#: Commands that change retrieval-stage inputs.
RETRIEVAL_LEVEL_OPS = {
    "set_bm25",
    "set_recall_top_k",
    "set_fusion_weights",
    "set_context",
    "replace_turn_query",
}
#: Commands that change answer/citation assembly inputs only.
CITATION_LEVEL_OPS = {"set_template", "update_template"}


@dataclass
class EngineState:
    docs: list[Document]
    turns: dict[tuple[str, int], QueryTurn]
    config: RetrievalConfig
    rules: RuleTable
    template_name: str

    def ordered_turns(self) -> list[QueryTurn]:
        return [self.turns[key] for key in sorted(self.turns)]

    def effective_query(self, turn: QueryTurn) -> str:
        if not turn.is_followup or not turn.context_ref:
            return turn.query
        parts: list[str] = []
        for ref in turn.context_ref:
            referenced = self.turns.get((turn.session_id, ref))
            if referenced is not None:
                parts.append(referenced.query)
        parts.append(turn.query)
        return " ".join(parts)


@dataclass
class CommandPlan:
    command: Command
    affected: list[tuple[str, int]] = field(default_factory=list)
    level: str = "retrieval"
    note: str = ""


def _downstream_closure(
    turns: dict[tuple[str, int], QueryTurn],
    seeds: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    """Seeds plus every follow-up turn that transitively references them."""
    affected = set(seeds)
    changed = True
    while changed:
        changed = False
        for key, turn in turns.items():
            if key in affected:
                continue
            if any(
                (turn.session_id, ref) in affected for ref in turn.context_ref
            ):
                affected.add(key)
                changed = True
    return sorted(affected)


def apply_command(
    state: EngineState,
    command: Command,
) -> CommandPlan:
    """Mutate engine state in place and return the invalidation plan."""
    args = command.args
    op = command.op
    plan = CommandPlan(command=command)
    if op == "set_bm25":
        bm25 = dict(state.config.bm25)
        bm25.update({k: float(v) for k, v in args.items()})
        state.config = state.config.replace(bm25=bm25)
        plan.level = "retrieval"
        plan.affected = _downstream_closure(
            state.turns, list(state.turns.keys())
        )
        plan.note = "bm25 params changed"
    elif op == "set_recall_top_k":
        state.config = state.config.replace(
            recall_top_k=int(args["recall_top_k"])
        )
        plan.level = "retrieval"
        plan.affected = _downstream_closure(
            state.turns, list(state.turns.keys())
        )
        plan.note = "recall top-k changed"
    elif op == "set_fusion_weights":
        weights = dict(state.config.fusion_weights)
        weights.update({k: float(v) for k, v in args.items()})
        state.config = state.config.replace(fusion_weights=weights)
        plan.level = "retrieval"
        plan.affected = _downstream_closure(
            state.turns, list(state.turns.keys())
        )
        plan.note = "fusion weights changed"
    elif op == "set_context":
        session_id = str(args["session_id"])
        turn_no = int(args["turn_no"])
        key = (session_id, turn_no)
        current = state.turns[key]
        refs = tuple(int(ref) for ref in args.get("context_ref", []))
        is_followup = bool(
            args.get("is_followup", True if refs else current.is_followup)
        )
        state.turns[key] = QueryTurn(
            session_id=current.session_id,
            turn_no=current.turn_no,
            query=current.query,
            is_followup=is_followup,
            context_ref=refs,
        )
        plan.level = "retrieval"
        plan.affected = _downstream_closure(state.turns, [key])
        plan.note = "session context changed"
    elif op == "replace_turn_query":
        session_id = str(args["session_id"])
        turn_no = int(args["turn_no"])
        key = (session_id, turn_no)
        current = state.turns[key]
        state.turns[key] = QueryTurn(
            session_id=current.session_id,
            turn_no=current.turn_no,
            query=str(args["query"]),
            is_followup=current.is_followup,
            context_ref=current.context_ref,
        )
        plan.level = "retrieval"
        plan.affected = _downstream_closure(state.turns, [key])
        plan.note = "single-turn query replaced"
    elif op == "set_template":
        state.template_name = str(args["template"])
        if state.template_name not in state.rules.templates:
            raise KeyError(f"unknown template: {state.template_name}")
        plan.level = "citation"
        plan.affected = _downstream_closure(
            state.turns, list(state.turns.keys())
        )
        plan.note = "rewrite template switched"
    elif op == "update_template":
        name = str(args["template"])
        state.rules = state.rules.with_template(name, str(args["body"]))
        plan.level = "citation"
        plan.affected = _downstream_closure(
            state.turns, list(state.turns.keys())
        )
        plan.note = "rewrite template body updated"
    else:
        raise KeyError(f"unknown command op: {op}")
    return plan


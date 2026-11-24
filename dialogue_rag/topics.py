"""Deterministic query similarity and topic segmentation."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

from .fixedpoint import as_fraction, fraction_float
from .hashing import cache_key, fingerprint
from .model import RuntimeState, Turn


_LATIN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def query_tokens(query: str) -> Counter[str]:
    text = query.lower()
    tokens: Counter[str] = Counter(_LATIN_RE.findall(text))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            tokens[f"cjk:{run}"] += 1
        for index in range(len(run) - 1):
            tokens[f"han:{run[index:index + 2]}"] += 1
        if len(run) >= 2:
            tokens[f"word:{run}"] += 1
    return tokens


def jaccard_similarity(left: str, right: str) -> Fraction:
    left_tokens = set(query_tokens(left))
    right_tokens = set(query_tokens(right))
    if not left_tokens and not right_tokens:
        return Fraction(1)
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return Fraction(intersection, union)


def stable_topic_id(first_turn_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", first_turn_id).strip("_")
    return f"TP_{safe}"


def build_segmentation_cache_key(state: RuntimeState, session_id: str) -> str:
    turns = state.sorted_turns(session_id)
    return cache_key(
        "topic-segmentation",
        {
            "session_id": session_id,
            "turn_fingerprints": [turn.content_fingerprint for turn in turns],
            "threshold": state.topic_threshold,
            "equal_rule": state.equal_rule,
        },
    )


def segment_session(state: RuntimeState, session_id: str) -> dict:
    turns = state.sorted_turns(session_id)
    threshold = as_fraction(state.topic_threshold)
    adjacencies: list[dict] = []
    turn_topic: dict[str, str] = {}
    topics: list[dict] = []
    current_turns: list[Turn] = []
    current_topic_id: str | None = None

    def flush() -> None:
        nonlocal current_turns, current_topic_id
        if not current_turns or current_topic_id is None:
            return
        topics.append(
            {
                "topic_id": current_topic_id,
                "session_id": session_id,
                "first_turn_id": current_turns[0].turn_id,
                "last_turn_id": current_turns[-1].turn_id,
                "turn_ids": [turn.turn_id for turn in current_turns],
                "seq_start": current_turns[0].seq,
                "seq_end": current_turns[-1].seq,
                "turn_count": len(current_turns),
                "member_fingerprint": fingerprint([turn.content_fingerprint for turn in current_turns]),
            }
        )
        current_turns = []
        current_topic_id = None

    for index, turn in enumerate(turns):
        starts_topic = index == 0
        boundary_record = None
        if index > 0:
            previous = turns[index - 1]
            similarity = jaccard_similarity(previous.query, turn.query)
            if similarity < threshold:
                relation = "below"
                switch = True
            elif similarity == threshold:
                relation = "equal"
                switch = state.equal_rule == "split"
            else:
                relation = "above"
                switch = False
            decision = "split" if switch else "merge"
            boundary_record = {
                "session_id": session_id,
                "after_turn_id": previous.turn_id,
                "turn_id": turn.turn_id,
                "previous_seq": previous.seq,
                "seq": turn.seq,
                "similarity": fraction_float(similarity, 12),
                "similarity_real": similarity,
                "threshold": fraction_float(threshold, 12),
                "threshold_real": threshold,
                "relation": relation,
                "equal_tie_rule": state.equal_rule,
                "decision": decision,
                "is_boundary": switch,
            }
            adjacencies.append(boundary_record)
            starts_topic = switch

        if starts_topic:
            flush()
            current_topic_id = stable_topic_id(turn.turn_id)
        if current_topic_id is None:
            current_topic_id = stable_topic_id(turn.turn_id)
        turn_topic[turn.turn_id] = current_topic_id
        current_turns.append(turn)
        if boundary_record is not None and boundary_record["relation"] == "equal":
            boundary_record = dict(boundary_record)
            boundary_record["boundary_topic_id"] = current_topic_id
            adjacencies[-1] = boundary_record

    flush()

    boundaries = [
        {key: value for key, value in record.items() if not key.endswith("_real")}
        for record in adjacencies
        if record["is_boundary"]
    ]
    equal_boundaries = [
        {
            "after_turn_id": record["after_turn_id"],
            "turn_id": record["turn_id"],
            "similarity": record["similarity"],
            "threshold": record["threshold"],
            "relation": "equal",
            "decision": record["decision"],
            "boundary_topic_id": record.get("boundary_topic_id"),
        }
        for record in adjacencies
        if record["relation"] == "equal"
    ]
    turn_assignments = [
        {
            "session_id": session_id,
            "turn_id": turn.turn_id,
            "seq": turn.seq,
            "topic_id": turn_topic[turn.turn_id],
            "is_topic_start": any(topic["first_turn_id"] == turn.turn_id for topic in topics),
        }
        for turn in turns
    ]
    key = build_segmentation_cache_key(state, session_id)
    return {
        "stage": "topic_segmentation",
        "session_id": session_id,
        "cache_key": key,
        "topics": topics,
        "turn_topics": turn_assignments,
        "adjacencies": [
            {key: value for key, value in record.items() if not key.endswith("_real")}
            for record in adjacencies
        ],
        "boundaries": boundaries,
        "equal_boundaries": equal_boundaries,
        "turn_topic_map": turn_topic,
    }

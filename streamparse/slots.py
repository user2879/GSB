from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .models import DecodeConfig, TokenDecision, Vocabulary


def fill_slots(
    vocab: Vocabulary,
    decisions: Sequence[TokenDecision],
    config: DecodeConfig,
) -> Dict[str, Any]:
    starts: List[int] = []
    for index, decision in enumerate(decisions):
        entry = vocab.entry(decision.token_id)
        if entry.slot_start:
            starts.append(index)

    filled: List[Dict[str, Any]] = []
    unfilled: List[Dict[str, Any]] = []
    for order, start in enumerate(starts):
        end = (starts[order + 1] - 1) if order + 1 < len(starts) else len(decisions) - 1
        token_ids = [item.token_id for item in decisions[start : end + 1]]
        entry = vocab.entry(decisions[start].token_id)
        frame_start = decisions[start].global_frame
        frame_end = decisions[end].global_frame
        length = end - start + 1
        item = {
            "slot": entry.slot or f"slot_{entry.pos.lower()}",
            "pos": entry.pos,
            "start_frame": frame_start,
            "end_frame": frame_end,
            "length_frames": length,
            "token_ids": token_ids,
            "text": vocab.render(token_ids),
            "subword_text": vocab.subword_text(token_ids),
        }
        if config.slot_min_frames <= length <= config.slot_max_frames:
            filled.append(item)
        else:
            item["reason"] = "outside_slot_frame_bounds"
            unfilled.append(item)
    return {"filled": filled, "unfilled": unfilled}

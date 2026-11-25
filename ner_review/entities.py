"""BIESO repair, span merging and deterministic type attribution."""

from __future__ import annotations

import math
from typing import Any

from .codes import Code
from .labels import LABEL_CODES, TYPE_CODES, parse_label


def repair_bieso(raw_labels: list[str]) -> dict[str, Any]:
    repaired = list(raw_labels)
    repairs: list[dict[str, Any]] = []
    open_start: int | None = None
    open_type: str | None = None

    def add(position: int, rule_code: str, before: str, after: str, basis: str, affected: list[int] | None = None) -> None:
        if before == after:
            return
        repairs.append(
            {
                "position": position,
                "rule_code": rule_code,
                "before": before,
                "after": after,
                "basis": basis,
                "affected_positions": sorted(set(affected or [position])),
            }
        )

    def close_span(end: int, trigger_position: int, trigger_label: str | None, jump: bool) -> None:
        nonlocal open_start, open_type
        if open_start is None or open_type is None:
            return
        before = repaired[end]
        if open_start == end:
            after = f"S-{open_type}"
            rule_code = Code.REPAIR_ORPHAN_B
            basis = f"B-{open_type} 后未形成 I/E 连续结构，固定修复为 S-{open_type}"
        else:
            after = f"E-{open_type}"
            rule_code = Code.REPAIR_LABEL_JUMP if jump else Code.REPAIR_UNCLOSED_SPAN
            basis = "标签跳跃打断当前片段，已将最后一个 I/B 位置闭合为 E" if jump else "片段未遇到 E 即被 O/文末终止，按连续性规则补 E"
        add(end, rule_code, before, after, basis, [open_start, end, trigger_position])
        repaired[end] = after
        open_start = None
        open_type = None

    for index, label in enumerate(raw_labels):
        parsed = parse_label(label)
        if parsed is None:
            if open_start is not None:
                close_span(index - 1, index, label, True)
            repaired[index] = "O"
            add(index, Code.UNKNOWN_TAG if hasattr(Code, "UNKNOWN_TAG") else "NER-E1004", label, "O", "解码标签不在枚举内，隔离为 O")
            continue
        prefix, entity_type = parsed

        if prefix == "O":
            if open_start is not None:
                close_span(index - 1, index, label, False)
            repaired[index] = "O"
            continue

        if prefix == "S":
            if open_start is not None:
                close_span(index - 1, index, label, True)
            repaired[index] = label
            continue

        if prefix == "B":
            if open_start is not None:
                close_span(index - 1, index, label, True)
            open_start = index
            open_type = entity_type
            repaired[index] = label
            continue

        if prefix == "I":
            if open_start is None or open_type is None:
                after = f"B-{entity_type}"
                add(index, Code.REPAIR_I_WITHOUT_B, label, after, "I 前无未闭合 B，固定将 I 提升为 B")
                repaired[index] = after
                open_start = index
                open_type = entity_type
                continue
            if entity_type != open_type:
                close_span(index - 1, index, label, True)
                after = f"B-{entity_type}"
                add(index, Code.REPAIR_LABEL_JUMP, label, after, f"I 类型与原 B-{open_type} 不一致，闭合旧片段并开启新片段", [index - 1, index])
                repaired[index] = after
                open_start = index
                open_type = entity_type
            else:
                repaired[index] = label
            continue

        if prefix == "E":
            if open_start is None or open_type is None:
                after = f"S-{entity_type}"
                add(index, Code.REPAIR_E_WITHOUT_B, label, after, "E 前无 B/I，单字片段固定修复为 S")
                repaired[index] = after
                continue
            if entity_type != open_type:
                close_span(index - 1, index, label, True)
                after = f"S-{entity_type}"
                add(index, Code.REPAIR_LABEL_JUMP, label, after, f"E 类型与原 B-{open_type} 不一致，旧片段闭合，当前 E 降为 S", [index - 1, index])
                repaired[index] = after
                open_start = None
                open_type = None
            else:
                repaired[index] = label
                open_start = None
                open_type = None

    if open_start is not None:
        close_span(len(raw_labels) - 1, len(raw_labels), None, False)

    return {"repaired_labels": repaired, "repairs": repairs}


def softmax_rows(candidate_scores: list[list[float]]) -> list[list[float]]:
    probabilities: list[list[float]] = []
    for row in candidate_scores:
        best = row[0]
        for value in row[1:]:
            if value > best:
                best = value
        exponents = [math.exp(value - best) for value in row]
        total = 0.0
        for value in exponents:
            total += value
        probabilities.append([value / total for value in exponents])
    return probabilities


def _extract_spans(labels: list[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    entity_type: str | None = None
    for index, label in enumerate(labels):
        parsed = parse_label(label)
        if parsed is None:
            continue
        prefix, current_type = parsed
        if prefix == "S":
            spans.append((index, index, current_type))
        elif prefix == "B":
            start = index
            entity_type = current_type
        elif prefix == "E" and start is not None and current_type == entity_type:
            spans.append((start, index, current_type))
            start = None
            entity_type = None
    if start is not None and entity_type is not None:
        spans.append((start, len(labels) - 1, entity_type))
    return sorted(spans)


def merge_entities(
    sample: dict[str, Any],
    viterbi_result: dict[str, Any],
    lexicon: list[dict[str, str]],
) -> dict[str, Any]:
    raw_labels = viterbi_result["labels"]
    repaired = repair_bieso(raw_labels)
    repaired_labels = repaired["repaired_labels"]
    repairs = repaired["repairs"]
    probabilities = softmax_rows(viterbi_result["candidate_scores"])
    spans = _extract_spans(repaired_labels)
    entities: list[dict[str, Any]] = []
    suppressions: list[dict[str, Any]] = []
    distribution = {entity_type: 0 for entity_type in TYPE_CODES}

    for entity_index, (start, end, structural_type) in enumerate(spans, start=1):
        text = "".join(sample["chars"][start : end + 1])
        candidates: dict[str, dict[str, Any]] = {}

        def add_candidate(entity_type: str, confidence: float, source: str) -> None:
            confidence = round(max(0.0, min(1.0, float(confidence))), 9)
            current = candidates.get(entity_type)
            if current is None or confidence > current["confidence"]:
                candidates[entity_type] = {"type": entity_type, "confidence": confidence, "sources": [source]}
            elif confidence == current["confidence"]:
                if source not in current["sources"]:
                    current["sources"].append(source)

        for position in range(start, end + 1):
            raw_label = raw_labels[position]
            parsed = parse_label(raw_label)
            if parsed and parsed[0] != "O":
                entity_type = parsed[1]
                add_candidate(entity_type, probabilities[position][LABEL_CODES[raw_label]], f"tag@{position}")
            repaired_label = repaired_labels[position]
            repaired_parsed = parse_label(repaired_label)
            if repaired_parsed and repaired_parsed[0] != "O":
                add_candidate(
                    repaired_parsed[1],
                    probabilities[position][LABEL_CODES[repaired_label]],
                    f"repaired_tag@{position}",
                )

        for entry in lexicon:
            if entry["text"] == text:
                add_candidate(entry["type"], float(entry.get("confidence", 0.95)), f"lexicon:{entry['id']}")

        if structural_type not in candidates:
            add_candidate(structural_type, 0.0, "structural_bieso")

        candidate_list = list(candidates.values())
        for candidate in candidate_list:
            candidate["type_code"] = TYPE_CODES[candidate["type"]]
            candidate["sources"] = sorted(candidate["sources"])
        candidate_list.sort(key=lambda item: (item["type_code"], -item["confidence"], item["type"]))
        selected = min(candidate_list, key=lambda item: (item["type_code"], -item["confidence"], item["type"]))
        selected_type = selected["type"]

        entity_id = f"{sample['id']}:E{entity_index:03d}"
        entity = {
            "entity_id": entity_id,
            "sample_id": sample["id"],
            "ordinal": entity_index,
            "start": start,
            "end": end,
            "text": text,
            "type": selected_type,
            "type_code": TYPE_CODES[selected_type],
            "confidence": selected["confidence"],
            "candidate_types": candidate_list,
            "decision_rule": "type_code_ascending_then_confidence_descending",
        }
        entities.append(entity)
        distribution[selected_type] += 1

        for candidate in candidate_list:
            if candidate["type"] == selected_type:
                continue
            suppression = {
                "suppression_id": f"{entity_id}:S{candidate['type_code']:02d}",
                "entity_id": entity_id,
                "sample_id": sample["id"],
                "start": start,
                "end": end,
                "text": text,
                "candidate_type": candidate["type"],
                "candidate_type_code": candidate["type_code"],
                "candidate_confidence": candidate["confidence"],
                "selected_type": selected_type,
                "reason": "type_code_ascending_then_confidence_descending",
            }
            suppressions.append(suppression)

    repairs_out = []
    for repair in repairs:
        item = dict(repair)
        item["sample_id"] = sample["id"]
        repairs_out.append(item)

    return {
        "sample_id": sample["id"],
        "raw_labels": raw_labels,
        "repaired_labels": repaired_labels,
        "repairs": repairs_out,
        "entities": entities,
        "suppressions": suppressions,
        "type_distribution": distribution,
    }

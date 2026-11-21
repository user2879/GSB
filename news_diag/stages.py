"""Pure, deterministic implementations of the five pipeline stages."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .config import WorkbenchConfig
from .errors import ConfigInvalid, DocumentInvalid


SPLIT_BOUNDARIES = frozenset("。！？!?；;…")
TRAILING_QUOTES = frozenset("”’」』）】〕》")


def document_source_text(document: dict[str, Any]) -> tuple[str, list[int]]:
    starts: list[int] = []
    cursor = 0
    for paragraph in document["paragraphs"]:
        starts.append(cursor)
        cursor += len(paragraph) + 1
    return "\n".join(document["paragraphs"]), starts


def split_stage(document: dict[str, Any], config: WorkbenchConfig) -> dict[str, Any]:
    paragraphs = document.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise DocumentInvalid("切分阶段输入段落为空", stage="split",
                              document_id=document.get("document_id"))
    if any(not isinstance(paragraph, str) or not paragraph.strip() for paragraph in paragraphs):
        raise DocumentInvalid("切分阶段输入包含空段落", stage="split",
                              document_id=document.get("document_id"))

    source_text, paragraph_starts = document_source_text(document)
    sentences: list[dict[str, Any]] = []
    sentence_index = 0
    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph_start = paragraph_starts[paragraph_index]
        sentence_start: int | None = None
        cursor = 0
        while cursor < len(paragraph):
            char = paragraph[cursor]
            if sentence_start is None and not char.isspace():
                sentence_start = cursor
            if char in SPLIT_BOUNDARIES:
                end = cursor + 1
                while end < len(paragraph) and (paragraph[end] in SPLIT_BOUNDARIES
                                                or paragraph[end] in TRAILING_QUOTES):
                    end += 1
                if sentence_start is not None:
                    sentence = _make_sentence(
                        document["document_id"], paragraph_index,
                        paragraph_start + sentence_start,
                        paragraph_start + end,
                        source_text[(paragraph_start + sentence_start):(paragraph_start + end)],
                        sentence_index,
                    )
                    if sentence is not None:
                        sentences.append(sentence)
                        sentence_index += 1
                sentence_start = None
                cursor = end
                continue
            cursor += 1
        if sentence_start is not None:
            end = len(paragraph)
            while end > sentence_start and paragraph[end - 1].isspace():
                end -= 1
            sentence = _make_sentence(
                document["document_id"], paragraph_index,
                paragraph_start + sentence_start,
                paragraph_start + end,
                source_text[(paragraph_start + sentence_start):(paragraph_start + end)],
                sentence_index,
            )
            if sentence is not None:
                sentences.append(sentence)
                sentence_index += 1

    if not sentences:
        raise DocumentInvalid("切分阶段未得到任何句子", stage="split",
                              document_id=document.get("document_id"))
    return {
        "document_id": document["document_id"],
        "source_text": source_text,
        "paragraph_offsets": paragraph_starts,
        "sentences": sentences,
    }


def _make_sentence(document_id: str, paragraph_index: int, start: int, end: int,
                   text: str, sentence_index: int) -> dict[str, Any] | None:
    left_trim = len(text) - len(text.lstrip())
    right_trimmed = text.rstrip()
    adjusted_start = start + left_trim
    adjusted_end = start + len(right_trimmed)
    adjusted_text = right_trimmed[left_trim:]
    if not adjusted_text:
        return None
    return {
        "sentence_id": f"s{sentence_index:04d}",
        "paragraph_index": paragraph_index,
        "start": adjusted_start,
        "end": adjusted_end,
        "text": adjusted_text,
    }


def tokenize(text: str, vocabulary: tuple[str, ...]) -> list[str]:
    ordered_vocabulary = sorted({item for item in vocabulary if item}, key=lambda x: (-len(x), x))
    tokens: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char.isascii() and char.isalnum():
            end = index + 1
            while end < len(text) and text[end].isascii() and text[end].isalnum():
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        matched = None
        for word in ordered_vocabulary:
            if text.startswith(word, index):
                matched = word
                break
        if matched is not None:
            tokens.append(matched)
            index += len(matched)
        elif "\u4e00" <= char <= "\u9fff":
            tokens.append(char)
            index += 1
        else:
            index += 1
    return tokens


def scoring_stage(document: dict[str, Any], split_payload: dict[str, Any],
                  config: WorkbenchConfig) -> dict[str, Any]:
    if config.fixed_decimal_places < 0 or config.fixed_decimal_places > 9:
        raise ConfigInvalid("定点小数位宽越界", stage="scoring",
                            document_id=document.get("document_id"),
                            details={"fixed_decimal_places": config.fixed_decimal_places})
    entities = document.get("entities")
    if not isinstance(entities, list):
        raise DocumentInvalid("打分阶段实体列表非法", stage="scoring",
                              document_id=document.get("document_id"))
    seen_entity_ids: set[str] = set()
    normalized_entities: list[dict[str, str]] = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise DocumentInvalid("打分阶段实体项必须是对象", stage="scoring",
                                  document_id=document.get("document_id"),
                                  details={"entity_index": index})
        entity_id = entity.get("id")
        entity_text = entity.get("text")
        if not isinstance(entity_id, str) or not entity_id.strip() or not isinstance(entity_text, str) or not entity_text.strip():
            raise DocumentInvalid("打分阶段实体缺少有效 id/text", stage="scoring",
                                  document_id=document.get("document_id"),
                                  details={"entity_index": index})
        if entity_id in seen_entity_ids:
            raise DocumentInvalid("打分阶段实体 ID 重复", stage="scoring",
                                  document_id=document.get("document_id"),
                                  details={"entity_index": index})
        seen_entity_ids.add(entity_id)
        normalized_entities.append({"id": entity_id, "text": entity_text})

    sentence_texts = [sentence["text"] for sentence in split_payload["sentences"]]
    doc_tokens = tokenize(split_payload["source_text"], config.vocabulary)
    token_counts = Counter(doc_tokens)
    total_tokens = len(doc_tokens)
    total_entities = len(normalized_entities)
    precision = config.fixed_decimal_places
    scale = 10 ** precision
    quantum = Decimal(1).scaleb(-precision)

    scored: list[dict[str, Any]] = []
    for position, sentence in enumerate(split_payload["sentences"]):
        sentence_tokens = tokenize(sentence["text"], config.vocabulary)
        if total_tokens:
            word_frequency_raw = Decimal(sum(token_counts[token] for token in sentence_tokens)) / Decimal(total_tokens)
        else:
            word_frequency_raw = Decimal(0)
        matched_entities = [entity["id"] for entity in normalized_entities
                            if entity["text"] in sentence["text"]]
        if total_entities:
            entity_coverage_raw = Decimal(len(matched_entities)) / Decimal(total_entities)
        else:
            entity_coverage_raw = Decimal(0)
        entity_coverage_scaled = _to_scaled(entity_coverage_raw, scale, quantum)
        position_decay_raw = config.position_decay_base ** position
        weighted = (
            config.word_frequency_weight * word_frequency_raw
            + config.entity_coverage_weight * entity_coverage_raw
            + config.position_decay_weight * position_decay_raw
        )
        score_scaled = _to_scaled(weighted, scale, quantum)
        score_sources = ["word_frequency", "position_decay"]
        if entity_coverage_raw > 0:
            score_sources.append("entity_coverage")
        scored.append({
            **sentence,
            "score": score_scaled,
            "score_precision": precision,
            "score_components": {
                "word_frequency": _to_scaled(config.word_frequency_weight * word_frequency_raw, scale, quantum),
                "entity_coverage": _to_scaled(config.entity_coverage_weight * entity_coverage_raw, scale, quantum),
                "position_decay": _to_scaled(config.position_decay_weight * position_decay_raw, scale, quantum),
            },
            "entity_coverage_scaled": entity_coverage_scaled,
            "raw_components": {
                "word_frequency": str(word_frequency_raw),
                "entity_coverage": str(entity_coverage_raw),
                "position_decay": str(position_decay_raw),
            },
            "entity_coverage_zero": entity_coverage_raw == 0,
            "score_sources": score_sources,
            "entity_ids": matched_entities,
            "token_count": len(sentence_tokens),
        })

    return {
        "document_id": document["document_id"],
        "precision": precision,
        "scale": scale,
        "document_token_count": total_tokens,
        "document_entity_count": total_entities,
        "sentences": scored,
    }


def _to_scaled(value: Decimal, scale: int, quantum: Decimal) -> int:
    return int((value.quantize(quantum, rounding=ROUND_HALF_UP) * scale).to_integral_value())


def selection_stage(document: dict[str, Any], scoring_payload: dict[str, Any],
                    config: WorkbenchConfig) -> dict[str, Any]:
    if config.max_summary_sentences <= 0:
        raise ConfigInvalid("最大摘要句数必须为正整数", stage="selection",
                            document_id=document.get("document_id"),
                            details={"max_summary_sentences": config.max_summary_sentences})
    scale = scoring_payload["scale"]
    minimum = int((config.entity_coverage_min * scale).to_integral_value(rounding=ROUND_HALF_UP))
    eligible = [sentence for sentence in scoring_payload["sentences"]
                if sentence["entity_coverage_scaled"] >= minimum]
    eligible.sort(key=lambda item: (-item["score"], item["start"], item["end"], item["sentence_id"]))
    selected = []
    for rank, sentence in enumerate(eligible[:config.max_summary_sentences], start=1):
        selected.append({
            **sentence,
            "rank": rank,
            "eligible": True,
        })
    return {
        "document_id": document["document_id"],
        "max_summary_sentences": config.max_summary_sentences,
        "entity_coverage_min_scaled": minimum,
        "eligible_sentence_count": len(eligible),
        "selected_sentences": selected,
    }


def validate_rewrite_inputs(config: WorkbenchConfig, document_id: str | None = None) -> None:
    version = config.synonym_table_version
    table = config.raw.get("synonym_tables", {}).get(version)
    if not isinstance(table, dict):
        raise ConfigInvalid(f"同义替换表版本不存在: {version}", stage="rewrite",
                            document_id=document_id,
                            details={"synonym_table_version": version})
    templates = table.get("templates")
    if not isinstance(templates, dict) or config.style_template_id not in templates:
        raise ConfigInvalid(f"风格模板未声明: {config.style_template_id}", stage="rewrite",
                            document_id=document_id,
                            details={"style_template_id": config.style_template_id,
                                     "synonym_table_version": version})
    entries = table.get("entries")
    if not isinstance(entries, list):
        raise ConfigInvalid("同义替换表 entries 非法", stage="rewrite", document_id=document_id)
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("template_id") not in templates:
            raise ConfigInvalid(f"同义替换表条目 {index} 引用不存在的模板", stage="rewrite",
                                document_id=document_id,
                                details={"entry_index": index,
                                         "style_template_id": entry.get("template_id") if isinstance(entry, dict) else None})
        source = entry.get("source")
        candidates = entry.get("candidates")
        if not isinstance(source, str) or not source:
            raise ConfigInvalid(f"同义替换表条目 {index}.source 非法", stage="rewrite",
                                document_id=document_id, details={"entry_index": index})
        if not isinstance(candidates, list) or not candidates or any(
            not isinstance(candidate, str) or not candidate for candidate in candidates
        ):
            raise ConfigInvalid(f"同义替换表条目 {index}.candidates 非法", stage="rewrite",
                                document_id=document_id, details={"entry_index": index})
    template = templates[config.style_template_id]
    limit = _rewrite_limit(config, template)
    if limit <= 0:
        raise ConfigInvalid("重写长度上限必须为正整数", stage="rewrite",
                            document_id=document_id, details={"max_rewrite_chars": limit})


def _rewrite_limit(config: WorkbenchConfig, template: dict[str, Any]) -> int:
    template_limit = template.get("max_chars", config.max_rewrite_chars)
    return min(config.max_rewrite_chars, template_limit)


def rewrite_stage(document: dict[str, Any], selection_payload: dict[str, Any],
                  config: WorkbenchConfig) -> dict[str, Any]:
    validate_rewrite_inputs(config, document.get("document_id"))
    template = config.template
    table = config.synonym_table
    entries = table["entries"]
    limit = _rewrite_limit(config, template)
    prefix = template.get("prefix", "")
    suffix = template.get("suffix", "")
    rewritten_sentences: list[dict[str, Any]] = []
    replacement_records: list[dict[str, Any]] = []
    truncations: list[dict[str, Any]] = []

    for selected in selection_payload["selected_sentences"]:
        source_text = selected["text"]
        output_chars: list[str] = []
        sentence_replacements: list[dict[str, Any]] = []
        cursor = 0
        while cursor < len(source_text):
            matched_entry_index: int | None = None
            matched_entry: dict[str, Any] | None = None
            for entry_index, entry in enumerate(entries):
                if entry["template_id"] != config.style_template_id:
                    continue
                if source_text.startswith(entry["source"], cursor):
                    matched_entry_index = entry_index
                    matched_entry = entry
                    break
            if matched_entry is None or matched_entry_index is None:
                output_chars.append(source_text[cursor])
                cursor += 1
                continue
            source = matched_entry["source"]
            replacement = matched_entry["candidates"][0]
            output_start = len(output_chars)
            output_chars.extend(replacement)
            record = {
                "source": source,
                "replacement": replacement,
                "candidate_index": 0,
                "entry_index": matched_entry_index,
                "source_start": cursor,
                "source_end": cursor + len(source),
                "output_start": output_start,
                "output_end": output_start + len(replacement),
                "sentence_id": selected["sentence_id"],
            }
            sentence_replacements.append(record)
            replacement_records.append(record)
            cursor += len(source)

        rewritten_core = "".join(output_chars)
        full_text = prefix + rewritten_core + suffix
        truncated = len(full_text) > limit
        final_text = full_text[:limit] if truncated else full_text
        if truncated:
            truncations.append({
                "sentence_id": selected["sentence_id"],
                "truncation_offset": limit,
                "original_length": len(full_text),
                "final_length": len(final_text),
            })
        for record in sentence_replacements:
            record["retained"] = record["output_start"] + len(prefix) < limit
        rewritten_sentences.append({
            "rank": selected["rank"],
            "sentence_id": selected["sentence_id"],
            "source_start": selected["start"],
            "source_end": selected["end"],
            "source_text": source_text,
            "text": final_text,
            "score": selected["score"],
            "score_precision": selected["score_precision"],
            "entity_ids": selected["entity_ids"],
            "source_sentence_ids": [selected["sentence_id"]],
            "style_template_id": config.style_template_id,
            "synonym_table_version": config.synonym_table_version,
            "replacements": sentence_replacements,
            "truncated": truncated,
            "truncation_offset": limit if truncated else None,
        })

    summary_text = "\n".join(sentence["text"] for sentence in rewritten_sentences)
    return {
        "document_id": document["document_id"],
        "style_template_id": config.style_template_id,
        "synonym_table_version": config.synonym_table_version,
        "max_rewrite_chars": limit,
        "rewritten_sentences": rewritten_sentences,
        "summary_text": summary_text,
        "replacement_records": replacement_records,
        "truncations": truncations,
        "source_sentence_ids": sorted({sid for sentence in rewritten_sentences
                                       for sid in sentence["source_sentence_ids"]}),
        "entity_ids": sorted({eid for sentence in rewritten_sentences
                              for eid in sentence["entity_ids"]}),
    }


def summarize_stage(rewrite_payloads: list[dict[str, Any]],
                    failures: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rewrite_payloads, key=lambda item: item["document_id"])
    summaries = []
    for payload in ordered:
        summaries.append({
            "document_id": payload["document_id"],
            "summary_text": payload["summary_text"],
            "sentences": payload["rewritten_sentences"],
            "replacement_records": payload["replacement_records"],
            "truncations": payload["truncations"],
            "source_sentence_ids": payload["source_sentence_ids"],
            "entity_ids": payload["entity_ids"],
            "style_template_id": payload["style_template_id"],
            "synonym_table_version": payload["synonym_table_version"],
        })
    sentence_count = sum(len(item["sentences"]) for item in summaries)
    replacement_count = sum(len(item["replacement_records"]) for item in summaries)
    truncation_count = sum(len(item["truncations"]) for item in summaries)
    document_failure_stages = {"load", "split", "scoring", "selection", "rewrite", "summary"}
    failed_document_ids = sorted({failure.get("document_id", "") for failure in failures
                                  if failure.get("document_id")
                                  and failure.get("stage") in document_failure_stages})
    return {
        "documents": summaries,
        "stats": {
            "document_count": len(summaries),
            "failed_document_count": len(failed_document_ids),
            "summary_sentence_count": sentence_count,
            "replacement_count": replacement_count,
            "truncation_count": truncation_count,
            "summary_character_count": sum(len(item["summary_text"]) for item in summaries),
        },
    }

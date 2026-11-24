"""Fixed-point sentence importance scoring with a float64 reference path."""

import numpy as np

from .hashing import combine
from .segmentation import tokenize


def _term_frequency_norm(tokens):
    if not tokens:
        return np.float64(0.0)
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    max_count = max(counts.values())
    return np.float64(max_count) / np.float64(len(tokens))


def _entity_coverage(sentence_text, entity_texts):
    if not entity_texts:
        return np.float64(0.0)
    hits = 0
    for entity in entity_texts:
        if entity in sentence_text:
            hits += 1
    return np.float64(hits) / np.float64(len(entity_texts))


def score_sentence(sentence, local_index, local_total, paragraph_count, paragraph_index,
                   entity_texts, vocabulary, config, q_format):
    """Return one deterministic score record.

    The float reference keeps a single fixed evaluation order:
    term component + entity component, then position decay applied last.
    Every intermediate value is float64.
    """
    tokens = tokenize(sentence["text"], vocabulary)
    tf_norm = _term_frequency_norm(tokens)
    entity_cov = _entity_coverage(sentence["text"], entity_texts)
    if local_total <= 1:
        position_factor = np.float64(1.0)
    else:
        position_factor = np.float64(1.0) - (
            np.float64(config.position_decay)
            * np.float64(local_index)
            / np.float64(local_total - 1)
        )

    term_component = np.float64(config.term_weight) * tf_norm
    entity_component = np.float64(config.entity_weight) * entity_cov
    reference_score = (term_component + entity_component) * position_factor
    fixed_score = q_format.quantize(reference_score)

    record = {
        "sentence_id": sentence["sentence_id"],
        "paragraph_pid": sentence["paragraph_pid"],
        "global_index": sentence["sentence_id"],
        "local_index": local_index,
        "start_offset": sentence["start_offset"],
        "text": sentence["text"],
        "tokens": list(tokens),
        "contributions": {
            "term_frequency_norm": float(tf_norm),
            "entity_coverage": float(entity_cov),
            "position_factor": float(position_factor),
            "term_component": float(term_component),
            "entity_component": float(entity_component),
        },
        "reference_score": float(reference_score),
        "fixed_score": fixed_score,
    }
    record["cache_key"] = combine(
        "score",
        sentence["text"],
        sentence["paragraph_pid"],
        local_index,
        local_total,
        list(entity_texts),
        list(vocabulary),
        config.fingerprint,
    )
    return record


def score_document(sentences, document, vocabulary, config, q_format):
    entity_texts = tuple(entity["text"] for entity in document["entities"])
    paragraph_count = len(document["paragraphs"])
    records = []
    grouped = {}
    for sentence in sentences:
        grouped.setdefault(sentence["paragraph_index"], []).append(sentence)
    local_lookup = {}
    for paragraph_index, group in grouped.items():
        for local_index, sentence in enumerate(group):
            local_lookup[sentence["sentence_id"]] = (local_index, len(group))
    for global_index, sentence in enumerate(sentences):
        local_index, local_total = local_lookup[sentence["sentence_id"]]
        record = score_sentence(
            sentence, local_index, local_total, paragraph_count,
            sentence["paragraph_index"], entity_texts, vocabulary, config, q_format,
        )
        record["global_index"] = global_index
        records.append(record)
    return records

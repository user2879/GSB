"""Coverage-constrained sentence selection and fixed/float boundary calls."""

from .hashing import combine


def _ranked_indices(score_records, key_name):
    """Sort by score desc, then start offset asc, then global index asc."""
    return sorted(
        range(len(score_records)),
        key=lambda idx: (
            -score_records[idx][key_name],
            score_records[idx]["start_offset"],
            score_records[idx]["global_index"],
        ),
    )


def _select(score_records, entity_texts, key_name, config):
    order = _ranked_indices(score_records, key_name)
    chosen = []
    covered = set()
    hit_cap = False
    for rank, index in enumerate(order):
        if len(chosen) >= config.max_sentences:
            hit_cap = True
            break
        chosen.append(index)
        sentence_text = score_records[index]["text"]
        for entity in entity_texts:
            if entity in sentence_text:
                covered.add(entity)
        coverage = len(covered) / len(entity_texts) if entity_texts else 1.0
        if coverage + 1e-15 >= config.coverage_floor:
            break
    if len(chosen) == config.max_sentences and order and len(chosen) < len(order):
        hit_cap = True
    return chosen, covered, hit_cap


def select_sentences(score_records, document, config):
    entity_texts = tuple(entity["text"] for entity in document["entities"])
    fixed_chosen, covered, hit_cap = _select(score_records, entity_texts, "fixed_score", config)
    float_chosen, _, _ = _select(score_records, entity_texts, "reference_score", config)

    fixed_set = set(fixed_chosen)
    float_set = set(float_chosen)
    boundary = []
    for index in sorted(fixed_set.symmetric_difference(float_set)):
        record = score_records[index]
        boundary.append(
            {
                "sentence_id": record["sentence_id"],
                "fixed_score": record["fixed_score"],
                "reference_score": record["reference_score"],
                "score_difference": abs(record["reference_score"] - record["fixed_score"] / (2 ** _frac_bits(config))),
                "in_fixed_selection": index in fixed_set,
                "in_float_selection": index in float_set,
                "decision": "fixed_includes_float_excludes"
                if index in fixed_set
                else "fixed_excludes_float_includes",
            }
        )

    actual_coverage = len(covered) / len(entity_texts) if entity_texts else 1.0
    selected_records = [score_records[index] for index in fixed_chosen]
    coverage_touch = actual_coverage + 1e-15 >= config.coverage_floor
    result = {
        "doc_id": document["doc_id"],
        "selected_order": fixed_chosen,
        "selected_sentence_ids": [record["sentence_id"] for record in selected_records],
        "actual_entity_coverage": actual_coverage,
        "covered_entities": sorted(covered),
        "boundary_sentences": boundary,
        "touch_flags": {
            "max_sentence_cap_touched": hit_cap,
            "coverage_floor_touched": coverage_touch,
        },
    }
    result["cache_key"] = combine(
        "select",
        [record["cache_key"] for record in score_records],
        sorted(entity_texts),
        config.fingerprint,
    )
    return result


def _frac_bits(config):
    return config.q_format.frac_bits

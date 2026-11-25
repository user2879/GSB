from __future__ import annotations

from fractions import Fraction

from search_proto.engine import SearchEngine
from search_proto.fixed import quantize_fraction
from search_proto.index import compute_document_stats
from search_proto.model import Document
from search_proto.ranking import rank_query
from search_proto.scoring import query_document_score_cache_key

from .conftest import make_config, make_docs, make_queries, make_stopwords, make_vocab


def test_quantize_fraction_rounds_half_away_from_zero() -> None:
    assert quantize_fraction(Fraction(1, 2), 1) == 1
    assert quantize_fraction(Fraction(3, 2), 1) == 2
    assert quantize_fraction(Fraction(-1, 2), 1) == -1
    assert quantize_fraction(Fraction(5, 2), 5) == 5


def test_fixed_width_and_quantum_change_real_units_and_intermediates() -> None:
    docs = make_docs()
    queries = make_queries()
    vocab = make_vocab()
    stopwords = make_stopwords()
    fine = SearchEngine(make_config(), stopwords, vocab, docs, queries)
    coarse = SearchEngine(
        make_config(fixed={"decimal_places": 2, "quantum": 10}),
        stopwords,
        vocab,
        docs,
        queries,
    )
    for engine in (fine, coarse):
        engine.materialize()
        engine.run_batches()

    fine_record = next(
        score for score in fine.score_records() if score["query_id"] == "Q0001" and score["doc_id"] == "D0001"
    )
    coarse_record = next(
        score for score in coarse.score_records() if score["query_id"] == "Q0001" and score["doc_id"] == "D0001"
    )
    assert (fine_record["fixed_scale"], fine_record["quantum_units"]) != (
        coarse_record["fixed_scale"],
        coarse_record["quantum_units"],
    )
    assert fine_record["fixed_score_units"] != coarse_record["fixed_score_units"]
    fine_intermediates = [
        field["saturation_fixed_units"]
        for term in fine_record["terms"]
        for field in term["fields"]
    ]
    coarse_intermediates = [
        field["saturation_fixed_units"]
        for term in coarse_record["terms"]
        for field in term["fields"]
    ]
    assert fine_intermediates != coarse_intermediates


def _score(doc_id: str, version: int, fixed: int, floating: float, tie_bias: int) -> dict:
    diagnostics = {
        "stage_max_abs_error_units": {
            "inverse_document_frequency": 0.2,
            "field_length_normalization": 0.1,
            "term_frequency_saturation": 0.6,
            "field_weight": 0.0,
            "accumulation": 1.0,
        },
        "first_tolerance_exceeded_stage": "term_frequency_saturation",
        "tolerance_units": 0.5,
        "variant_scores": {
            "float_reference": floating,
            "idf_quantized_only": floating,
            "length_quantized_only": floating,
            "saturation_quantized_only": fixed / 100,
            "field_weight_quantized_only": floating,
            "fixed": fixed / 100,
        },
        "fixed_minus_float_units": fixed - floating * 100,
    }
    return {
        "query_id": "Q",
        "doc_id": doc_id,
        "doc_version": version,
        "document_content_fingerprint": None,
        "fixed_score_units": fixed,
        "fixed_score": f"{fixed / 100:.2f}",
        "float_score": floating,
        "fixed_scale": 100,
        "quantum_units": 1,
        "terms": [],
        "diagnostics": diagnostics,
    }


def _score_with_variants(
    doc_id: str,
    fixed: int,
    floating: float,
    *,
    saturation: float,
    length: float,
) -> dict:
    score = _score(doc_id, 0, fixed, floating, 0)
    score["diagnostics"]["variant_scores"].update(
        {
            "saturation_quantized_only": saturation,
            "length_quantized_only": length,
        }
    )
    return score


def test_topk_tie_break_boundary_and_flip_attribution() -> None:
    config = make_config(fixed={"decimal_places": 2, "quantum": 1}, top_k=1)
    query = make_queries()[0]
    scores = [
        _score("D0002", 0, 100, 3.0, 1),
        _score("D0001", 1, 100, 2.0, 2),
    ]
    ranking = rank_query(query, scores, config)

    assert ranking["top_k"][0]["doc_id"] == "D0001"
    assert ranking["all_float_order"][0] == "D0002"
    assert ranking["order_flip"]["pair_flip_count"] == 1
    assert ranking["order_flip"]["evidence"][0]["cause"] == "tie_break_rule"
    assert len(ranking["boundary"]["documents"]) == 2
    assert all(
        document["attribution"] == "tie_break_rule"
        for document in ranking["boundary"]["documents"]
    )
    assert all(
        document["first_tolerance_exceeded_stage"]
        == "term_frequency_saturation"
        for document in ranking["boundary"]["documents"]
    )


def test_quantization_flip_attribution_distinguishes_saturation_and_length() -> None:
    config = make_config(fixed={"decimal_places": 2, "quantum": 1}, top_k=1)
    query = make_queries()[0]

    saturation_scores = [
        _score_with_variants(
            "D0001", 90, 2.0, saturation=1.0, length=2.0
        ),
        _score_with_variants(
            "D0002", 100, 1.0, saturation=2.0, length=1.0
        ),
    ]
    saturation_ranking = rank_query(query, saturation_scores, config)
    assert saturation_ranking["order_flip"]["evidence"][0]["cause"] == (
        "term_frequency_saturation_rounding"
    )

    length_scores = [
        _score_with_variants(
            "D0001", 90, 2.0, saturation=2.0, length=0.9
        ),
        _score_with_variants(
            "D0002", 100, 1.0, saturation=1.0, length=1.0
        ),
    ]
    length_ranking = rank_query(query, length_scores, config)
    assert length_ranking["order_flip"]["evidence"][0]["cause"] == (
        "field_length_normalization_quantization"
    )


def test_score_cache_key_ignores_unreferenced_document_content() -> None:
    vocab = make_vocab()
    stopwords = make_stopwords()
    config = make_config()
    engine = SearchEngine(config, stopwords, vocab, make_docs(), make_queries())
    engine.materialize()
    query = make_queries()[0]
    d1 = engine.document_stats["D0001"]
    d3 = engine.document_stats["D0003"]
    key1 = query_document_score_cache_key(query, d1, engine.term_stats, config)
    key3 = query_document_score_cache_key(query, d3, engine.term_stats, config)
    assert key1 != key3
    first = compute_document_stats(
        Document.create("D9", 0, {"title": ["gamma"], "body": ["zeta"]}),
        config,
        stopwords,
    )
    second = compute_document_stats(
        Document.create("D9", 0, {"title": ["delta"], "body": ["zeta"]}),
        config,
        stopwords,
    )
    assert first["content_fingerprint"] != second["content_fingerprint"]
    assert query_document_score_cache_key(
        query, first, engine.term_stats, config
    ) == query_document_score_cache_key(query, second, engine.term_stats, config)

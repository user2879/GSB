from depedit.codecs import (
    v1_parse_document,
    v1_serialize_document,
    v2_parse_document,
    v2_serialize_document,
    load_json,
    load_text,
)
from depedit.config import AppConfig
from depedit.relations import load_mapping


def _mapping(sample_dir):
    return load_mapping(sample_dir / "relation_map.json")


def test_v1_roundtrip_preserves_structure(sample_dir):
    cfg = AppConfig()
    mapping = _mapping(sample_dir)
    sentences = v1_parse_document(
        load_text(sample_dir / "parse_v1.txt"), cfg, mapping["v1_to_v2"]
    )
    assert len(sentences) == 8
    text, losses = v1_serialize_document(sentences, cfg, mapping["v2_to_v1"])
    again = v1_parse_document(text, cfg, mapping["v1_to_v2"])
    for a, b in zip(sentences, again):
        assert a.sent_id == b.sent_id
        assert len(a.arcs) == len(b.arcs)
        for arc_a, arc_b in zip(a.arcs, b.arcs):
            assert arc_a.normalized_head() == arc_b.normalized_head()
            assert arc_a.relation == arc_b.relation
            assert abs(arc_a.score - arc_b.score) < 1e-12


def test_v2_origin_conversion(sample_dir):
    cfg0 = AppConfig()
    cfg0.v2_head_origin = 0
    mapping = _mapping(sample_dir)
    sentences = v1_parse_document(
        load_text(sample_dir / "parse_v1.txt"), cfg0, mapping["v1_to_v2"]
    )
    doc, sidecar = v2_serialize_document(sentences, cfg0)
    parsed = v2_parse_document(doc, cfg0, mapping["named_relations"])
    for a, b in zip(sentences, parsed):
        for arc_a, arc_b in zip(a.arcs, b.arcs):
            assert arc_a.normalized_head() == arc_b.normalized_head()
    # 每个句子恰好一个空头部进入旁路元数据。
    nulls = {(x["sent_id"], x["word_idx"]) for x in sidecar["null_heads"]}
    assert len({sid for sid, _ in nulls}) == 8


def test_v2_one_based_origin(sample_dir):
    cfg = AppConfig()
    cfg.v2_head_origin = 1
    mapping = _mapping(sample_dir)
    sentences = v1_parse_document(
        load_text(sample_dir / "parse_v1.txt"), cfg, mapping["v1_to_v2"]
    )
    doc, _ = v2_serialize_document(sentences, cfg)
    parsed = v2_parse_document(doc, cfg, mapping["named_relations"])
    for a, b in zip(sentences, parsed):
        for arc_a, arc_b in zip(a.arcs, b.arcs):
            assert arc_a.normalized_head() == arc_b.normalized_head()


def test_score_precision_and_permille(sample_dir):
    cfg = AppConfig()
    mapping = _mapping(sample_dir)
    sentences = v1_parse_document(
        load_text(sample_dir / "parse_v1.txt"), cfg, mapping["v1_to_v2"]
    )
    sent = sentences[0]
    for arc in sent.arcs:
        assert 0.0 <= arc.score <= 1.0
        assert arc.precision.startswith("permille")

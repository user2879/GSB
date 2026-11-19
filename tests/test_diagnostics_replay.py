from depedit.codecs import v1_parse_document, load_text
from depedit.config import AppConfig
from depedit.diagnostics import roundtrip_compare
from depedit.models import Corpus
from depedit.relations import load_mapping
from depedit.replay import apply_edit_sequence, clone_corpus


def _corpus(sample_dir):
    cfg = AppConfig()
    mapping = load_mapping(sample_dir / "relation_map.json")
    sentences = v1_parse_document(
        load_text(sample_dir / "parse_v1.txt"), cfg, mapping["v1_to_v2"]
    )
    return Corpus(sentences={s.sent_id: s for s in sentences}), cfg, mapping


def test_roundtrip_histogram_and_score(sample_dir):
    corpus, cfg, mapping = _corpus(sample_dir)
    diag = roundtrip_compare(
        [corpus.sentences[sid] for sid in corpus.sorted_ids()], cfg, mapping
    )
    # 千分比精确往返，头词偏差全为 0。
    assert all(k == "0" for k in diag["head_deviation_histogram"])
    assert diag["max_score_deviation"] <= 0.0005
    assert diag["relation_restore_failures"] == []


def test_replay_determinism(sample_dir):
    corpus, _, _ = _corpus(sample_dir)
    sequence = [
        {"op": "set_head", "sent_id": "S001", "word_idx": 2, "new_head": 1},
        {"op": "mark_reparse", "sent_id": "S002"},
        {"op": "replace_relation", "sent_id": "S003",
         "old_relation": _first_rel(corpus, "S003"), "new_relation": "advmod"},
        {"op": "set_head", "sent_id": "S001", "word_idx": 1, "new_head": 999},
    ]
    baseline = clone_corpus(corpus)
    apply_edit_sequence(baseline, sequence)
    replayed = clone_corpus(corpus)
    summary = apply_edit_sequence(replayed, sequence)
    from depedit.fingerprints import content_hash

    assert content_hash(
        {sid: s.to_dict() for sid, s in baseline.sentences.items()}
    ) == content_hash({sid: s.to_dict() for sid, s in replayed.sentences.items()})
    assert len(summary["rejected"]) == 1
    assert summary["rejected"][0]["business_code"] == "E_HEAD_STRUCTURE"


def _first_rel(corpus, sid):
    return corpus.sentences[sid].arcs[0].relation

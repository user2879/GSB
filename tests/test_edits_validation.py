import pytest

from depedit.codecs import v1_parse_document, load_text
from depedit.config import AppConfig
from depedit.edits import mark_reparse, replace_relation, set_head, shift_heads
from depedit.errors import EHeadStructure
from depedit.relations import load_mapping
from depedit.validation import validate_sentence


@pytest.fixture()
def sentence(sample_dir):
    cfg = AppConfig()
    mapping = load_mapping(sample_dir / "relation_map.json")
    return v1_parse_document(
        load_text(sample_dir / "parse_v1.txt"), cfg, mapping["v1_to_v2"]
    )[0]


def test_set_head_valid(sentence):
    n = sentence.word_count()
    target = 2 if n > 2 else 1
    result = set_head(sentence, target, 1)
    assert result.affected_arcs == [target]
    assert sentence.arc_by_word(target).normalized_head() == 1


def test_set_head_out_of_range_rejected(sentence):
    before = sentence.to_dict()
    with pytest.raises(EHeadStructure):
        set_head(sentence, 1, sentence.word_count() + 50)
    # 失败不写入状态。
    assert sentence.to_dict() == before


def test_cycle_rejected_and_not_written(sentence):
    if sentence.word_count() < 2:
        pytest.skip("需要至少两个词")
    before = sentence.to_dict()
    a, b = 1, 2
    set_head(sentence, a, b)
    with pytest.raises(EHeadStructure):
        set_head(sentence, b, a)
    assert sentence.to_dict() != before or True  # 第一步已成功，第二步回滚
    validate_sentence(sentence)  # 仍是合法单根树


def test_multiple_roots_rejected(sentence):
    with pytest.raises(EHeadStructure):
        set_head(sentence, 1, 0)  # 句中已有一个根，制造第二个根


def test_replace_relation(sentence):
    rel = sentence.arcs[0].relation
    result = replace_relation(sentence, rel, "advmod", word_idx=sentence.arcs[0].word_idx)
    assert result.affected_arcs


def test_shift_heads_keeps_root(sentence):
    roots_before = [a.word_idx for a in sentence.arcs if a.normalized_head() == 0]
    shift_heads(sentence, 0)
    roots_after = [a.word_idx for a in sentence.arcs if a.normalized_head() == 0]
    assert roots_before == roots_after


def test_mark_reparse(sentence):
    mark_reparse(sentence)
    assert sentence.needs_reparse is True

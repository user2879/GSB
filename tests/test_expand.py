"""业务要求 2：拍位到 tick 展开、确定性与重叠检测。"""
import pytest

from mwb.errors import WorkbenchError
from mwb.expand import beat_to_tick, expand_score
from mwb.models import Note, Score
from mwb.parse import parse_score
from mwb.samples import generate_score


def test_beat_to_tick_rule():
    assert beat_to_tick(0, 480) == 0
    assert beat_to_tick(2, 480) == 960
    assert beat_to_tick(1, 480) == 480


def test_expand_deterministic():
    score = parse_score.__self__ if False else None
    import json, os, tempfile
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "s.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(generate_score(), handle)
    parsed = parse_score(path)
    placed_a, total_a, overlaps_a = expand_score(parsed)
    placed_b, total_b, overlaps_b = expand_score(parsed)
    assert total_a == total_b
    assert [(n.note_id, n.start_tick, n.end_tick) for n in placed_a] == [
        (n.note_id, n.start_tick, n.end_tick) for n in placed_b
    ]
    assert overlaps_a == overlaps_b
    # 时间轴终点为最后一个音符的结束 tick（B08: start=15 拍, 时值 2 拍）
    assert total_a == 17 * 480


def test_cross_voice_overlap_recorded():
    score = Score("t", 480, [
        Note("A1", 60, 0, 2, "vA", 64),
        Note("B1", 48, 1, 2, "vB", 64),
    ])
    placed, total, overlaps = expand_score(score)
    assert overlaps and overlaps[0]["voice_a"] == "vA"
    assert overlaps[0]["start_tick"] == 480 and overlaps[0]["end_tick"] == 960


def test_same_voice_overlap_rejected():
    score = Score("t", 480, [
        Note("A1", 60, 0, 2, "vA", 64),
        Note("A2", 62, 1, 2, "vA", 64),
    ])
    with pytest.raises(WorkbenchError) as info:
        expand_score(score)
    assert info.value.code == "E_OVERLAP_SAME_VOICE"

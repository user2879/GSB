"""业务要求 3：表达状态演进与逐 tick 可定位查询。"""
from mwb.chunkstate import compute_chunk
from mwb.curves import pedal_track, tempo_at_ticks, velocity_map
from mwb.expand import expand_score
from mwb.parse import parse_score
from mwb.parse_params import parse_expression
from mwb.samples import CHUNK_SIZE, generate_expression, generate_score
import json, os, tempfile


def _load():
    tmp = tempfile.mkdtemp()
    sp = os.path.join(tmp, "s.json")
    pp = os.path.join(tmp, "p.json")
    with open(sp, "w", encoding="utf-8") as handle:
        json.dump(generate_score(), handle)
    with open(pp, "w", encoding="utf-8") as handle:
        json.dump(generate_expression(), handle)
    return parse_score(sp), parse_expression(pp)


def test_tempo_interpolation_and_pedal():
    score, params = _load()
    _, total, _ = expand_score(score)
    tempo = tempo_at_ticks(params, score.ticks_per_beat, total)
    assert tempo[0] == 96.0
    assert tempo[total - 1] == 132.0
    pedal = pedal_track(params, score.ticks_per_beat, total)
    assert pedal[0] == 1 and pedal[score.ticks_per_beat * 5] == 0


def test_velocity_mapping_monotone():
    _, params = _load()
    values = [velocity_map(params, v) for v in range(128)]
    assert values[0] == 10 and values[-1] == 120
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_chunk_states_and_query():
    score, params = _load()
    placed, total, _ = expand_score(score)
    chunk = compute_chunk(0, CHUNK_SIZE, total, placed, params, score.ticks_per_beat)
    assert chunk["declared_ticks"] == CHUNK_SIZE
    assert len(chunk["tempo"]) == CHUNK_SIZE and len(chunk["pedal"]) == CHUNK_SIZE
    # 逐 tick 定位：第一个 tick 上 melody M01 处于 active
    first = chunk["per_tick_notes"][0]
    assert any(n["note_id"] == "M01" and n["state"] == "active" for n in first)
    sample = first[0]
    assert set(sample) == {"note_id", "state", "velocity", "tempo", "pedal"}
    # 依赖键覆盖五类参数
    joined = " ".join(chunk["dependency_keys"])
    for prefix in ("tempo:", "velocity:", "stretch:", "articulation:", "pedal:"):
        assert prefix in joined


def test_staccato_has_silent_tail():
    score, params = _load()
    placed, total, _ = expand_score(score)
    chunk = compute_chunk(1, CHUNK_SIZE, total, placed, params, score.ticks_per_beat)
    seg = next(s for s in chunk["note_segments"] if s["note_id"] == "B02")
    assert seg["active_end_tick"] < seg["actual_end_tick"]

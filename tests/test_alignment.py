"""Requirement 3: overlap alignment conflicts are recorded, never overwritten."""

from asr_stream_editor.audio import plan_chunks, plan_frames
from asr_stream_editor.runner import load_resources
from asr_stream_editor.streaming import ChunkResult, merge_chunks


def _fake_chunk(chunks, index, span, label_seq, words):
    labels = [words.index(label_seq[frame - span[0]])
              for frame in range(span[0], span[1])]
    return ChunkResult(index=index, chunk=chunks[index], frame_span=span,
                       labels=labels, tokens=[])


def test_overlap_conflict_is_recorded_and_owner_wins(samples_dir):
    pm, ac, sc, clips, _ = load_resources(samples_dir)
    clip = clips[0]
    words = pm.words
    chunks = plan_chunks(clip.n_samples, sc.chunk_ms, sc.overlap_ms)
    frame_shift = plan_frames(clip.n_samples, ac.frame_length_ms,
                              ac.frame_shift_ms).frame_shift
    r0 = _fake_chunk(chunks, 0, (0, 40), ["a"] * 40, words)
    r1 = _fake_chunk(chunks, 1, (32, 72),
                     ["bo"] * 8 + ["ci"] * 32, words)
    merged = merge_chunks([r0, r1], words, frame_shift, clip.n_samples,
                          sc.significant_digits)
    assert len(merged.conflicts) == 8
    conflict = merged.conflicts[0]
    assert conflict["owner_chunk"] == 0
    assert conflict["conflicting_chunk"] == 1
    assert conflict["kept_word"] == "a"
    assert conflict["dropped_word"] == "bo"
    # one word output per timestamp, earliest chunk's label retained
    assert merged.frame_owner[32] == 0
    assert words[merged.frame_labels[32]] == "a"


def test_agreeing_overlap_produces_no_conflict(samples_dir):
    pm, ac, sc, clips, _ = load_resources(samples_dir)
    clip = clips[0]
    words = pm.words
    chunks = plan_chunks(clip.n_samples, sc.chunk_ms, sc.overlap_ms)
    frame_shift = plan_frames(clip.n_samples, ac.frame_length_ms,
                              ac.frame_shift_ms).frame_shift
    r0 = _fake_chunk(chunks, 0, (0, 40), ["a"] * 40, words)
    r1 = _fake_chunk(chunks, 1, (32, 72), ["a"] * 8 + ["ci"] * 32, words)
    merged = merge_chunks([r0, r1], words, frame_shift, clip.n_samples,
                          sc.significant_digits)
    assert merged.conflicts == []

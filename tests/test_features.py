import numpy as np

from asr_stream_editor.audio import plan_frames
from asr_stream_editor.features import compute_features, features_artifact
from asr_stream_editor.runner import load_resources


def test_last_frame_zero_padded_flag(samples_dir):
    pm, ac, sc, clips, _ = load_resources(samples_dir)
    clip = clips[0]
    features = compute_features(clip.samples, ac, pm, sc.significant_digits)
    plan = plan_frames(clip.n_samples, ac.frame_length_ms, ac.frame_shift_ms)
    assert features.plan.n_frames == plan.n_frames
    artifact = features_artifact(features, clip.source, ac.frame_shift_ms)
    flagged = [row for row in artifact if row["last_frame_zero_padded"]]
    if plan.last_frame_padded:
        assert len(flagged) == 1
        assert flagged[0]["frame_index"] == plan.n_frames - 1
    else:
        assert flagged == []


def test_features_and_emissions_are_elementwise_deterministic(samples_dir):
    pm, ac, sc, clips, _ = load_resources(samples_dir)
    clip = clips[0]
    first = compute_features(clip.samples, ac, pm, sc.significant_digits)
    second = compute_features(clip.samples, ac, pm, sc.significant_digits)
    np.testing.assert_array_equal(first.log_mel, second.log_mel)
    np.testing.assert_array_equal(first.emissions, second.emissions)
    assert first.phonemes == second.phonemes
    assert first.emissions.dtype == np.float64


def test_missing_candidates_filled_with_smooth(samples_dir):
    pm, ac, sc, clips, _ = load_resources(samples_dir)
    clip = clips[0]
    features = compute_features(clip.samples, ac, pm, sc.significant_digits)
    missing_seen = False
    for frame_index, phoneme in enumerate(features.phonemes):
        rows = ac.emission_table[phoneme]
        for word_index, word in enumerate(features.words):
            if word not in rows:
                missing_seen = True
                assert features.emissions[frame_index, word_index] == ac.log_smooth
    assert missing_seen


def test_features_depend_only_on_sample_content(samples_dir):
    pm, ac, sc, clips, _ = load_resources(samples_dir)
    clip = clips[0]
    from asr_stream_editor.audio import plan_chunks
    chunks = plan_chunks(clip.n_samples, sc.chunk_ms, sc.overlap_ms)
    chunk = chunks[1]
    # take a frame-aligned segment so framing geometry is identical
    start = chunk.frame_start_sample if hasattr(chunk, "frame_start_sample") else chunk.start_sample
    part = clip.samples[start:chunk.end_sample]
    part_features = compute_features(part, ac, pm, sc.significant_digits)
    standalone_features = compute_features(part.copy(), ac, pm, sc.significant_digits)
    n_part_frames = part_features.plan.n_frames
    np.testing.assert_array_equal(
        part_features.log_mel[:n_part_frames],
        standalone_features.log_mel[:n_part_frames])
    np.testing.assert_array_equal(
        part_features.emissions[:n_part_frames],
        standalone_features.emissions[:n_part_frames])

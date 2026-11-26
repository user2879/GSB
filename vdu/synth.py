"""Deterministic synthesis of dialogue audio and verbatim transcripts.

All PCM is generated from a 256-entry integer sine table with integer phase
accumulators.  There is no floating point and no random state, so repeated
generation yields byte identical WAV files.
"""

import json
import math
import os
import struct
import wave

SAMPLE_RATE = 16000
SINE_BITS = 8
SINE_SIZE = 1 << SINE_BITS
SINE_TABLE = [
    int(round(32767 * math.sin(2 * math.pi * i / SINE_SIZE)))
    for i in range(SINE_SIZE)
]

WORD_DUR = 1600
WORD_GAP = 240
RAMP = 160

EMOTION_PROFILE = {
    "calm": {"pitch": 1.00, "amp": 0.55},
    "happy": {"pitch": 1.25, "amp": 0.75},
    "angry": {"pitch": 0.80, "amp": 0.95},
    "sad": {"pitch": 0.85, "amp": 0.45},
}
SPEAKER_BASE_PITCH = {"S1": 132.0, "S2": 168.0}
CONTROLLED_VOCAB = [
    "ni", "hao", "jin", "tian", "qi", "bu", "cuo",
    "wo", "hen", "gao", "xing", "zai", "jian", "wei",
]


def _tone_word(length, freq_hz, amplitude, lfo_phase):
    """Render one word as integer PCM with a triangle amplitude LFO."""
    phase_step = int(round(freq_hz * (SINE_SIZE << 16) / SAMPLE_RATE))
    phase = 0
    out = [0] * length
    for i in range(length):
        if i < RAMP:
            env = amplitude * i // RAMP
        elif i > length - RAMP:
            env = amplitude * (length - i) // RAMP
        else:
            env = amplitude
        lfo = 256 + ((i + lfo_phase) % 64) * 4
        env = env * lfo // 256
        out[i] = (env * SINE_TABLE[(phase >> 16) & (SINE_SIZE - 1)]) >> 15
        phase += phase_step
    return out


def _render_segment(segment):
    pcm = [0] * (segment["end"] - segment["start"])
    for word in segment["words"]:
        profile = EMOTION_PROFILE[word["emotion"]]
        base = SPEAKER_BASE_PITCH[segment["speaker"]]
        freq = base * profile["pitch"]
        amplitude = int(round(profile["amp"] * 26000))
        tone = _tone_word(
            word["end"] - word["start"],
            freq,
            amplitude,
            (len(word["word"]) * 17) & 63,
        )
        offset = word["start"] - segment["start"]
        for i, sample in enumerate(tone):
            pcm[offset + i] += sample
    return pcm


def render_pcm(transcript):
    """Mix every segment into a single mono int16 PCM list."""
    total = max(seg["end"] for seg in transcript["segments"])
    pcm = [0] * total
    for segment in transcript["segments"]:
        seg_pcm = _render_segment(segment)
        for i, sample in enumerate(seg_pcm):
            pos = segment["start"] + i
            pcm[pos] += sample
    clipped = [max(-32768, min(32767, sample)) for sample in pcm]
    return clipped


def write_wav(path, pcm, sample_rate=SAMPLE_RATE):
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack("<%dh" % len(pcm), *pcm))


def make_words(seg_start, tokens, emotions, label_version,
               word_dur=WORD_DUR, gap=WORD_GAP):
    words = []
    cursor = seg_start
    for token, emotion in zip(tokens, emotions):
        words.append({
            "word": token,
            "start": cursor,
            "end": cursor + word_dur,
            "emotion": emotion,
            "annotator_version": label_version,
        })
        cursor += word_dur + gap
    return words, cursor - gap


def demo_segment_plan(label_version):
    """Four segments, one overlapping speaker pair and four emotion classes."""
    plan = [
        ("seg01", "S1", 0, ["ni", "hao", "jin", "tian"],
         ["happy", "happy", "happy", "happy"]),
        ("seg02", "S2", 6000, ["wo", "hen", "gao", "xing"],
         ["angry", "angry", "angry", "angry"]),
        ("seg03", "S1", 14000, ["tian", "qi", "bu", "cuo"],
         ["calm", "calm", "calm", "calm"]),
        ("seg04", "S2", 20000, ["zai", "jian", "wei", "ni"],
         ["sad", "sad", "sad", "sad"]),
    ]
    segments = []
    for seg_id, speaker, start, tokens, emotions in plan:
        words, end = make_words(start, tokens, emotions, label_version)
        segments.append({
            "segment_id": seg_id,
            "speaker": speaker,
            "start": start,
            "end": end,
            "label_version": label_version,
            "words": words,
        })
    return segments


def build_transcript(label_version=3):
    segments = demo_segment_plan(label_version)
    return {
        "sample_id": "dialogue_demo",
        "audio": "dialogue_demo.wav",
        "controlled_vocab": list(CONTROLLED_VOCAB),
        "label_version": label_version,
        "segments": segments,
    }


def relabel_transcript(transcript, new_version, changed_segments):
    """Return a deep-copied transcript with a bumped label version.

    Only segments listed in ``changed_segments`` receive new emotion labels,
    which lets tests prove that untouched segments keep reusable evidence.
    """
    new = json.loads(json.dumps(transcript))
    new["label_version"] = new_version
    flip = {"happy": "calm", "angry": "sad", "calm": "happy", "sad": "angry"}
    for segment in new["segments"]:
        segment["label_version"] = new_version
        if segment["segment_id"] in changed_segments:
            for word in segment["words"]:
                word["emotion"] = flip[word["emotion"]]
                word["annotator_version"] = new_version
        else:
            for word in segment["words"]:
                word["annotator_version"] = new_version
    return new


def default_config():
    return {
        "sample_rate": 16000,
        "frame_length": 200,
        "frame_hop": 100,
        "window_length": 800,
        "window_hop": 400,
        "pitch_min_lag": 32,
        "pitch_max_lag": 240,
        "pitch_decimation": 4,
        "cooccur_window": 2,
        "semantic_weight": 0.35,
        "acoustic_weight": 0.65,
        "calibration_gain": 0.25,
        "half_life_rounds": 4,
        "decay_num": 7,
        "decay_den": 8,
        "emotion_thresholds": {
            "negative": -0.6,
            "neutral": 0.0,
            "positive": 0.6,
        },
        "emotion_polarity": {
            "angry": -1, "sad": -1,
            "calm": 0,
            "happy": 1,
        },
        "quantization": {
            "coarse": {"frac_bits": 8, "total_bits": 32},
            "fine": {"frac_bits": 10, "total_bits": 33},
            "reference": {"frac_bits": 14, "total_bits": 40},
        },
        "tolerance": {
            "window": 12,
            "word": 24,
            "calibration": 48,
            "segment": 96,
        },
        "scheduler": {
            "max_workers": 3,
            "batch_budget": 9,
            "segment_quota": 6,
            "test_step_delay": 0.0,
        },
    }


def generate_demo(samples_dir, configs_dir, label_version=3):
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)
    transcript = build_transcript(label_version)
    pcm = render_pcm(transcript)
    wav_path = os.path.join(samples_dir, transcript["audio"])
    write_wav(wav_path, pcm)
    json_path = os.path.join(samples_dir, "dialogue_demo.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(transcript, handle, indent=2, ensure_ascii=False)
    config_path = os.path.join(configs_dir, "calibration.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(default_config(), handle, indent=2, ensure_ascii=False)
    return wav_path, json_path, config_path

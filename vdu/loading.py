"""Loading and deterministic validation of samples, transcripts and configs."""

import json
import os
import wave

from .errors import AudioFormatError, InputInvalidError

REQUIRED_RATE = 16000
REQUIRED_CHANNELS = 1
REQUIRED_WIDTH = 2


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    try:
        return json.loads(text), text
    except json.JSONDecodeError as exc:
        raise InputInvalidError(
            f"malformed JSON: {exc.msg}", path, exc.lineno
        ) from exc


def find_line(text, needle):
    """Return the 1-based line of the first literal ``needle`` occurrence."""
    for index, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return index
    return 1


def validate_wav(path):
    """Verify RIFF/WAVE container and PCM parameters.

    Returns the number of frames.  Any structural violation raises
    :class:`AudioFormatError` with code ``E_AUDIO_FORMAT``.
    """
    if not os.path.exists(path):
        raise AudioFormatError("wav file does not exist", path)
    try:
        with open(path, "rb") as handle:
            tag = handle.read(4)
            if tag != b"RIFF":
                raise AudioFormatError("missing RIFF container tag", path)
            handle.seek(0)
            with wave.open(handle, "rb") as wav:
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                rate = wav.getframerate()
                frames = wav.getnframes()
                data_size = wav.getnframes() * channels * width
                if rate != REQUIRED_RATE:
                    raise AudioFormatError(
                        f"sample rate {rate} != {REQUIRED_RATE}", path)
                if channels != REQUIRED_CHANNELS:
                    raise AudioFormatError(
                        f"channels {channels} != {REQUIRED_CHANNELS}", path)
                if width != REQUIRED_WIDTH:
                    raise AudioFormatError(
                        f"sample width {width} != {REQUIRED_WIDTH} bytes", path)
        with open(path, "rb") as handle:
            raw = handle.read()
        data_pos = raw.find(b"data")
        if data_pos < 0:
            raise AudioFormatError("missing data chunk", path)
        declared = int.from_bytes(raw[data_pos + 4:data_pos + 8], "little")
        if declared != data_size:
            raise AudioFormatError(
                f"data chunk length {declared} != expected {data_size}", path)
        if len(raw) < data_pos + 8 + declared:
            raise AudioFormatError(
                "data chunk length exceeds physical file size", path)
    except wave.Error as exc:
        raise AudioFormatError(f"invalid wave container: {exc}", path) from exc
    return frames


def validate_config(config, path="config"):
    required = [
        "frame_length", "frame_hop", "window_length", "window_hop",
        "cooccur_window", "calibration_gain", "half_life_rounds",
        "decay_num", "decay_den", "emotion_thresholds",
        "emotion_polarity", "quantization", "tolerance", "scheduler",
    ]
    for key in required:
        if key not in config:
            raise InputInvalidError(f"missing config key {key!r}", path)
    if config["frame_hop"] <= 0 or config["window_hop"] <= 0:
        raise InputInvalidError("hop sizes must be positive", path)
    if config["window_length"] < config["frame_length"]:
        raise InputInvalidError(
            "window_length must be >= frame_length", path)
    if config["decay_num"] <= 0 or config["decay_den"] <= 0:
        raise InputInvalidError("decay coefficients must be positive", path)
    if config["decay_num"] > config["decay_den"]:
        raise InputInvalidError(
            "decay numerator must not exceed denominator", path)
    scheduler = config["scheduler"]
    if scheduler["max_workers"] < 1:
        raise InputInvalidError("max_workers must be >= 1", path)
    if scheduler["batch_budget"] < 1 or scheduler["segment_quota"] < 1:
        raise InputInvalidError("budgets and quotas must be >= 1", path)
    return config


def validate_transcript(transcript, audio_frames, path="transcript.json",
                        text=None):
    text = text or json.dumps(transcript)
    vocab = transcript.get("controlled_vocab")
    if not isinstance(vocab, list) or not vocab:
        raise InputInvalidError("controlled_vocab missing or empty", path)
    vocab_set = set(vocab)
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        raise InputInvalidError("segments missing or empty", path)

    seen_ids = set()
    seen_intervals = set()
    previous_version = None
    for seg in segments:
        seg_id = seg.get("segment_id")
        line = find_line(text, str(seg_id)) if seg_id is not None else 1
        if not seg_id or seg_id in seen_ids:
            raise InputInvalidError(
                f"duplicate or empty segment_id {seg_id!r}", path, line)
        seen_ids.add(seg_id)

        speaker = seg.get("speaker")
        if not isinstance(speaker, str) or not speaker:
            raise InputInvalidError(
                f"segment {seg_id} has invalid speaker", path, line)
        start, end = seg.get("start"), seg.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            raise InputInvalidError(
                f"segment {seg_id} bounds must be integers", path, line)
        if start < 0 or end <= start:
            raise InputInvalidError(
                f"segment {seg_id} has invalid interval [{start},{end})",
                path, line)
        if end > audio_frames:
            raise InputInvalidError(
                f"segment {seg_id} end {end} exceeds audio frames "
                f"{audio_frames}", path, line)
        interval_key = (start, end, speaker)
        if interval_key in seen_intervals:
            raise InputInvalidError(
                f"duplicate (start,end,speaker) interval {interval_key}",
                path, line)
        seen_intervals.add(interval_key)

        version = seg.get("label_version")
        if not isinstance(version, int):
            raise InputInvalidError(
                f"segment {seg_id} label_version must be integer",
                path, line)
        if previous_version is not None and version < previous_version:
            raise InputInvalidError(
                f"label_version {version} < previous {previous_version}",
                path, line)
        previous_version = version

        words = seg.get("words")
        if not isinstance(words, list) or not words:
            raise InputInvalidError(
                f"segment {seg_id} has no words", path, line)
        word_end_prev = None
        for word in words:
            wline = find_line(text, str(word.get("word", "")))
            token = word.get("word")
            if token not in vocab_set:
                raise InputInvalidError(
                    f"word {token!r} outside controlled vocab",
                    path, wline)
            w_start, w_end = word.get("start"), word.get("end")
            if not isinstance(w_start, int) or not isinstance(w_end, int):
                raise InputInvalidError(
                    f"word {token!r} offsets must be integers", path, wline)
            if w_end <= w_start:
                raise InputInvalidError(
                    f"word {token!r} has non-positive span", path, wline)
            if word_end_prev is not None and w_start < word_end_prev:
                raise InputInvalidError(
                    f"word {token!r} offsets out of order", path, wline)
            if w_start < start or w_end > end:
                raise InputInvalidError(
                    f"word {token!r} offsets [{w_start},{w_end}) escape "
                    f"segment [{start},{end})", path, wline)
            if not isinstance(word.get("emotion"), str):
                raise InputInvalidError(
                    f"word {token!r} missing emotion label", path, wline)
            if not isinstance(word.get("annotator_version"), int):
                raise InputInvalidError(
                    f"word {token!r} annotator_version must be integer",
                    path, wline)
            word_end_prev = w_end
    return True


def load_sample(sample_dir, transcript_name="dialogue_demo.json"):
    transcript_path = os.path.join(sample_dir, transcript_name)
    transcript, text = load_json(transcript_path)
    wav_path = os.path.join(sample_dir, transcript["audio"])
    frames = validate_wav(wav_path)
    validate_transcript(transcript, frames, transcript_path, text)
    return {
        "transcript_path": transcript_path,
        "wav_path": wav_path,
        "transcript": transcript,
        "frames": frames,
    }


def load_config(config_path):
    config, _ = load_json(config_path)
    validate_config(config, config_path)
    return config

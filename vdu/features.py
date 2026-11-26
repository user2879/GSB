"""Windowed fixed-point acoustic feature extraction.

The accumulation order is explicit: outer loop over frame indices ascending,
inner loop over feature dimensions ascending.  No array reduction defaults and
no global random state are used, so identical inputs produce byte-identical
feature integers.
"""

from . import fixedpoint as fp
from .pcmio import read_pcm

FEATURE_DIMS = 6
ENERGY_SCALE = 1 << 14


def _normalize_energy(raw_energy):
    """Bound energy into 0..ENERGY_SCALE using deterministic integer scaling."""
    return min(ENERGY_SCALE, raw_energy >> 6)


def frame_starts(total, length, hop):
    starts = []
    pos = 0
    while pos + length <= total:
        starts.append(pos)
        pos += hop
    return starts


def window_starts(total, length, hop):
    starts = []
    pos = 0
    while pos < total:
        starts.append(pos)
        if pos + length >= total:
            break
        pos += hop
    return starts


def _decimate(samples, factor):
    return [samples[i] for i in range(0, len(samples), factor)]


def _pitch_lag(samples, cfg):
    """Integer autocorrelation pitch period; zero means unvoiced."""
    dec = _decimate(samples, cfg["pitch_decimation"])
    count = len(dec)
    min_lag = max(1, cfg["pitch_min_lag"] // cfg["pitch_decimation"])
    max_lag = min(count - 1, cfg["pitch_max_lag"] // cfg["pitch_decimation"])
    zero_energy = 0
    for value in dec:
        zero_energy += value * value
    if zero_energy == 0:
        return 0
    best_lag = 0
    best_corr = 0
    lag = min_lag
    while lag <= max_lag:
        corr = 0
        for i in range(count - lag):
            corr += dec[i] * dec[i + lag]
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
        lag += 1
    # Voicing gate: best correlation must exceed 30% of zero-lag energy.
    if best_corr * 100 <= zero_energy * 30:
        return 0
    return best_lag * cfg["pitch_decimation"]


def frame_measures(pcm, start, length, cfg):
    frame = pcm[start:start + length]
    energy = 0
    zcr = 0
    previous = 0
    for sample in frame:
        energy += sample if sample >= 0 else -sample
        sign = 1 if sample >= 0 else -1
        if previous != 0 and sign != previous:
            zcr += 1
        previous = sign
    return {
        "energy": energy,
        "zcr": zcr,
        "pitch": _pitch_lag(frame, cfg),
    }


def _segments_for_frame(frame_start, frame_end, segments):
    hits = []
    for seg in segments:
        if frame_start < seg["end"] and seg["start"] < frame_end:
            hits.append(seg["segment_id"])
    return hits


def build_windows(pcm, transcript, config, tier):
    length = config["frame_length"]
    hop = config["frame_hop"]
    win_len = config["window_length"]
    win_hop = config["window_hop"]
    segments = transcript["segments"]
    frames = frame_starts(len(pcm), length, hop)
    frame_meas = [frame_measures(pcm, start, length, config)
                  for start in frames]

    windows = []
    for win_index, win_start in enumerate(
            window_starts(len(pcm), win_len, win_hop)):
        win_end = min(win_start + win_len, len(pcm))
        member_frames = [idx for idx, start in enumerate(frames)
                         if win_start <= start < win_end]
        seg_refs = {}
        for idx in member_frames:
            f_start = frames[idx]
            f_end = min(f_start + length, len(pcm))
            for seg_id in _segments_for_frame(f_start, f_end, segments):
                seg_refs[seg_id] = seg_refs.get(seg_id, 0) + 1

        sum_energy = 0
        max_energy = 0
        sum_zcr = 0
        voiced_lags = []
        rhythm = 0
        previous_energy = None
        for idx in member_frames:  # frame index ascending, fixed order
            meas = frame_meas[idx]
            sum_energy += meas["energy"]  # dim 0 accumulator
            if meas["energy"] > max_energy:  # dim 1 accumulator
                max_energy = meas["energy"]
            sum_zcr += meas["zcr"]  # dim 2 accumulator
            if meas["pitch"] > 0:
                voiced_lags.append(meas["pitch"])  # dims 3,4 source
            if previous_energy is not None and meas["energy"] > previous_energy:
                rhythm += 1  # dim 5 accumulator
            previous_energy = meas["energy"]

        n_frames = max(1, len(member_frames))
        mean_energy = fp.div_by_int(
            fp.from_int(_normalize_energy(sum_energy), tier), n_frames, tier)
        env_max = fp.from_int(_normalize_energy(max_energy), tier)
        mean_zcr = fp.div_by_int(fp.from_int(sum_zcr, tier), n_frames, tier)
        if voiced_lags:
            pitch_sum = 0
            pitch_min = voiced_lags[0]
            pitch_max = voiced_lags[0]
            for lag in voiced_lags:
                pitch_sum += lag
                if lag < pitch_min:
                    pitch_min = lag
                if lag > pitch_max:
                    pitch_max = lag
            pitch_mean = fp.div_by_int(fp.from_int(pitch_sum, tier),
                                       len(voiced_lags), tier)
            pitch_var = fp.from_int(pitch_max - pitch_min, tier)
        else:
            pitch_mean = 0
            pitch_var = 0
        rhythm_fp = fp.from_int(rhythm, tier)

        features = [0] * FEATURE_DIMS
        # Dimension index ascending, fixed assignment order.
        features[0] = fp.clamp_to_width(mean_energy, tier)
        features[1] = fp.clamp_to_width(env_max, tier)
        features[2] = fp.clamp_to_width(mean_zcr, tier)
        features[3] = fp.clamp_to_width(pitch_mean, tier)
        features[4] = fp.clamp_to_width(pitch_var, tier)
        features[5] = fp.clamp_to_width(rhythm_fp, tier)

        windows.append({
            "window_id": "w%04d" % win_index,
            "start": win_start,
            "end": win_end,
            "frame_indices": list(member_frames),
            "segment_refs": dict(sorted(seg_refs.items())),
            "features": features,
        })
    return windows


def load_windows(wav_path, transcript, config, tier):
    pcm = read_pcm(wav_path)
    return build_windows(pcm, transcript, config, tier)

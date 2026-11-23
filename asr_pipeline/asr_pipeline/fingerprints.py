"""Content fingerprints and derived cache keys."""

import hashlib
import json
import os


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_key(*parts):
    """Join pre-hashed or primitive parts into one stable cache key.

    Inputs that are already full SHA-256 fingerprints are re-hashed whole
    (never truncated) so a single changed audio byte invalidates every
    downstream cache entry.
    """
    joined = "|".join(str(part) for part in parts)
    return sha256_text(joined)


def config_fingerprint(score_params, schedule_config):
    from .config import canonical_json

    payload = {
        "score": {k: v for k, v in score_params.items()},
        "schedule": schedule_config,
    }
    return sha256_text(canonical_json(payload))


def acoustic_fingerprint(score_params):
    """Inputs that change frame features / raw emission logits."""
    from .config import canonical_json
    payload = {
        key: score_params[key]
        for key in (
            "frame_length_ms", "frame_shift_ms", "mel_bands", "vocab",
            "emission_table", "log_smooth", "significant_digits",
        )
    }
    return sha256_text(canonical_json(payload))


def lm_fingerprint(score_params):
    """Inputs that change Viterbi transitions but not the emissions."""
    from .config import canonical_json
    payload = {
        key: score_params[key]
        for key in ("vocab", "bigram_probs", "unigram_probs", "backoff_weight")
    }
    return sha256_text(canonical_json(payload))


def audio_cache_key(audio_fingerprint):
    return short_key("audio", audio_fingerprint)


def emission_cache_key(audio_fingerprint, acoustic_fingerprint, quant_mode, frac_bits,
                       shard_index, frame_start, frame_end):
    return short_key(
        "emission", audio_fingerprint, acoustic_fingerprint, quant_mode, frac_bits,
        shard_index, frame_start, frame_end,
    )


def decode_cache_key(emission_key, lm_fingerprint, quant_mode, frac_bits, shard_index):
    return short_key("decode", emission_key, lm_fingerprint, quant_mode, frac_bits,
                     shard_index)


def stitch_cache_key(decode_keys, config_fingerprint, overlap_ms):
    return short_key("stitch", decode_keys, config_fingerprint, overlap_ms)


def atomic_write_json(path, payload):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

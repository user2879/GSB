"""Content fingerprints and stage cache-key derivation."""

import hashlib
import json


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(obj):
    return sha256_text(canonical(obj))


def file_fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(stage, upstream, config_fp, tier_name, label_fp=None):
    payload = {
        "stage": stage,
        "upstream": upstream,
        "config": config_fp,
        "tier": tier_name,
        "label": label_fp or "none",
    }
    return stage + ":" + sha256_text(canonical(payload))[:24]


def config_fingerprint(config):
    return fingerprint(config)


def label_fingerprint(transcript):
    payload = [
        {"segment_id": seg["segment_id"],
         "label_version": seg["label_version"],
         "words": [
             {"index": idx,
              "emotion": word["emotion"],
              "annotator_version": word["annotator_version"]}
             for idx, word in enumerate(seg["words"])
         ]}
        for seg in transcript["segments"]
    ]
    return fingerprint(payload)


def structure_fingerprint(transcript):
    payload = [
        {"segment_id": seg["segment_id"], "speaker": seg["speaker"],
         "start": seg["start"], "end": seg["end"],
         "words": [
             {"index": idx, "word": word["word"],
              "start": word["start"], "end": word["end"]}
             for idx, word in enumerate(seg["words"])
         ]}
        for seg in transcript["segments"]
    ]
    return fingerprint(payload)

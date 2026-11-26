"""Reusable artifact ledger with a per-key hit/invalidation audit trail."""

import json
import os

from .fingerprint import fingerprint, cache_key


class Ledger:
    """Persists cache keys, references and the hit/invalidate audit trail."""

    def __init__(self, path):
        self.path = path
        self.entries = {}
        self.audit = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                blob = json.load(handle)
            self.entries = blob.get("entries", {})
            self.audit = blob.get("audit", [])

    def record(self, key, stage, refs, status, reason, artifact=None):
        self.entries[key] = {
            "stage": stage,
            "refs": refs,
            "status": status,
            "reason": reason,
            "artifact": artifact,
        }
        self.audit.append({
            "key": key,
            "stage": stage,
            "refs": refs,
            "status": status,
            "reason": reason,
        })

    def lookup(self, key):
        return self.entries.get(key)

    def reusable(self, key):
        existing = self.entries.get(key)
        return (existing is not None and existing["status"] == "valid"
                and existing.get("artifact") is not None)

    def reuse(self, key, reason="cache hit"):
        existing = self.entries[key]
        self.audit.append({
            "key": key,
            "stage": existing["stage"],
            "refs": existing["refs"],
            "status": "reused",
            "reason": reason,
        })
        return existing

    def save(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"entries": self.entries, "audit": self.audit},
                      handle, indent=2, ensure_ascii=False)


def window_key(window, audio_fp, cfg_fp, tier_name):
    upstream = {
        "audio": audio_fp,
        "window": {"start": window["start"], "end": window["end"],
                   "frame_indices": window["frame_indices"],
                   "segment_refs": window["segment_refs"]},
    }
    return cache_key("window", fingerprint(upstream), cfg_fp, tier_name)


def semantic_input_fp(sequence, position, config):
    k = config["cooccur_window"]
    lo = max(0, position - k)
    hi = min(len(sequence), position + k + 1)
    neighborhood = [
        {"emotion": sequence[i]["emotion"], "word": sequence[i]["word"]}
        for i in range(lo, hi)
    ]
    return fingerprint({"neighborhood": neighborhood,
                        "cooccur_window": k})


def word_key(struct_fp, semantic_fp, window_key_list, token, emotion,
             cfg_fp, tier_name, label_fp):
    upstream = {
        "structure": struct_fp,
        "semantic_input": semantic_fp,
        "windows": sorted(window_key_list),
        "token": token,
        "emotion": emotion,
    }
    return cache_key("word", fingerprint(upstream), cfg_fp, tier_name,
                     label_fp)


def calibration_key(word_key_list, cfg_fp, tier_name, label_fp):
    return cache_key("calibration", fingerprint(sorted(word_key_list)),
                     cfg_fp, tier_name, label_fp)


def segment_key(seg_id, calibration_key_list, word_key_list, window_key_list,
                cfg_fp, tier_name, label_fp):
    upstream = {
        "segment_id": seg_id,
        "calibration": sorted(calibration_key_list),
        "words": sorted(word_key_list),
        "windows": sorted(window_key_list),
    }
    return cache_key("segment", fingerprint(upstream), cfg_fp, tier_name,
                     label_fp)

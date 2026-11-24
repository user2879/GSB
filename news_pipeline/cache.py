"""Content-addressed stage cache with sentence/rule level reuse tracking."""

import json
import os

from .hashing import canonical


class FingerprintCache:
    """Persists stage artifacts keyed by upstream content/config fingerprints."""

    def __init__(self, path):
        self.path = path
        self.entries = {}
        self.reuse_log = []
        self.invalidation_log = []
        self.run_hits = {"segmentation": 0, "scoring": 0, "selection": 0,
                         "rewrite": 0, "summary": 0}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                self.entries = json.load(handle)

    def get(self, stage, cache_key):
        entry = self.entries.get(stage, {}).get(cache_key)
        if entry is not None:
            self.reuse_log.append({"stage": stage, "cache_key": cache_key, "artifact_id": entry.get("artifact_id")})
            self.run_hits[stage] = self.run_hits.get(stage, 0) + 1
        return entry

    def put(self, stage, cache_key, artifact, artifact_id, dependencies=None):
        self.entries.setdefault(stage, {})[cache_key] = {
            "artifact_id": artifact_id,
            "dependencies": dependencies or [],
            "artifact": artifact,
        }

    def diff_against(self, previous_keys):
        """Record which (stage, key) pairs disappeared after an edit run."""
        current = {(stage, key) for stage, bucket in self.entries.items() for key in bucket}
        for stage, key in previous_keys:
            if (stage, key) not in current:
                self.invalidation_log.append({"stage": stage, "cache_key": key})

    def snapshot_keys(self):
        return [(stage, key) for stage, bucket in self.entries.items() for key in bucket]

    def save(self, tmp_path):
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(self.entries, handle, ensure_ascii=False, sort_keys=True, indent=2)

    def drain_reuse_log(self):
        items = list(self.reuse_log)
        self.reuse_log = []
        return items

    def drain_invalidation_log(self):
        items = sorted(self.invalidation_log, key=lambda item: (item["stage"], item["cache_key"]))
        self.invalidation_log = []
        return items

    def stats(self):
        return {stage: len(bucket) for stage, bucket in sorted(self.entries.items())}


def sentence_dependencies(score_record, rewrite_record=None):
    """Original sentence + hit rule sets backing one pipeline sentence."""
    deps = {
        "sentence_id": score_record["sentence_id"],
        "source_sentence": score_record["text"],
        "paragraph_pid": score_record["paragraph_pid"],
        "hit_rules": [item["rule_id"] for item in (rewrite_record or {}).get("applied_rules", [])],
    }
    return deps


def stable_canonical(obj):
    return canonical(obj)

"""Content-fingerprint cache for category-level precise invalidation."""

import json
from pathlib import Path

from .fingerprint import sha256_json


def detection_key(sample_id, image_fp, config):
    return sha256_json(
        {
            "kind": "detection",
            "sample": sample_id,
            "image": image_fp,
            "params": config.detection_params_fingerprint(),
        }
    )


def category_fingerprints(sample, category):
    desc = [a for a in sample.get("description", []) if a["category"] == category]
    label_a = [e for e in sample.get("label_a", []) if e["category"] == category]
    label_b = [e for e in sample.get("label_b", []) if e["category"] == category]
    return {
        "description": sha256_json(desc),
        "label_a": sha256_json(label_a),
        "label_b": sha256_json(label_b),
    }


def category_key(sample_id, category, image_fp, fps, config):
    return sha256_json(
        {
            "kind": "category",
            "sample": sample_id,
            "category": category,
            "image": image_fp,
            "description": fps["description"],
            "label_a": fps["label_a"],
            "label_b": fps["label_b"],
            "config": config.fingerprint(),
        }
    )


class CacheStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"detection": {}, "categories": {}}

    def get_detection(self, key):
        return self.data["detection"].get(key)

    def put_detection(self, key, value):
        self.data["detection"][key] = value

    def get_category(self, key):
        return self.data["categories"].get(key)

    def put_category(self, key, value):
        self.data["categories"][key] = value

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self.data, ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            tmp.replace(self.path)

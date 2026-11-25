"""Persistent fragment-level cache keys and precise impact planning."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .hashing import fingerprint


class CacheStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self.data: dict[str, dict[str, Any]] = {"entries": {}}
        if self.path and self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("entries"), dict):
                    self.data = loaded
            except json.JSONDecodeError:
                self.data = {"entries": {}}

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self.data["entries"].get(key)
            return None if entry is None else entry.get("value")

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self.data["entries"][key] = {"value": value}

    def get_or_set(self, key: str, producer):
        with self._lock:
            if key in self.data["entries"]:
                return self.data["entries"][key]["value"], True
            value = producer()
            self.data["entries"][key] = {"value": value}
            return value, False

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def entry_content_fingerprint(entry: dict[str, Any]) -> str:
    return fingerprint(
        {
            "term": entry["term"],
            "frequency": entry["frequency"],
            "pos": entry["pos"],
            "pos_tags": entry.get("pos_tags", [entry["pos"]]),
            "enabled": entry["enabled"],
            "tag_weights": entry.get("tag_weights", {}),
        },
        20,
    )


def entry_fingerprint_map(lexicon: dict[str, Any]) -> dict[str, str]:
    return {entry["id"]: entry_content_fingerprint(entry) for entry in lexicon["entries"]}


def segment_fragment_key(
    fragment: dict[str, Any],
    index: dict[str, Any],
    effective_rule_fp: str,
    rule_package_fp: str | None = None,
) -> str:
    refs: dict[str, dict[str, Any]] = {}
    for entry_id in fragment.get("entry_refs", []):
        entry = index["by_id"][entry_id]
        refs[entry_id] = {"content_fp": entry_content_fingerprint(entry), "version": entry["version"]}
    scoped_revision = fingerprint(refs, 16)
    return fingerprint(
        {
            "kind": "segment_candidate",
            "doc_id": fragment["fragment_id"].split(":", 1)[0],
            "start": fragment["start"],
            "end": fragment["end"],
            "surface_fp": fingerprint(fragment["surface"], 16),
            "refs": refs,
            "lexicon_revision": scoped_revision,
            "rule_package_fingerprint": rule_package_fp or effective_rule_fp,
            "effective_rule_fingerprint": effective_rule_fp,
            "rule_ids": fragment.get("rule_ids", []),
            "algorithm": "deterministic-fmm-rule-v1",
        },
        24,
    )


def tag_candidate_key(
    segment_key: str,
    fragment: dict[str, Any],
    config: dict[str, Any],
    scoped_weights: dict[str, float],
) -> str:
    fp_config = config["fixed_point"]
    return fingerprint(
        {
            "kind": "tag_candidate_set",
            "segment_key": segment_key,
            "tag_codes": fragment.get("tag_codes", []),
            "candidate_scores": [(item["tag"], item["emission_raw"]) for item in fragment.get("candidate_scores", [])],
            "scoped_transition_weights": dict(sorted(scoped_weights.items())),
            "fraction_bits": config["effective_fraction_bits"],
            "quantization_tier": fp_config["quantization_tier"],
            "rounding_mode": fp_config["rounding_mode"],
        },
        24,
    )


def path_total_key(tag_key: str, previous_path_key: str | None, config: dict[str, Any]) -> str:
    return fingerprint(
        {
            "kind": "viterbi_path_total",
            "tag_key": tag_key,
            "upstream_path_key": previous_path_key,
            "fraction_bits": config["effective_fraction_bits"],
            "quantization_tier": config["fixed_point"]["quantization_tier"],
            "algorithm": "fixed-point-viterbi-v1",
        },
        24,
    )


def final_annotation_key(path_key: str, selected_code: int, exchange_version: str | None = None) -> str:
    return fingerprint(
        {
            "kind": "final_annotation",
            "path_key": path_key,
            "selected_code": selected_code,
            "exchange_version": exchange_version,
        },
        24,
    )


def serialization_key(annotation_key_value: str, exchange_version: str, schema: dict[str, Any]) -> str:
    return fingerprint(
        {
            "kind": "serialization_product",
            "annotation_key": annotation_key_value,
            "exchange_version": exchange_version,
            "schema": schema,
        },
        24,
    )


def build_manifest(
    lexicon: dict[str, Any],
    index: dict[str, Any],
    weights_document: dict[str, Any],
    transitions: dict[str, Any],
    rules_document: dict[str, Any],
    rule_package_fp: str,
    config: dict[str, Any],
    segmentations: list[dict[str, Any]],
    taggings: list[dict[str, Any]],
    exchange_version: str,
) -> dict[str, Any]:
    tagging_by_doc = {item["doc_id"]: item for item in taggings}
    fragments: list[dict[str, Any]] = []
    for segmentation in sorted(segmentations, key=lambda item: item["doc_seq"]):
        tagging = tagging_by_doc[segmentation["doc_id"]]
        tagged_by_seq = {item["fragment_seq"]: item for item in tagging["fragments"]}
        previous_path_key: str | None = None
        for fragment in segmentation["fragments"]:
            tagged = tagged_by_seq[fragment["fragment_seq"]]
            seg_key = segment_fragment_key(
                fragment,
                index,
                segmentation["effective_rule_fingerprint"],
                rule_package_fp,
            )
            scoped_weights = {name: transitions["named"].get(name, transitions["default"]) for name in tagged["weight_refs"]}
            candidate_key = tag_candidate_key(seg_key, tagged, config, scoped_weights)
            p_key = path_total_key(candidate_key, previous_path_key, config)
            annotation_key = final_annotation_key(p_key, tagged["selected_code"], exchange_version)
            previous_path_key = p_key
            fragments.append(
                {
                    "doc_id": segmentation["doc_id"],
                    "doc_seq": segmentation["doc_seq"],
                    "fragment_id": fragment["fragment_id"],
                    "fragment_seq": fragment["fragment_seq"],
                    "start": fragment["start"],
                    "end": fragment["end"],
                    "surface": fragment["surface"],
                    "entry_refs": list(fragment["entry_refs"]),
                    "weight_refs": list(tagged["weight_refs"]),
                    "rule_ids": list(fragment["rule_ids"]),
                    "effective_rule_fingerprint": segmentation["effective_rule_fingerprint"],
                    "keys": {
                        "segment": seg_key,
                        "tag_candidates": candidate_key,
                        "path_total": p_key,
                        "annotation": annotation_key,
                        "jv1": serialization_key(annotation_key, "jv1", config["jv1_widths"]),
                        "jv2": serialization_key(annotation_key, "jv2", {"version": "2.0"}),
                    },
                }
            )
    return {
        "lexicon_revision": lexicon.get("revision"),
        "entry_fingerprints": entry_fingerprint_map(lexicon),
        "transition_fingerprint": transitions["fingerprint"],
        "transition_weights": transitions["named"],
        "rule_package_fingerprint": segmentation and segmentations[0]["rule_package_fingerprint"],
        "rule_fingerprints_by_doc": {seg["doc_id"]: seg["effective_rule_fingerprint"] for seg in segmentations},
        "config_hash": config["config_hash"],
        "effective_fraction_bits": config["effective_fraction_bits"],
        "quantization_tier": config["fixed_point"]["quantization_tier"],
        "exchange_version": exchange_version,
        "fragments": sorted(fragments, key=lambda item: (item["doc_seq"], item["fragment_seq"])),
    }


def _changed_entries(old: dict[str, Any], new: dict[str, Any]) -> tuple[set[str], set[str]]:
    old_fp = old.get("entry_fingerprints", {})
    new_fp = new.get("entry_fingerprints", {})
    changed = {entry_id for entry_id, value in new_fp.items() if old_fp.get(entry_id) != value}
    changed.update(entry_id for entry_id in old_fp if entry_id not in new_fp)
    added_terms: set[str] = set()
    return changed, added_terms


def plan_impact(
    old_manifest: dict[str, Any] | None,
    new_manifest: dict[str, Any],
    documents: list[dict[str, Any]] | None = None,
    new_lexicon: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic invalidation and reusable-fragment lists.

    The plan is a dependency classification.  Warm incremental runs turn reuse
    entries into real cache hits; a blank rebuild computes the same final
    artifacts but has cold cache-miss events.
    """

    if not old_manifest:
        return {"invalidations": [], "reuses": [], "changed_entry_ids": [], "changed_weight_keys": [], "rebuild_reason": "no-baseline-manifest"}

    changed_entry_ids, _ = _changed_entries(old_manifest, new_manifest)
    old_weights = old_manifest.get("transition_weights", {})
    new_weights = new_manifest.get("transition_weights", {})
    changed_weight_keys = {key for key, value in new_weights.items() if old_weights.get(key) != value}
    changed_weight_keys.update(key for key in old_weights if key not in new_weights)
    old_rules = old_manifest.get("rule_fingerprints_by_doc", {})
    new_rules = new_manifest.get("rule_fingerprints_by_doc", {})
    changed_rule_docs = {doc_id for doc_id, value in new_rules.items() if old_rules.get(doc_id) != value}
    changed_rule_docs.update(doc_id for doc_id in old_rules if doc_id not in new_rules)
    quant_changed = (
        old_manifest.get("effective_fraction_bits") != new_manifest.get("effective_fraction_bits")
        or old_manifest.get("quantization_tier") != new_manifest.get("quantization_tier")
        or old_manifest.get("config_hash") != new_manifest.get("config_hash")
    )
    exchange_changed = old_manifest.get("exchange_version") != new_manifest.get("exchange_version")

    # Added terms can create matches at positions that did not previously refer
    # to their entry id.  Scan declared document text for those new terms.
    added_entry_ids = {
        entry_id
        for entry_id, fp_value in new_manifest.get("entry_fingerprints", {}).items()
        if entry_id not in old_manifest.get("entry_fingerprints", {})
    }
    added_occurrences: dict[str, list[tuple[int, int, str]]] = {}
    if added_entry_ids and new_lexicon and documents:
        entries_by_id = {entry["id"]: entry for entry in new_lexicon["entries"]}
        for doc in documents:
            occurrences: list[tuple[int, int, str]] = []
            for entry_id in added_entry_ids:
                term = entries_by_id[entry_id]["term"]
                start = doc["text"].find(term)
                while start >= 0:
                    occurrences.append((start, start + len(term), entry_id))
                    start = doc["text"].find(term, start + 1)
            if occurrences:
                added_occurrences[doc["doc_id"]] = occurrences

    old_fragments = sorted(old_manifest.get("fragments", []), key=lambda item: (item["doc_seq"], item["fragment_seq"]))
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for fragment in old_fragments:
        by_doc.setdefault(fragment["doc_id"], []).append(fragment)

    invalidations: list[dict[str, Any]] = []
    invalid_keys: set[tuple[str, int, str]] = set()

    def add_invalidation(fragment: dict[str, Any], stage: str, basis: str) -> None:
        dedupe_key = (fragment["doc_id"], fragment["fragment_seq"], stage)
        if dedupe_key in invalid_keys:
            return
        invalid_keys.add(dedupe_key)
        invalidations.append(
            {
                "doc_id": fragment["doc_id"],
                "doc_seq": fragment["doc_seq"],
                "fragment_id": fragment["fragment_id"],
                "fragment_seq": fragment["fragment_seq"],
                "stage": stage,
                "basis": basis,
                "cache_key": fragment["keys"].get(stage if stage in fragment["keys"] else "segment"),
            }
        )

    for doc_id, fragments in sorted(by_doc.items()):
        downstream_from: int | None = None
        for fragment in fragments:
            direct_reasons: list[str] = []
            if changed_entry_ids.intersection(fragment.get("entry_refs", [])):
                direct_reasons.append("referenced-entry-changed")
            if changed_weight_keys.intersection(fragment.get("weight_refs", [])):
                direct_reasons.append("referenced-transition-weight-changed")
            if doc_id in changed_rule_docs:
                direct_reasons.append("effective-rule-changed")
            for start, end, entry_id in added_occurrences.get(doc_id, []):
                if fragment["start"] < end and start < fragment["end"]:
                    direct_reasons.append(f"added-entry-match:{entry_id}")
                    break

            if direct_reasons:
                downstream_from = fragment["start"] if downstream_from is None else min(downstream_from, fragment["start"])
                for reason in direct_reasons:
                    stage = "segment" if "rule" in reason or "entry" in reason or "added" in reason else "path_total"
                    add_invalidation(fragment, stage, reason)

            if downstream_from is not None and fragment["start"] >= downstream_from:
                if not any(item["fragment_id"] == fragment["fragment_id"] and item["stage"] == "path_total" for item in invalidations):
                    add_invalidation(fragment, "path_total", "downstream-of-direct-dependency-change")
                if "segment" not in fragment["keys"]:
                    pass

            if quant_changed:
                add_invalidation(fragment, "tag_candidates", "quantization-config-changed")
                add_invalidation(fragment, "path_total", "quantization-config-changed")
                add_invalidation(fragment, "annotation", "quantization-config-changed")
            if exchange_changed:
                add_invalidation(fragment, "jv1" if new_manifest.get("exchange_version") == "jv1" else "jv2", "exchange-version-changed")

    invalid_segment_fragments = {
        (item["doc_id"], item["fragment_seq"])
        for item in invalidations
        if item["stage"] == "segment"
    }
    invalid_annotation_fragments = {
        (item["doc_id"], item["fragment_seq"])
        for item in invalidations
        if item["stage"] in {"path_total", "tag_candidates", "annotation"}
    }
    reuses: list[dict[str, Any]] = []
    for fragment in old_fragments:
        key = (fragment["doc_id"], fragment["fragment_seq"])
        if key in invalid_segment_fragments:
            continue
        reuses.append(
            {
                "doc_id": fragment["doc_id"],
                "doc_seq": fragment["doc_seq"],
                "fragment_id": fragment["fragment_id"],
                "fragment_seq": fragment["fragment_seq"],
                "stage": "segment",
                "basis": "segment-dependency-keys-unchanged",
                "cache_key": fragment["keys"]["segment"],
            }
        )
        if key not in invalid_annotation_fragments and not quant_changed:
            reuses.append(
                {
                    "doc_id": fragment["doc_id"],
                    "doc_seq": fragment["doc_seq"],
                    "fragment_id": fragment["fragment_id"],
                    "fragment_seq": fragment["fragment_seq"],
                    "stage": "annotation",
                    "basis": "dependency-keys-unchanged",
                    "cache_key": fragment["keys"]["annotation"],
                }
            )
        if not exchange_changed:
            reuses.append(
                {
                    "doc_id": fragment["doc_id"],
                    "doc_seq": fragment["doc_seq"],
                    "fragment_id": fragment["fragment_id"],
                    "fragment_seq": fragment["fragment_seq"],
                    "stage": "serialization",
                    "basis": "exchange-dependency-keys-unchanged",
                    "cache_key": fragment["keys"]["jv2"],
                }
            )

    invalidations.sort(key=lambda item: (item["doc_seq"], item["fragment_seq"], item["stage"], item["basis"]))
    reuses.sort(key=lambda item: (item["doc_seq"], item["fragment_seq"], item["stage"]))
    return {
        "invalidations": invalidations,
        "reuses": reuses,
        "changed_entry_ids": sorted(changed_entry_ids),
        "changed_weight_keys": sorted(changed_weight_keys),
        "changed_rule_docs": sorted(changed_rule_docs),
        "quant_changed": quant_changed,
        "exchange_changed": exchange_changed,
    }

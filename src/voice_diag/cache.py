from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .util import canonical_json, json_hash


def audio_fingerprint(audio_content_hash: str) -> str:
    return json_hash({"kind": "audio-content", "sha256": audio_content_hash})


def char_offset_fingerprint(interval: Mapping[str, Any]) -> str:
    return json_hash(
        {
            "kind": "char-offsets",
            "interval_id": interval["interval_id"],
            "speaker_id": interval["speaker_id"],
            "text": interval["text"],
            "chars": interval["chars"],
        }
    )


def window_geometry_fingerprint(window: Mapping[str, Any]) -> str:
    return json_hash(
        {
            "kind": "window-geometry",
            "window_id": window["window_id"],
            "window_index": window["window_index"],
            "sample_start": window["sample_start"],
            "sample_end": window["sample_end"],
            "frame_indices": window["frame_indices"],
        }
    )


def feature_cache_key(
    *,
    window: Mapping[str, Any],
    segments: Iterable[Mapping[str, Any]],
    intervals: Iterable[Mapping[str, Any]],
    parameters: Mapping[str, Any],
    global_version: int,
    config_hash: str,
) -> tuple[str, dict[str, Any]]:
    audio_deps = {
        str(segment["interval_id"]): {
            "fragment_id": segment["fragment_id"],
            "content_hash": segment["content_hash"],
            "local_start": segment["local_start"],
            "local_end": segment["local_end"],
        }
        for segment in sorted(segments, key=lambda item: str(item["interval_id"]))
    }
    char_deps = {
        str(interval["interval_id"]): char_offset_fingerprint(interval)
        for interval in sorted(intervals, key=lambda item: str(item["interval_id"]))
    }
    dependencies = {
        "global_version": global_version,
        "config_hash": config_hash,
        "window": window_geometry_fingerprint(window),
        "audio": audio_deps,
        "chars": char_deps,
        "parameters": dict(parameters),
    }
    key = json_hash({"kind": "feature", **dependencies})
    return key, dependencies


def posterior_cache_key(
    *,
    feature_key: str,
    projection_fingerprint: str,
    command_codes: Iterable[int],
    quant_profile: str,
    global_version: int,
) -> tuple[str, dict[str, Any]]:
    dependencies = {
        "upstream_feature": feature_key,
        "projection": projection_fingerprint,
        "command_codes": sorted(int(code) for code in command_codes),
        "quant_profile": quant_profile,
        "global_version": global_version,
    }
    return json_hash({"kind": "posterior", **dependencies}), dependencies


def attribution_cache_key(
    *,
    posterior_key: str,
    window: Mapping[str, Any],
    interval_overlap: Iterable[Mapping[str, Any]],
    candidate_mask_values: Iterable[bool],
    adjudication_parameters: Mapping[str, Any],
    quant_profile: str,
    global_version: int,
) -> tuple[str, dict[str, Any]]:
    dependencies = {
        "upstream_posterior": posterior_key,
        "window": window_geometry_fingerprint(window),
        "intervals": [
            {
                "interval_id": str(item["interval_id"]),
                "start_sample": int(item["start_sample"]),
                "end_sample": int(item["end_sample"]),
                "overlap_frames": int(item["overlap_frames"]),
                "trust_ppm": int(item["trust_ppm"]),
            }
            for item in sorted(interval_overlap, key=lambda value: str(value["interval_id"]))
        ],
        "candidate_mask": [bool(value) for value in candidate_mask_values],
        "adjudication": dict(adjudication_parameters),
        "quant_profile": quant_profile,
        "global_version": global_version,
    }
    return json_hash({"kind": "attribution", **dependencies}), dependencies


def interval_cache_key(
    *,
    interval_id: str,
    source_posterior_keys: Iterable[str],
    receipt_parameters: Mapping[str, Any],
    global_version: int,
) -> tuple[str, dict[str, Any]]:
    dependencies = {
        "interval_id": interval_id,
        "window_posteriors": sorted(source_posterior_keys),
        "receipt_parameters": dict(receipt_parameters),
        "global_version": global_version,
    }
    return json_hash({"kind": "interval-evidence", **dependencies}), dependencies


def conclusion_cache_key(
    *,
    attribution_keys: Iterable[str],
    adjudication_parameters: Mapping[str, Any],
    receipt_parameters: Mapping[str, Any],
    quant_profile: str,
    global_version: int,
) -> tuple[str, dict[str, Any]]:
    dependencies = {
        "window_attributions": sorted(attribution_keys),
        "adjudication": dict(adjudication_parameters),
        "receipt_parameters": dict(receipt_parameters),
        "quant_profile": quant_profile,
        "global_version": global_version,
    }
    return json_hash({"kind": "command-conclusion", **dependencies}), dependencies


def _changed_map_keys(old: Any, new: Any) -> list[str]:
    if not isinstance(old, dict) or not isinstance(new, dict):
        return []
    return sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))


def explain_dependency_change(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if old.get("global_version") != new.get("global_version"):
        reasons.append("global_version_changed")
    if old.get("config_hash") != new.get("config_hash"):
        reasons.append("config_hash_changed")
    if old.get("window") != new.get("window"):
        reasons.append("window_geometry_changed")
    audio_keys = _changed_map_keys(old.get("audio"), new.get("audio"))
    if audio_keys:
        reasons.append("audio_content_changed:" + ",".join(audio_keys))
    chars_keys = _changed_map_keys(old.get("chars"), new.get("chars"))
    if chars_keys:
        reasons.append("transcript_offsets_changed:" + ",".join(chars_keys))
    if old.get("parameters") != new.get("parameters"):
        reasons.append("parameters_changed")
    if old.get("upstream_feature") != new.get("upstream_feature"):
        reasons.append("upstream_feature_changed")
    if old.get("upstream_posterior") != new.get("upstream_posterior"):
        reasons.append("upstream_posterior_changed")
    if old.get("projection") != new.get("projection"):
        reasons.append("projection_changed")
    if old.get("command_codes") != new.get("command_codes"):
        reasons.append("command_vocabulary_changed")
    if old.get("intervals") != new.get("intervals"):
        reasons.append("interval_overlap_changed")
    if old.get("adjudication") != new.get("adjudication"):
        reasons.append("adjudication_parameters_changed")
    if old.get("quant_profile") != new.get("quant_profile"):
        reasons.append("quant_profile_changed")
    if old.get("window_posteriors") != new.get("window_posteriors"):
        reasons.append("interval_upstream_window_changed")
    if old.get("window_attributions") != new.get("window_attributions"):
        reasons.append("conclusion_upstream_window_changed")
    if old.get("receipt_parameters") != new.get("receipt_parameters"):
        reasons.append("receipt_parameters_changed")
    if not reasons:
        reasons.append("cache_key_changed")
    return reasons


@dataclass(frozen=True)
class CacheProbe:
    payload: Any | None
    reused: bool
    key: str
    record: dict[str, Any] | None = None


@dataclass
class ReuseLedger:
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    reuse_audit: list[dict[str, Any]] = field(default_factory=list)
    invalidation_audit: list[dict[str, Any]] = field(default_factory=list)
    miss_audit: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ReuseLedger":
        if not isinstance(value, Mapping):
            return cls()
        entries = value.get("entries", {})
        return cls(
            entries=dict(entries) if isinstance(entries, dict) else {},
            reuse_audit=[dict(item) for item in value.get("reuse_audit", []) if isinstance(item, dict)],
            invalidation_audit=[
                dict(item) for item in value.get("invalidation_audit", []) if isinstance(item, dict)
            ],
            miss_audit=[dict(item) for item in value.get("miss_audit", []) if isinstance(item, dict)],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": {key: self.entries[key] for key in sorted(self.entries)},
            "reuse_audit": list(self.reuse_audit),
            "invalidation_audit": list(self.invalidation_audit),
            "miss_audit": list(self.miss_audit),
        }

    def probe(
        self,
        *,
        cache_id: str,
        stage: str,
        key: str,
        dependencies: Mapping[str, Any],
        refs: Mapping[str, Any],
        stable_key: Iterable[Any],
    ) -> CacheProbe:
        existing = self.entries.get(cache_id)
        if existing is not None and existing.get("key") == key:
            payload = existing.get("payload")
            self.reuse_audit.append(
                {
                    "cache_id": cache_id,
                    "stage": stage,
                    "key": key,
                    "refs": dict(refs),
                    "stable_key": list(stable_key),
                }
            )
            return CacheProbe(payload=payload, reused=True, key=key, record=existing)
        if existing is not None:
            old_dependencies = existing.get("dependencies", {})
            reasons = explain_dependency_change(old_dependencies, dependencies)
            self.invalidation_audit.append(
                {
                    "cache_id": cache_id,
                    "stage": stage,
                    "old_key": existing.get("key"),
                    "new_key": key,
                    "reasons": reasons,
                    "old_refs": existing.get("refs", {}),
                    "new_refs": dict(refs),
                    "stable_key": list(stable_key),
                }
            )
            del self.entries[cache_id]
        else:
            self.miss_audit.append(
                {
                    "cache_id": cache_id,
                    "stage": stage,
                    "key": key,
                    "refs": dict(refs),
                    "stable_key": list(stable_key),
                }
            )
        return CacheProbe(payload=None, reused=False, key=key)

    def put(
        self,
        *,
        cache_id: str,
        stage: str,
        key: str,
        dependencies: Mapping[str, Any],
        refs: Mapping[str, Any],
        stable_key: Iterable[Any],
        payload: Any,
    ) -> None:
        self.entries[cache_id] = {
            "cache_id": cache_id,
            "stage": stage,
            "key": key,
            "dependencies": dict(dependencies),
            "refs": dict(refs),
            "stable_key": list(stable_key),
            "payload": payload,
        }

    def prune_audit(self, *, max_entries: int = 5000) -> None:
        if len(self.reuse_audit) > max_entries:
            self.reuse_audit = self.reuse_audit[-max_entries:]
        if len(self.invalidation_audit) > max_entries:
            self.invalidation_audit = self.invalidation_audit[-max_entries:]
        if len(self.miss_audit) > max_entries:
            self.miss_audit = self.miss_audit[-max_entries:]

    def report(self) -> dict[str, Any]:
        return {
            "reuse_count": len(self.reuse_audit),
            "invalidation_count": len(self.invalidation_audit),
            "miss_count": len(self.miss_audit),
            "reuse_entries": sorted(
                self.reuse_audit,
                key=lambda item: (str(item.get("stage")), str(item.get("cache_id"))),
            ),
            "invalidation_entries": sorted(
                self.invalidation_audit,
                key=lambda item: (str(item.get("stage")), str(item.get("cache_id"))),
            ),
        }


def stable_cache_audit_digest(ledger: ReuseLedger) -> str:
    return json_hash(
        {
            "reuse": sorted(
                canonical_json(item) for item in ledger.reuse_audit
            ),
            "invalidation": sorted(
                canonical_json(item) for item in ledger.invalidation_audit
            ),
        }
    )

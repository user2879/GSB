from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import VERSION_FINGERPRINT
from .config import Config, deep_merge
from .fingerprinting import derive_cache_key, fingerprint
from .formats import result_to_records, serialize_v1, serialization_cache_key
from .models import Request
from .pipeline import RequestResult, _normalization_cache_key, analyze_request


@dataclass
class EngineRun:
    results: dict[str, RequestResult]
    records: dict[str, list[dict[str, Any]]]
    v1_lines: dict[str, list[str]]
    ledger: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": {rid: result.to_dict() for rid, result in self.results.items()},
            "records": self.records,
            "v1_lines": self.v1_lines,
            "ledger": self.ledger,
        }


class IncrementalEngine:
    def __init__(self, config: Config):
        self.config = config
        self.result_cache: dict[str, RequestResult] = {}
        self.segment_cache: dict[str, list[dict[str, Any]]] = {}
        self.serial_cache: dict[str, dict[str, Any]] = {}
        self.run_count = 0

    def update_config(self, patch: dict[str, Any]) -> Config:
        raw = deep_merge(self.config.raw, patch)
        self.config = Config(raw, source=self.config.source)
        return self.config

    def run(self, requests: dict[str, Request] | list[Request]) -> EngineRun:
        self.run_count += 1
        config = self.config
        if isinstance(requests, list):
            request_map = {item.request_id: item for item in requests}
        else:
            request_map = dict(requests)

        results: dict[str, RequestResult] = {}
        records_by_request: dict[str, list[dict[str, Any]]] = {}
        lines_by_request: dict[str, list[str]] = {}
        events: list[dict[str, Any]] = []
        hit_requests: list[str] = []
        recomputed_requests: list[str] = []
        hit_segments: list[str] = []
        recomputed_segments: list[str] = []
        hit_products: list[str] = []
        recomputed_products: list[str] = []

        for request_id in sorted(request_map):
            request = request_map[request_id]
            content_fp = request.content_fingerprint if hasattr(request, "content_fingerprint") else None
            from .pipeline import request_content_fingerprint

            content_fp = request_content_fingerprint(request)
            norm_key = _normalization_cache_key(request, config, content_fp)
            referenced_weights_fp = fingerprint(
                {
                    channel: config.raw["channels"][channel]
                    for channel in sorted(request.channels)
                },
                16,
            )
            result_key = derive_cache_key(
                "request_result",
                normalization=norm_key,
                top_k=config.top_k,
                sampling=config.sampling_fingerprint,
                quantization=config.quantization_fingerprint,
                version=config.derive_version_fingerprint(VERSION_FINGERPRINT),
            )
            segment_events: list[dict[str, Any]] = []
            for channel in sorted(request.channels):
                from .pipeline import segment_fingerprint

                candidates = [c for c in request.candidates if c.channel == channel]
                seg_fp = segment_fingerprint(candidates, channel, config)
                seg_key = derive_cache_key(
                    "normalization_segment",
                    segment_fingerprint=seg_fp,
                    version=config.derive_version_fingerprint(VERSION_FINGERPRINT),
                )
                segment_hit = seg_key in self.segment_cache
                segment_events.append(
                    {
                        "channel": channel,
                        "status": "HIT" if segment_hit else "RECOMPUTED",
                        "cache_key": seg_key,
                        "fingerprint": seg_fp,
                    }
                )
                if segment_hit:
                    hit_segments.append(f"{request_id}:{channel}")
                else:
                    recomputed_segments.append(f"{request_id}:{channel}")

            if result_key in self.result_cache:
                result = self.result_cache[result_key]
                request_status = "HIT"
                hit_requests.append(request_id)
            else:
                result = analyze_request(request, config, self.segment_cache)
                self.result_cache[result_key] = result
                for key in result.cache_keys.values():
                    self.result_cache[key] = result
                request_status = "RECOMPUTED"
                recomputed_requests.append(request_id)

            results[request_id] = result
            records = result_to_records(result, config)
            records_by_request[request_id] = records

            v2_key = serialization_cache_key(
                "v2",
                records,
                config,
                channel_weights_fingerprint=referenced_weights_fp,
            )
            if v2_key in self.serial_cache:
                v2_status = "HIT"
                hit_products.append(f"{request_id}:v2")
            else:
                self.serial_cache[v2_key] = {"records": records}
                v2_status = "RECOMPUTED"
                recomputed_products.append(f"{request_id}:v2")

            v1_key = serialization_cache_key(
                "v1",
                records,
                config,
                channel_weights_fingerprint=referenced_weights_fp,
            )
            if v1_key in self.serial_cache:
                v1_lines = self.serial_cache[v1_key]["v1_lines"]
                v1_status = "HIT"
                hit_products.append(f"{request_id}:v1")
            else:
                v1_lines, failures = serialize_v1(records, config)
                self.serial_cache[v1_key] = {"v1_lines": v1_lines, "failures": [f.to_dict() for f in failures]}
                v1_status = "RECOMPUTED"
                recomputed_products.append(f"{request_id}:v1")
            lines_by_request[request_id] = list(v1_lines)

            candidate_set = sorted((request_id, cid) for cid in result.candidate_ids)
            events.append(
                {
                    "stable_sort_key": f"{request_id}\x1frequest",
                    "request_id": request_id,
                    "status": request_status,
                    "content_fingerprint": content_fp,
                    "referenced_channels": result.channels,
                    "candidate_set": candidate_set,
                    "cache_keys": result.cache_keys,
                    "normalization_cache_key": norm_key,
                    "segments": segment_events,
                    "serialization": {
                        "v1": {
                            "status": v1_status,
                            "cache_key": v1_key,
                            "candidate_set": candidate_set,
                        },
                        "v2": {
                            "status": v2_status,
                            "cache_key": v2_key,
                            "candidate_set": candidate_set,
                        },
                    },
                    "config_hashes": result.config_hashes,
                }
            )

        ledger = {
            "run_number": self.run_count,
            "config_hash": config.config_hash,
            "channel_weights_hash": config.channel_weights_fingerprint,
            "sampling_hash": config.sampling_fingerprint,
            "events": events,
            "hit_requests": hit_requests,
            "recomputed_requests": recomputed_requests,
            "hit_segments": hit_segments,
            "recomputed_segments": recomputed_segments,
            "hit_products": hit_products,
            "recomputed_products": recomputed_products,
            "request_order": sorted(request_map),
        }
        return EngineRun(results, records_by_request, lines_by_request, ledger)


def compare_result_fields(a: EngineRun, b: EngineRun) -> dict[str, Any]:
    """Compare all deterministic business fields while ignoring cache HIT/MISS status."""
    fields = [
        "scores",
        "ranked_ids",
        "top_k_ids",
        "truncated_ids",
        "boundary",
        "layers",
        "sampled_ids",
        "unsampled_topk_ids",
        "shuffle_order",
        "cache_keys",
    ]
    mismatches: list[dict[str, str]] = []
    if sorted(a.results) != sorted(b.results):
        mismatches.append({"field": "request_order", "a": str(sorted(a.results)), "b": str(sorted(b.results))})
    for rid in sorted(set(a.results) & set(b.results)):
        da = a.results[rid].to_dict()
        db = b.results[rid].to_dict()
        for field_name in fields:
            if da.get(field_name) != db.get(field_name):
                mismatches.append({"request_id": rid, "field": field_name})
    return {"identical": not mismatches, "mismatches": mismatches}

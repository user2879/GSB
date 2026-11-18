"""Diagnostic report assembly for a completed pipeline run."""
from __future__ import annotations

from typing import Any

from . import cache as cache_mod
from .config import PairManifest, PipelineConfig
from .error_tracking import build_pair_error_report, build_sample_error_report
from .pipeline import VoiceprintPipeline


def build_report(pipeline: VoiceprintPipeline, pair_manifest: PairManifest,
                 elapsed_s: float) -> dict[str, Any]:
    cfg: PipelineConfig = pipeline.config
    base_width, ref_width = pipeline.base_width, pipeline.ref_width

    sample_errors: dict[str, Any] = {}
    for sid in sorted(pipeline.aggregations):
        if sid not in pipeline.embeddings:
            continue
        base_agg = pipeline.aggregations[sid][base_width]
        ref_agg = pipeline.aggregations[sid][ref_width]
        sample_errors[sid] = build_sample_error_report(
            sid, base_agg, ref_agg,
            pipeline.embeddings[sid][base_width],
            pipeline.embeddings[sid][ref_width],
            cfg.tolerance,
        )

    pair_errors: dict[str, Any] = {}
    pair_table: list[dict[str, Any]] = []
    for pair in sorted(pair_manifest.pairs, key=lambda record: record.pair_id):
        per = pipeline.pair_results.get(pair.pair_id, {})
        base_pair = per.get(base_width)
        ref_pair = per.get(ref_width)
        entry = {
            "pair_id": pair.pair_id,
            "sample_a": pair.sample_a,
            "sample_b": pair.sample_b,
            "declared_relation": pair.relation,
            "reference_keys": pipeline.keys["pairs"][pair.pair_id]["references"],
            "base": _pair_view(base_pair),
            "reference_width": _pair_view(ref_pair),
        }
        pair_table.append(entry)
        if base_pair is not None and ref_pair is not None:
            pair_errors[pair.pair_id] = build_pair_error_report(
                pair.pair_id, sample_errors[pair.sample_a], sample_errors[pair.sample_b],
                base_pair, ref_pair, cfg.tolerance,
            )

    failures = sorted(
        (f.to_dict() if hasattr(f, "to_dict") else f for f in pipeline.failures),
        key=lambda item: item["sort_key"],
    )
    reused = sorted(pipeline.reused)
    recomputed = sorted(pipeline.recomputed)

    report = {
        "report_version": 1,
        "config_hash": cfg.config_hash,
        "injection_hash": pipeline.injection_hash,
        "input_fp": pipeline.index["input_fp"],
        "widths": {"base": base_width, "reference": ref_width},
        "quant_levels": cfg.quant_levels,
        "threshold": cfg.similarity_threshold,
        "tie_rule": cfg.tie_rule,
        "samples": _sample_section(pipeline),
        "embeddings": _embedding_section(pipeline),
        "pair_similarity": pair_table,
        "pair_summary": _pair_summary(pair_table),
        "boundary_sensitive": pipeline.boundary_sensitive,
        "error_propagation": {
            "by_sample": sample_errors,
            "by_pair": pair_errors,
        },
        "incremental": {
            "reused": reused,
            "recomputed": recomputed,
            "reused_count": len(reused),
            "recomputed_count": len(recomputed),
        },
        "failures": failures,
        "snapshot": {
            "global_version": pipeline.version,
            "completed_stages": pipeline.manifest.get("completed_stages", []),
            "artifact_count": len(pipeline.manifest.get("artifact_versions", {})),
        },
        "missing_referenced_samples": pipeline.index["missing_referenced"],
        "elapsed_seconds": elapsed_s,
    }
    return report


def _pair_view(pair: Any) -> Any:
    if pair is None:
        return None
    return pair.to_dict()


def _sample_section(pipeline: VoiceprintPipeline) -> dict[str, Any]:
    section: dict[str, Any] = {}
    for sid in sorted(pipeline.features):
        frames = pipeline.features[sid]
        entry = {
            "sample_id": sid,
            "n_frames": frames.n_frames,
            "tail_frame_indices": [i for i, flag in enumerate(frames.is_tail) if flag],
            "feature_fp": frames.feature_fp,
        }
        if sid in pipeline.aggregations:
            agg = pipeline.aggregations[sid][pipeline.base_width]
            entry["aggregation"] = {
                "valid_frame_count": agg.valid_frame_count,
                "total_frame_count": agg.total_frame_count,
                "valid_frame_indices": agg.valid_frame_indices,
                "stats_dim": int(agg.stats_vector.shape[0]),
                "aggregation_fp": agg.aggregation_fp,
                "partial_summary": agg.partial_summary,
            }
        section[sid] = entry
    return section


def _embedding_section(pipeline: VoiceprintPipeline) -> dict[str, Any]:
    section: dict[str, Any] = {}
    for sid in sorted(pipeline.embeddings):
        base = pipeline.embeddings[sid][pipeline.base_width]
        ref = pipeline.embeddings[sid][pipeline.ref_width]
        float_ref = pipeline.float_embeddings.get(sid)
        section[sid] = {
            "fixed_point": base.to_dict(),
            "reference_width": ref.to_dict(),
            "float_reference": None if float_ref is None else {
                "sample_id": float_ref["sample_id"],
                "embedding_float": float_ref["embedding_float"].tolist(),
                "embedding_fp": float_ref["embedding_fp"],
            },
            "float_differs_from_fixed": _float_differs(base, float_ref),
        }
    return section


def _float_differs(base: Any, float_ref: Any) -> bool:
    if float_ref is None:
        return False
    return bool((base.embedding_float != float_ref["embedding_float"]).any())


def _pair_summary(pair_table: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"same_speaker": [], "different_speaker": []}
    for entry in pair_table:
        relation = entry["declared_relation"]
        base = entry["base"]
        if base is None:
            continue
        summary[relation].append({
            "pair_id": entry["pair_id"],
            "decision": base["decision"],
            "correct": base["decision"] == relation,
            "similarity": base["similarity"],
            "is_threshold_boundary": base["is_threshold_boundary"],
        })
    out: dict[str, Any] = {}
    for relation, rows in summary.items():
        correct = sum(1 for row in rows if row["correct"])
        out[relation] = {
            "pair_count": len(rows),
            "correct_decisions": correct,
            "accuracy": (correct / len(rows)) if rows else None,
            "boundary_pairs": [row["pair_id"] for row in rows if row["is_threshold_boundary"]],
        }
    return out

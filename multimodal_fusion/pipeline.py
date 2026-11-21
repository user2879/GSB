"""End-to-end orchestration for deterministic multimodal fusion."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .budget import BudgetDecision, admit_samples
from .cache import BoundedBlockCache
from .config import FusionConfig, load_config
from .diagnostics import compare_precisions
from .features import VisualStage, extract_visual, visual_stage_key
from .fusion import FusionStage, fuse_and_decide, fusion_stage_key
from .inputs import InputLoadResult, SampleInput, input_content_hash, load_inputs
from .snapshot import SnapshotStore, manifest_payload, restore_decision
from .text import TextStage, encode_text, text_stage_key
from .util import canonical_json, hash_json, q_to_string


SCHEMA_VERSION = 1


@dataclass
class ProfileAnalysis:
    visual: VisualStage
    text: TextStage
    fusion: FusionStage
    keys: dict[str, str]


@dataclass
class SampleAnalysis:
    sample: SampleInput
    configured: ProfileAnalysis
    higher: ProfileAnalysis


class _ReuseCounter:
    def __init__(self) -> None:
        self._counts = {"visual": 0, "text": 0, "fusion": 0}
        self._lock = threading.Lock()

    def increment(self, stage: str) -> None:
        with self._lock:
            self._counts[stage] += 1

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return dict(sorted(self._counts.items()))


def _load_or_compute_visual(
    sample: SampleInput,
    config: FusionConfig,
    profile: Any,
    block_cache: BoundedBlockCache[dict[str, Any]],
    store: SnapshotStore,
    reuse: _ReuseCounter,
) -> VisualStage:
    key = visual_stage_key(sample, config, profile)
    cached = store.get("visual", key)
    if cached is not None:
        try:
            stage = VisualStage.from_dict(cached)
            reuse.increment("visual")
            return stage
        except (KeyError, TypeError, ValueError):
            pass
    stage = extract_visual(sample, config, profile, block_cache)
    store.put("visual", key, stage.as_dict())
    return stage


def _load_or_compute_text(
    sample: SampleInput,
    config: FusionConfig,
    profile: Any,
    store: SnapshotStore,
    reuse: _ReuseCounter,
) -> TextStage:
    key = text_stage_key(sample, config, profile)
    cached = store.get("text", key)
    if cached is not None:
        try:
            stage = TextStage.from_dict(cached)
            reuse.increment("text")
            return stage
        except (KeyError, TypeError, ValueError):
            pass
    stage = encode_text(sample, config, profile)
    store.put("text", key, stage.as_dict())
    return stage


def _load_or_compute_fusion(
    visual: VisualStage,
    text: TextStage,
    config: FusionConfig,
    profile: Any,
    store: SnapshotStore,
    reuse: _ReuseCounter,
) -> FusionStage:
    key = fusion_stage_key(visual, text, config, profile)
    cached = store.get("fusion", key)
    if cached is not None:
        try:
            stage = FusionStage.from_dict(cached)
            reuse.increment("fusion")
            return stage
        except (KeyError, TypeError, ValueError):
            pass
    stage = fuse_and_decide(visual.vector, text.vector, config, profile)
    store.put("fusion", key, stage.as_dict())
    return stage


def _analyze_profile(
    sample: SampleInput,
    config: FusionConfig,
    profile: Any,
    block_cache: BoundedBlockCache[dict[str, Any]],
    store: SnapshotStore,
    reuse: _ReuseCounter,
) -> ProfileAnalysis:
    visual = _load_or_compute_visual(sample, config, profile, block_cache, store, reuse)
    text = _load_or_compute_text(sample, config, profile, store, reuse)
    fusion = _load_or_compute_fusion(visual, text, config, profile, store, reuse)
    return ProfileAnalysis(
        visual=visual,
        text=text,
        fusion=fusion,
        keys={
            "visual": visual.visual_cache_key,
            "text": text.text_cache_key,
            "fusion": fusion_stage_key(visual, text, config, profile),
        },
    )


def _analyze_sample(
    sample: SampleInput,
    config: FusionConfig,
    block_cache: BoundedBlockCache[dict[str, Any]],
    store: SnapshotStore,
    reuse: _ReuseCounter,
) -> SampleAnalysis:
    configured = _analyze_profile(sample, config, config.profile, block_cache, store, reuse)
    higher = _analyze_profile(sample, config, config.diagnostic_profile, block_cache, store, reuse)
    return SampleAnalysis(sample=sample, configured=configured, higher=higher)


def _sample_report(analysis: SampleAnalysis, config: FusionConfig) -> dict[str, Any]:
    low = analysis.configured
    high = analysis.higher
    error_trace = compare_precisions(
        configured_visual=low.visual,
        configured_text=low.text,
        configured_fusion=low.fusion,
        higher_visual=high.visual,
        higher_text=high.text,
        higher_fusion=high.fusion,
        configured_profile=config.profile,
        higher_profile=config.diagnostic_profile,
        tolerance_q=config.tolerance_q,
    )
    return {
        "sample_id": analysis.sample.sample_id,
        "status": "scored",
        "score_q": low.fusion.score_q,
        "score": q_to_string(low.fusion.score_q, config.profile.fractional_bits),
        "threshold_q": low.fusion.threshold_q,
        "threshold": q_to_string(low.fusion.threshold_q, config.profile.fractional_bits),
        "decision": low.fusion.decision,
        "boundary_sample": low.fusion.boundary,
        "visual": low.visual.as_dict(),
        "text": low.text.as_dict(),
        "fusion": low.fusion.as_dict(),
        "cache_keys": {
            "configured": dict(low.keys),
            "higher": dict(high.keys),
        },
        "error_trace": error_trace,
    }


def _global_version(config: FusionConfig, inputs: InputLoadResult) -> str:
    return hash_json(
        {
            "schema": SCHEMA_VERSION,
            "config_hash": config.config_hash,
            "input_content_hash": input_content_hash(inputs.samples, config),
            "configured_profile": config.profile.as_dict(),
            "diagnostic_profile": config.diagnostic_profile.as_dict(),
        },
        namespace="fusion-global-version-v1",
    )


def _stage_versions(analyses: list[SampleAnalysis]) -> dict[str, str]:
    visual_keys: list[str] = []
    text_keys: list[str] = []
    fusion_keys: list[str] = []
    decisions: list[dict[str, Any]] = []
    for analysis in sorted(analyses, key=lambda value: value.sample.sample_id):
        for profile in (analysis.configured, analysis.higher):
            visual_keys.append(profile.keys["visual"])
            text_keys.append(profile.keys["text"])
            fusion_keys.append(profile.keys["fusion"])
        decisions.append(
            {
                "sample_id": analysis.sample.sample_id,
                "score_q": analysis.configured.fusion.score_q,
                "threshold_q": analysis.configured.fusion.threshold_q,
                "decision": analysis.configured.fusion.decision,
                "boundary_sample": analysis.configured.fusion.boundary,
            }
        )
    return {
        "feature_quantization": hash_json(sorted(set(visual_keys)), namespace="visual-stage-version-v1"),
        "text_encoding": hash_json(sorted(set(text_keys)), namespace="text-stage-version-v1"),
        "weighted_fusion": hash_json(sorted(set(fusion_keys)), namespace="fusion-stage-version-v1"),
        "threshold_decision": hash_json(decisions, namespace="decision-version-v1"),
    }


def _write_report(path: str | Path, report: dict[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp = report_path.with_name(f".{report_path.name}.tmp-{threading.get_ident()}")
    payload = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        try:
            import os

            os.fsync(handle.fileno())
        except OSError:
            pass
    temp.replace(report_path)


def _merge_failures(inputs: InputLoadResult, budget: BudgetDecision) -> list[dict[str, Any]]:
    failures = [*inputs.failures, *budget.pending]
    return sorted(failures, key=lambda value: (value["sample_id"], value["code"]))


def run_pipeline(
    config_path: str | Path,
    input_dir: str | Path,
    state_dir: str | Path = ".fusion_state",
    report_path: str | Path | None = None,
    *,
    config: FusionConfig | None = None,
) -> dict[str, Any]:
    config = config or load_config(config_path)
    inputs = load_inputs(config, input_dir)
    global_version = _global_version(config, inputs)
    budget = admit_samples(inputs.samples, config)
    admitted_ids = set(budget.admitted)
    admitted = [sample for sample in inputs.samples if sample.sample_id in admitted_ids]

    store = SnapshotStore(state_dir)
    reuse = _ReuseCounter()
    block_cache: BoundedBlockCache[dict[str, Any]] = BoundedBlockCache(config.cache_blocks)
    analyses: dict[str, SampleAnalysis] = {}
    checkpoint_count = 0
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=config.threads, thread_name_prefix="fusion") as executor:
            futures = {
                executor.submit(_analyze_sample, sample, config, block_cache, store, reuse): sample.sample_id
                for sample in admitted
            }
            for future in as_completed(futures):
                sample_id = futures[future]
                analyses[sample_id] = future.result()
                completed += 1
                if completed % config.snapshot_interval == 0:
                    store.flush()
                    checkpoint_count += 1
        if completed % config.snapshot_interval != 0:
            store.flush()
            checkpoint_count += 1

        ordered_analyses = [analyses[sample_id] for sample_id in sorted(analyses)]
        sample_reports = [_sample_report(analysis, config) for analysis in ordered_analyses]
        stage_versions = _stage_versions(ordered_analyses)
        boundary_samples = [
            report["sample_id"] for report in sample_reports if report["boundary_sample"]
        ]
        failures = _merge_failures(inputs, budget)
        restore = restore_decision(
            store,
            current_global_version=global_version,
            config_hash=config.config_hash,
            tier=config.tier,
            reused=reuse.as_dict(),
        )
        cache_stats = block_cache.stats().as_dict()
        manifest = manifest_payload(
            global_version=global_version,
            config_hash=config.config_hash,
            tier=config.tier,
            fractional_bits=config.fractional_bits,
            word_bits=config.word_bits,
            stage_versions=stage_versions,
            artifact_counts=store.artifact_counts(),
            snapshot_interval=config.snapshot_interval,
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "global_version": global_version,
            "config_hash": config.config_hash,
            "visual_config_hash": config.visual_config_hash,
            "text_config_hash": config.text_config_hash,
            "fusion_config_hash": config.fusion_config_hash,
            "quantization": {
                "configured": config.profile.as_dict(),
                "diagnostic": config.diagnostic_profile.as_dict(),
                "tolerance_q": config.tolerance_q,
                "tolerance": q_to_string(config.tolerance_q, config.fractional_bits),
            },
            "fusion": {
                "weights": {key: str(config.weights[key]) for key in sorted(config.weights)},
                "temperature": str(config.temperature),
                "threshold": str(config.threshold),
                "tie_break": config.tie_break,
            },
            "runtime": {
                "threads_configured": config.threads,
                "thread_pool_workers": config.threads,
                "snapshot_interval": config.snapshot_interval,
                "checkpoint_count": checkpoint_count,
            },
            "budgets": {
                "global_macs": config.global_macs,
                "batch_points": config.batch_points,
                "consumed_macs": budget.consumed_macs,
                "consumed_points": budget.consumed_points,
                "remaining_macs": budget.remaining_macs,
                "remaining_points": budget.remaining_points,
                "cache_blocks": config.cache_blocks,
                "admitted_sample_ids": list(budget.admitted),
                "pending_sample_ids": [item["sample_id"] for item in budget.pending],
            },
            "cache": cache_stats,
            "input": inputs.scan,
            "snapshot": {
                "state_dir": str(Path(state_dir)),
                "committed": False,
                "stage_versions": stage_versions,
                "manifest": manifest,
            },
            "restore": restore,
            "samples": sample_reports,
            "boundary_samples": boundary_samples,
            "failure_list": failures,
            "summary": {
                "input_samples": inputs.scan.get("image_count", 0),
                "valid_samples": len(inputs.samples),
                "invalid_samples": len(inputs.failures),
                "scored_samples": len(sample_reports),
                "pending_samples": len(budget.pending),
                "positive_samples": sum(1 for sample in sample_reports if sample["decision"] == "positive"),
                "negative_samples": sum(1 for sample in sample_reports if sample["decision"] == "negative"),
                "decision_flips": sum(
                    1 for sample in sample_reports if sample["error_trace"]["decision_flip"]
                ),
            },
        }
        report["snapshot"]["committed"] = True
        store.commit(manifest)
        if report_path is not None:
            _write_report(report_path, report)
        return report
    except BaseException:
        store.abort()
        raise

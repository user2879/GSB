"""End-to-end deterministic voiceprint pipeline orchestration."""
from __future__ import annotations

import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import cache as cache_mod
from .aggregation import aggregate
from .audio import WaveData, list_wav_files, read_wav
from .config import PairManifest, PairRecord, PipelineConfig
from .embedding import build_projection_matrix, fixed_embedding, float_reference_embedding
from .error_tracking import build_pair_error_report, build_sample_error_report
from .errors import (
    AudioFormatError,
    ConfigMismatchError,
    PairDependencyError,
    PipelineError,
    SignalShutdown,
    make_failure,
)
from .features import extract_frames
from .similarity import check_boundary_sensitivity, compare_pair
from .util import (
    atomic_write_json,
    fingerprint,
    read_json,
    sha256_bytes,
    sha256_text,
)
from .aggregation import Aggregation
from .embedding import EmbeddingResult
from .features import FrameFeatures
from .similarity import PairSimilarity

_GRACEFUL_SIGNALS = (signal.SIGTERM, signal.SIGINT)


@dataclass
class RunArtifacts:
    features: dict[str, Any]
    aggregations: dict[str, Any]
    embeddings_base: dict[str, Any]
    embeddings_ref: dict[str, Any]
    embeddings_float: dict[str, Any]
    pairs_base: dict[str, Any]
    pairs_ref: dict[str, Any]
    failures: list[dict[str, Any]]
    boundary_sensitive: list[dict[str, Any]]
    sample_errors: dict[str, Any]
    pair_errors: dict[str, Any]
    reused: list[str]
    recomputed: list[str]
    manifest: dict[str, Any]


class ShutdownController:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._prev: dict[int, Any] = {}

    def install(self) -> None:
        for sig in _GRACEFUL_SIGNALS:
            try:
                self._prev[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle)
            except (ValueError, OSError):
                continue

    def restore(self) -> None:
        for sig, handler in self._prev.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                continue

    def _handle(self, signum: int, frame: Any) -> None:
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def checkpoint(self) -> None:
        if self.requested:
            raise SignalShutdown("graceful shutdown requested")


def _input_index(input_dir: str, pair_manifest: PairManifest) -> dict[str, Any]:
    wav_paths = list_wav_files(input_dir)
    index: dict[str, Any] = {}
    for path in wav_paths:
        sample_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "rb") as handle:
            pcm_container_fp = sha256_bytes(handle.read())
        index[sample_id] = {
            "path": path,
            "container_fp": pcm_container_fp,
        }
    referenced = pair_manifest.sample_ids
    missing = sorted(sid for sid in referenced if sid not in index)
    return {
        "samples": index,
        "wav_order": sorted(index),
        "missing_referenced": missing,
        "input_fp": sha256_text(fingerprint(
            {sid: index[sid]["container_fp"] for sid in sorted(index)}
        )),
    }


def _plan_keys(config: PipelineConfig, sample_ids: list[str], pairs: list[PairRecord],
                base_width: int, ref_width: int) -> dict[str, Any]:
    plan: dict[str, Any] = {"samples": {}, "pairs": {}}
    for sid in sample_ids:
        plan["samples"][sid] = {"widths": {}}
    for width in (base_width, ref_width):
        for sid in sample_ids:
            per = plan["samples"][sid]
            per["widths"][width] = {
                "feature": None,
                "aggregation": None,
                "embedding": None,
            }
    return plan


class VoiceprintPipeline:
    def __init__(self, config: PipelineConfig, pair_manifest: PairManifest,
                 input_dir: str, work_dir: str, injection_hash: str) -> None:
        self.config = config
        self.pair_manifest = pair_manifest
        self.input_dir = input_dir
        self.work_dir = work_dir
        self.injection_hash = injection_hash
        self.base_width = config.fixed_point_bits
        self.ref_width = config.fixed_point_bits + 1
        self.store = cache_mod.SnapshotStore(work_dir)
        self.shutdown = ShutdownController()
        self.index = _input_index(input_dir, pair_manifest)
        self.keys = self._build_key_plan()
        self.version = 0
        self.manifest: dict[str, Any] = {}
        self.reused: set[str] = set()
        self.recomputed: set[str] = set()
        self.failures: list[Any] = []
        self.features: dict[str, Any] = {}
        self.aggregations: dict[str, dict[int, Any]] = {}
        self.embeddings: dict[str, dict[int, Any]] = {}
        self.float_embeddings: dict[str, Any] = {}
        self.pair_results: dict[str, dict[int, Any]] = {}
        self.projection_matrix: np.ndarray | None = None
        self._projection_dim: int | None = None

    def _build_key_plan(self) -> dict[str, Any]:
        cfg = self.config
        sample_plan: dict[str, Any] = {}
        for sid, info in self.index["samples"].items():
            feature_key = cache_mod.feature_cache_key(
                sid, info["container_fp"], cfg.scope_hash("feature")
            )
            widths: dict[int, dict[str, str]] = {}
            for width in (self.base_width, self.ref_width):
                agg_key = cache_mod.aggregation_cache_key(feature_key, cfg.scope_hash("stats"))
                emb_key = cache_mod.embedding_cache_key(
                    agg_key, cfg.scope_hash("quant"), cfg.scope_hash("embedding"),
                    width, cfg.quant_levels,
                )
                widths[width] = {
                    cache_mod.STAGE_FEATURES: feature_key,
                    cache_mod.STAGE_AGGREGATION: agg_key,
                    cache_mod.STAGE_EMBEDDING: emb_key,
                }
            sample_plan[sid] = {"feature_key": feature_key, "widths": widths}

        pair_plan: dict[str, Any] = {}
        for pair in self.pair_manifest.pairs:
            per_width: dict[int, str] = {}
            for width in (self.base_width, self.ref_width):
                key_a = sample_plan[pair.sample_a]["widths"][width][cache_mod.STAGE_EMBEDDING]
                key_b = sample_plan[pair.sample_b]["widths"][width][cache_mod.STAGE_EMBEDDING]
                per_width[width] = cache_mod.similarity_cache_key(
                    key_a, key_b, cfg.scope_hash("similarity"), width
                )
            pair_plan[pair.pair_id] = {
                "pair": pair,
                "references": cache_mod.pair_references(
                    pair.pair_id, pair.sample_a, pair.sample_b
                ),
                "widths": per_width,
                "embedding_refs": {
                    width: [
                        sample_plan[pair.sample_a]["widths"][width][cache_mod.STAGE_EMBEDDING],
                        sample_plan[pair.sample_b]["widths"][width][cache_mod.STAGE_EMBEDDING],
                    ]
                    for width in (self.base_width, self.ref_width)
                },
            }
        return {"samples": sample_plan, "pairs": pair_plan}

    def _fresh_manifest(self) -> dict[str, Any]:
        return {
            "global_version": 0,
            "config_hash": self.config.config_hash,
            "injection_hash": self.injection_hash,
            "input_fp": self.index["input_fp"],
            "completed_stages": [],
            "cursors": {},
            "artifact_versions": {},
            "sample_keys": self.keys["samples"],
            "pair_keys": {
                pid: {k: v for k, v in plan.items() if k != "pair"}
                for pid, plan in self.keys["pairs"].items()
            },
            "failures": [],
        }

    def initialize_run(self, resume: bool, allow_config_change: bool = False) -> None:
        if resume and not self.store.has_manifest():
            raise ConfigMismatchError("resume requested but no snapshot manifest exists")
        if resume:
            previous = self.store.read_manifest()
            if not allow_config_change:
                # Config and injection changes invalidate the whole contract -> hard mismatch.
                # Input content changes are legitimate incremental updates: per-stage cache
                # keys carry the content fingerprint, so only affected artifacts miss and rerun.
                for field in ("config_hash", "injection_hash"):
                    if previous.get(field) != self._fresh_manifest()[field]:
                        raise ConfigMismatchError(
                            f"snapshot {field} {previous.get(field)!r} != current "
                            f"{self._fresh_manifest()[field]!r}; refusing to resume"
                        )
            elif previous.get("injection_hash") != self._fresh_manifest()["injection_hash"]:
                raise ConfigMismatchError(
                    "injection manifest changed; only config-scoped changes may be applied "
                    "with allow_config_change"
                )
            self.manifest = previous
            self.manifest["input_fp"] = self.index["input_fp"]
            if allow_config_change:
                self.manifest["config_hash"] = self.config.config_hash
            self.version = int(previous["global_version"])
            self.failures = list(previous.get("failures", []))
            self._failure_keys = {
                (entry["stage"], entry["item_id"]) for entry in self.failures
            }
        else:
            self.manifest = self._fresh_manifest()
            self.store.write_manifest(self.manifest)
            self.version = 0
            self._failure_keys: set[tuple[str, str]] = set()

    def _commit(self, stage: str, cursor: int) -> None:
        self.version += 1
        self.manifest["global_version"] = self.version
        completed = set(self.manifest["completed_stages"])
        completed.add(stage)
        self.manifest["completed_stages"] = sorted(completed)
        self.manifest["cursors"][stage] = cursor
        self.manifest["failures"] = [f.to_dict() if hasattr(f, "to_dict") else f
                                     for f in self.failures]
        self.store.commit(self.manifest)

    def _flush_batch(self, stage: str, batch: list[tuple[str, Any]], cursor: int) -> None:
        """Allocate one new version directory, write the whole batch, then swap manifest."""
        if not batch:
            return
        self.version += 1
        stage_map = self.manifest.setdefault("artifact_versions", {}).setdefault(stage, {})
        for key, payload in batch:
            self.store.write_artifact(self.version, stage, key, payload)
            stage_map[key] = self.version
        self.manifest["global_version"] = self.version
        completed = set(self.manifest["completed_stages"])
        completed.add(stage)
        self.manifest["completed_stages"] = sorted(completed)
        self.manifest["cursors"][stage] = cursor
        self.manifest["failures"] = [
            f.to_dict() if hasattr(f, "to_dict") else f for f in self.failures
        ]
        self.store.commit(self.manifest)

    def _map_threads(self, stage: str, identifiers: list[str],
                     worker: Callable[[str], tuple[str, Any]],
                     commit_interval: int) -> dict[str, Any]:
        results: dict[str, Any] = {}
        ordered = sorted(identifiers)
        pending = list(ordered)
        max_workers = min(self.config.max_workers, max(1, len(pending)))
        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix=f"vp-{stage}") as pool:
            in_flight = {}
            batch: list[tuple[str, Any]] = []
            while pending or in_flight:
                self.shutdown.checkpoint()
                while pending and len(in_flight) < max_workers:
                    identifier = pending.pop(0)
                    future = pool.submit(worker, identifier)
                    in_flight[future] = identifier
                for future in as_completed(list(in_flight)):
                    identifier = in_flight.pop(future)
                    try:
                        key, payload, restored = future.result()
                        results[identifier] = restored
                        batch.append((key, payload))
                    except SignalShutdown:
                        raise
                    except Exception as exc:  # isolation: keep processing the rest
                        code = getattr(exc, "code", "E_UNKNOWN")
                        dedup_key = (stage, identifier)
                        if dedup_key not in self._failure_keys:
                            self._failure_keys.add(dedup_key)
                            self.failures.append(make_failure(identifier, stage, code, exc))
                    if len(batch) >= commit_interval:
                        self._flush_batch(stage, batch, len(results))
                        batch = []
        self._flush_batch(stage, batch, len(results))
        return results

    def _artifact_version_of(self, stage: str, key: str) -> int | None:
        record = self.manifest.get("artifact_versions", {}).get(stage, {})
        return record.get(key)

    def _store_artifact(self, stage: str, key: str, payload: Any) -> None:
        self.store.write_artifact(self.version, stage, key, payload)
        stage_map = self.manifest.setdefault("artifact_versions", {}).setdefault(stage, {})
        stage_map[key] = self.version

    def _load_artifact(self, stage: str, key: str) -> Any:
        version = self._artifact_version_of(stage, key)
        if version is None:
            return None
        if not self.store.artifact_exists(version, stage, key):
            return None
        return self.store.read_artifact(version, stage, key)

    def _get_or_run(self, stage: str, key: str, identifiers: list[str],
                    worker: Callable[[str], Any],
                    reconstruct: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        needed: list[str] = []
        for identifier in identifiers:
            payload = self._load_artifact(stage, key(identifier))
            if payload is not None:
                self.reused.add(cache_mod.stable_task_key(stage, identifier))
                loaded[identifier] = reconstruct(payload)
            else:
                needed.append(identifier)
        fresh = self._map_threads(stage, needed, worker,
                                  self.config.snapshot_commit_interval) if needed else {}
        for identifier in fresh:
            self.recomputed.add(cache_mod.stable_task_key(stage, identifier))
        loaded.update(fresh)
        return loaded

    def run_feature_stage(self) -> None:
        sample_ids = sorted(self.index["samples"])
        def key_of(sid: str) -> str:
            return self.keys["samples"][sid]["feature_key"]

        def worker(sid: str) -> tuple[str, dict, FrameFeatures]:
            info = self.index["samples"][sid]
            wave = read_wav(info["path"], sid)
            frames = extract_frames(wave, self.config)
            payload = frames.to_dict()
            return key_of(sid), payload, frames

        self.features = self._get_or_run(
            cache_mod.STAGE_FEATURES, key_of, sample_ids, worker,
            FrameFeatures.from_dict,
        )

    def run_aggregation_stage(self) -> None:
        self.aggregations = {}
        for width in (self.base_width, self.ref_width):
            def key_of(sid: str, width: int = width) -> str:
                return self.keys["samples"][sid]["widths"][width][cache_mod.STAGE_AGGREGATION]

            def worker(sid: str) -> tuple[str, dict, Aggregation]:
                frames = self.features[sid]
                agg = aggregate(frames, self.config)
                payload = agg.to_dict()
                return key_of(sid), payload, agg

            result = self._get_or_run(
                cache_mod.STAGE_AGGREGATION, key_of, sorted(self.features), worker,
                Aggregation.from_dict,
            )
            for sid, agg in result.items():
                self.aggregations.setdefault(sid, {})[width] = agg

    def _ensure_projection(self, feature_dim: int) -> np.ndarray:
        if self.projection_matrix is None or self._projection_dim != feature_dim:
            self.projection_matrix = build_projection_matrix(self.config, feature_dim)
            self._projection_dim = feature_dim
        return self.projection_matrix

    def run_embedding_stage(self) -> None:
        self.embeddings = {}
        self.float_embeddings = {}
        for width in (self.base_width, self.ref_width):
            def key_of(sid: str, width: int = width) -> str:
                return self.keys["samples"][sid]["widths"][width][cache_mod.STAGE_EMBEDDING]

            def worker(sid: str, width: int = width) -> tuple[str, dict, EmbeddingResult]:
                agg = self.aggregations[sid][width]
                matrix = self._ensure_projection(agg.stats_vector.shape[0])
                emb = fixed_embedding(agg, self.config, width, matrix)
                payload = emb.to_dict()
                return key_of(sid), payload, emb

            result = self._get_or_run(
                cache_mod.STAGE_EMBEDDING, key_of, sorted(self.aggregations), worker,
                EmbeddingResult.from_dict,
            )
            for sid, emb in result.items():
                self.embeddings.setdefault(sid, {})[width] = emb
        # Float reference is a deterministic view rebuilt every run (not a cached artifact),
        # so resume runs produce exactly the same embeddings section as uninterrupted runs.
        for sid in sorted(self.embeddings):
            agg = self.aggregations[sid][self.base_width]
            matrix = self._ensure_projection(agg.stats_vector.shape[0])
            self.float_embeddings[sid] = float_reference_embedding(
                agg, self.config, matrix
            )

    def run_similarity_stage(self) -> None:
        self.pair_results = {}
        self.boundary_sensitive: list[dict[str, Any]] = []
        for width in (self.base_width, self.ref_width):
            def key_of(pair_id: str, width: int = width) -> str:
                return self.keys["pairs"][pair_id]["widths"][width]

            def worker(pair_id: str, width: int = width) -> tuple[str, dict, PairSimilarity]:
                pair = self.keys["pairs"][pair_id]["pair"]
                if (pair.sample_a not in self.embeddings
                        or pair.sample_b not in self.embeddings):
                    raise PairDependencyError(
                        f"pair {pair_id} depends on a failed sample embedding"
                    )
                emb_a = self.embeddings[pair.sample_a][width]
                emb_b = self.embeddings[pair.sample_b][width]
                sim = compare_pair(pair, emb_a, emb_b, self.config, width)
                payload = sim.to_dict()
                return key_of(pair_id), payload, sim

            result = self._get_or_run(
                cache_mod.STAGE_SIMILARITY, key_of,
                sorted(self.keys["pairs"]), worker, PairSimilarity.from_dict,
            )
            for pair_id, sim in result.items():
                self.pair_results.setdefault(pair_id, {})[width] = sim

        for pair_id in sorted(self.pair_results):
            per = self.pair_results[pair_id]
            if self.base_width in per and self.ref_width in per:
                flag = check_boundary_sensitivity(per[self.base_width], per[self.ref_width])
                if flag is not None:
                    self.boundary_sensitive.append(flag)
                    if ("similarity", pair_id) not in self._failure_keys:
                        self._failure_keys.add(("similarity", pair_id))
                        from .errors import BoundarySensitiveError
                        self.failures.append(make_failure(
                            pair_id, cache_mod.STAGE_SIMILARITY,
                            BoundarySensitiveError.code,
                            BoundarySensitiveError(flag["message"]),
                        ))

    def run(self, resume: bool = False, allow_config_change: bool = False) -> int:
        self.shutdown.install()
        try:
            self.reused = set()
            self.recomputed = set()
            self.initialize_run(resume, allow_config_change=allow_config_change)
            try:
                self.run_feature_stage()
                self.run_aggregation_stage()
                self.run_embedding_stage()
                self.run_similarity_stage()
            except SignalShutdown:
                self._commit("shutdown", self._stage_cursor())
                return 0
            referenced = {
                version
                for stage_map in self.manifest.get("artifact_versions", {}).values()
                for version in stage_map.values()
            }
            self.store.prune_old_versions(keep=3, referenced_versions=referenced)
            return 0
        finally:
            self.shutdown.restore()

    def _stage_cursor(self) -> int:
        return len(self.features) + len(self.aggregations) + len(self.embeddings)

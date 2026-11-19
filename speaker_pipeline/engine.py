"""Incremental, content-fingerprinted speaker pipeline engine."""

import json
import os
import shutil
import time

import numpy as np

from .audioio import (
    assert_finite,
    list_wav_files,
    read_wav,
)
from .clustering import (
    apply_overrides,
    attribute_intervals,
    base_agglomerative,
)
from .commands import apply_audio_command, update_override_ledger
from .concurrency import BoundedWorkerPool
from .config import Config
from .errors import (
    AudioFormatError,
    BusinessError,
    NonFiniteSampleError,
)
from .features import extract_features
from .fingerprint import (
    bytes_fingerprint,
    content_fingerprint,
    derived_key,
    vector_fingerprint,
)
from .injections import (
    apply_file_injections,
    check_command_injection,
    check_threshold_injection,
    read_non_finite_poison,
)
from .segmentation import segment_file
from .store import ProductStore

FEATURE_PARAM_KEYS = ("frame_length_ms", "frame_shift_ms", "feature_dim")
SEG_PARAM_KEYS = (
    "energy_threshold",
    "spectral_flatness_threshold",
    "overlap_similarity_threshold",
    "min_speech_ms",
    "min_silence_ms",
)


class PipelineEngine:
    def __init__(self, work_dir, config, store_dir, injections=None):
        self.work_dir = os.path.abspath(work_dir)
        self.config = config
        self.injections = injections or []
        self.store = ProductStore(store_dir)
        self.pool = BoundedWorkerPool(int(config.data["worker_threads"]))
        self.failures = []
        self.stage_timings = {}
        self.override_ledger = []
        self.command_cursor = 0
        self.global_version = 0
        self.threshold = float(config.data["cluster_distance_threshold"])

    # ------------------------------------------------------------------
    def _inject_mapping(self, fault):
        return {
            item["file"]: item["id"]
            for item in self.injections
            if item["fault"] == fault and "file" in item
        }

    def run_pipeline(self):
        """Execute all stages over the current working audio directory."""
        self.failures = []
        self.store.reset_audit()
        timings = {}
        started = time.perf_counter()

        file_paths = list_wav_files(self.work_dir)
        file_names = [os.path.basename(path) for path in file_paths]
        non_finite_map = self._inject_mapping("non_finite")

        # Stage 1+2: per-file audio -> features -> segments.
        per_file = {}
        t0 = time.perf_counter()
        for rank, (path, file_name) in enumerate(
            zip(file_paths, file_names)
        ):
            record = self._process_file(
                rank, path, file_name, non_finite_map
            )
            if record is not None:
                per_file[file_name] = record
        timings["feature_and_segmentation_ms"] = round(
            (time.perf_counter() - t0) * 1000.0, 3
        )

        # Global ordered interval table.
        t0 = time.perf_counter()
        intervals, interval_meta, fingerprints, min_keys, vectors = (
            self._collect_intervals(per_file)
        )
        timings["interval_collection_ms"] = round(
            (time.perf_counter() - t0) * 1000.0, 3
        )

        # Stage 3: deterministic clustering (with override ledger).
        t0 = time.perf_counter()
        try:
            check_threshold_injection(self.config, self.injections)
            clustering, attribution = self._cluster_and_attribute(
                intervals, fingerprints, min_keys, vectors
            )
        except BusinessError as exc:
            self.failures.append(exc.to_record())
            clustering, attribution = None, None
        timings["clustering_and_attribution_ms"] = round(
            (time.perf_counter() - t0) * 1000.0, 3
        )

        self.failures.sort(
            key=lambda row: (
                row.get("file") or "",
                row.get("interval_index") is None,
                row.get("interval_index") or -1,
                row.get("stage") or "",
                row.get("code") or "",
            )
        )

        result = {
            "files": file_names,
            "intervals": intervals,
            "interval_meta": interval_meta,
            "clusters": clustering["clusters"] if clustering else [],
            "attribution": attribution or [],
            "failures": self.failures,
            "reused_products": list(self.store.reused),
            "recomputed_products": list(self.store.recomputed),
            "worker_high_water_mark": self.pool.high_water_mark,
            "worker_threads": self.config.data["worker_threads"],
            "threshold": self.threshold,
        }
        result["stage_timings_ms"] = timings
        result["total_pipeline_ms"] = round(
            (time.perf_counter() - started) * 1000.0, 3
        )
        return result

    # ------------------------------------------------------------------
    def _process_file(self, rank, path, file_name, non_finite_map):
        # Audio validation / read.
        corrupt_map = self._inject_mapping("corrupt_wav")
        try:
            with open(path, "rb") as handle:
                raw_bytes = handle.read()
            audio_key = bytes_fingerprint(raw_bytes)
            samples, meta = read_wav(path)
            if file_name in non_finite_map:
                position = read_non_finite_poison(path)
                if position is not None and position < len(samples):
                    samples[position] = np.nan
                assert_finite(
                    samples,
                    file_name,
                    injection_id=non_finite_map[file_name],
                )
            assert_finite(samples, file_name)
        except AudioFormatError as exc:
            if exc.injection_id is None and file_name in corrupt_map:
                exc.injection_id = corrupt_map[file_name]
            self.failures.append(exc.to_record(file_name=file_name))
            return None
        except NonFiniteSampleError as exc:
            self.failures.append(exc.to_record(file_name=file_name))
            return None

        feature_params = self.config.param_subset(FEATURE_PARAM_KEYS)
        feature_key = derived_key(
            "feature", audio_key, params=feature_params
        )

        def produce_features():
            return extract_features(
                samples, self.config, worker_pool=self.pool
            )

        feature_product, _ = self.store.get_or_compute(
            feature_key,
            "feature",
            produce_features,
            references=[f"audio:{file_name}:{audio_key[:12]}"],
        )

        seg_params = self.config.param_subset(SEG_PARAM_KEYS)
        seg_key = derived_key(
            "segmentation",
            feature_product["_cache_key"],
            params=seg_params,
        )

        def produce_segments():
            return segment_file(feature_product, self.config)

        seg_product, _ = self.store.get_or_compute(
            seg_key,
            "segmentation",
            produce_segments,
            references=[feature_product["_cache_key"]],
        )
        for failure in seg_product.get("failures", []):
            failure = dict(failure)
            failure["file"] = file_name
            self.failures.append(failure)
        return {
            "rank": rank,
            "file": file_name,
            "feature_key": feature_key,
            "segment_key": seg_key,
            "segments": seg_product["intervals"],
        }

    def _collect_intervals(self, per_file):
        intervals, interval_meta, fingerprints, min_keys, vectors = (
            [],
            [],
            [],
            [],
            [],
        )
        for file_name in sorted(per_file):
            record = per_file[file_name]
            for segment in record["segments"]:
                vector = np.asarray(
                    segment["mean_voiceprint"], dtype=np.float64
                )
                fp = (
                    f"{file_name}#"
                    f"{segment['start_frame']}-{segment['end_frame']}#"
                    + vector_fingerprint(vector)[:12]
                )
                intervals.append(dict(segment))
                interval_meta.append(
                    {
                        "file": file_name,
                        "interval_fingerprint": fp,
                        "upstream_feature_key": record["feature_key"],
                        "upstream_segment_key": record["segment_key"],
                    }
                )
                fingerprints.append(fp)
                min_keys.append((record["rank"], segment["start_frame"]))
                vectors.append(vector)
        order = sorted(
            range(len(intervals)),
            key=lambda idx: (
                intervals[idx]["start_ms"],
                intervals[idx]["end_frame"],
                interval_meta[idx]["file"],
            ),
        )
        return (
            [intervals[i] for i in order],
            [interval_meta[i] for i in order],
            [fingerprints[i] for i in order],
            [min_keys[i] for i in order],
            [vectors[i] for i in order],
        )

    def _cluster_and_attribute(self, intervals, fingerprints, min_keys,
                               vectors):
        ledger_fp = content_fingerprint(self.override_ledger)
        cluster_key = derived_key(
            "clustering",
            *[
                f"{fp}:{vector_fingerprint(vec)[:12]}"
                for fp, vec in zip(fingerprints, vectors)
            ],
            threshold=self.threshold,
            ledger=ledger_fp,
        )

        def produce_cluster():
            base = base_agglomerative(
                vectors, min_keys, self.threshold
            )
            assignments, clusters = apply_overrides(
                base, fingerprints, self.override_ledger
            )
            centroids = {}
            for cluster in clusters:
                members = cluster["interval_fingerprints"]
                member_vectors = [
                    vectors[fingerprints.index(member)] for member in members
                ]
                centroids[cluster["cluster_id"]] = np.mean(
                    np.asarray(member_vectors), axis=0
                )
            return {
                "assignments": assignments,
                "clusters": clusters,
                "centroids": {
                    cid: [float(x) for x in centroid]
                    for cid, centroid in centroids.items()
                },
            }

        cluster_product, _ = self.store.get_or_compute(
            cluster_key,
            "clustering",
            produce_cluster,
            references=sorted(set(fingerprints)),
        )

        attr_key = derived_key(
            "attribution",
            cluster_key,
            threshold=self.threshold,
        )

        def produce_attribution():
            centroids = {
                cid: np.asarray(vec, dtype=np.float64)
                for cid, vec in cluster_product["centroids"].items()
            }
            rows = attribute_intervals(
                intervals,
                cluster_product["assignments"],
                centroids,
                fingerprints,
                worker_pool=self.pool,
            )
            return {"rows": rows}

        attr_product, _ = self.store.get_or_compute(
            attr_key,
            "attribution",
            produce_attribution,
            references=[cluster_key],
        )
        return (
            {"clusters": cluster_product["clusters"]},
            attr_product["rows"],
        )

    # ------------------------------------------------------------------
    def apply_command(self, command, current_result=None):
        """Apply one editing command and return its materialization info."""
        command = dict(command)
        command.setdefault("sequence", self.command_cursor)
        ctype = command["type"]

        if ctype in ("trim", "replace"):
            check_command_injection(command)
            info = apply_audio_command(command, self.work_dir)
        elif ctype == "set_threshold":
            value = float(command["cluster_distance_threshold"])
            if value <= 0:
                from .errors import InvalidThresholdError

                raise InvalidThresholdError(
                    "cluster_distance_threshold must be > 0",
                    stage="clustering",
                    injection_id=command.get("injection_id"),
                )
            self.threshold = value
            info = {"type": "set_threshold", "threshold": self.threshold}
        elif ctype in ("merge_clusters", "split_cluster"):
            if current_result is None:
                current_result = self.run_pipeline()
            update_override_ledger(
                self.override_ledger,
                command,
                current_result["clusters"],
                [
                    row["interval_fingerprint"]
                    for row in current_result["attribution"]
                ],
            )
            info = {"type": ctype, "sequence": command["sequence"]}
        else:
            from .errors import CommandError

            raise CommandError(
                f"unsupported command type {ctype}", stage="command"
            )
        self.command_cursor += 1
        return info

    # ------------------------------------------------------------------
    def build_manifest(self, result):
        completed = [
            {
                "file": meta["file"],
                "interval_fingerprint": meta["interval_fingerprint"],
            }
            for meta in result["interval_meta"]
        ]
        # Accumulate every product version currently present in the store,
        # not only the products touched in the latest run, so the manifest
        # remains a complete inventory of recoverable stage products.
        stage_products = sorted(
            name[:-5].replace("__", ":")
            for name in os.listdir(self.store.product_dir)
            if name.endswith(".json")
        )
        return {
            "global_version": self.global_version,
            "config_hash": self.config.hash_key(),
            "command_cursor": self.command_cursor,
            "threshold": self.threshold,
            "override_ledger": self.override_ledger,
            "completed_intervals": completed,
            "stage_products": stage_products,
            "failures": result["failures"],
        }

    def commit_snapshot(self, result, crash_before_commit=False):
        self.global_version += 1
        manifest = self.build_manifest(result)
        return self.store.commit_snapshot(
            manifest, crash_before_commit=crash_before_commit
        )

    def restore(self):
        """Restore from the last fully committed snapshot, if any."""
        manifest = self.store.load_manifest()
        if manifest is None:
            self.store.cleanup_temp()
            return None
        self.global_version = int(manifest["global_version"])
        self.command_cursor = int(manifest["command_cursor"])
        self.threshold = float(manifest["threshold"])
        self.override_ledger = list(manifest.get("override_ledger", []))
        self.store.cleanup_temp()
        return manifest

    # ------------------------------------------------------------------
    def run_command_stream(self, commands, snapshot_interval=None,
                           crash_at_sequence=None):
        """Replay commands incrementally, snapshotting on an interval."""
        interval = (
            self.config.data["snapshot_interval"]
            if snapshot_interval is None
            else snapshot_interval
        )
        report_entries = []
        result = self.run_pipeline()
        initial_manifest = self.commit_snapshot(result)
        report_entries.append(
            {
                "event": "initial_snapshot",
                "version": initial_manifest["global_version"],
            }
        )
        for command in commands:
            sequence = command.get("sequence", self.command_cursor)
            try:
                self.apply_command(command, current_result=result)
            except BusinessError as exc:
                self.failures.append(exc.to_record(file_name=command.get("file")))
                report_entries.append(
                    {
                        "event": "command_rejected",
                        "sequence": sequence,
                        "code": exc.code,
                        "injection_id": getattr(exc, "injection_id", None),
                    }
                )
                continue
            result = self.run_pipeline()
            if (
                crash_at_sequence is not None
                and sequence == crash_at_sequence
            ):
                self.commit_snapshot(
                    result, crash_before_commit=True
                )
            if self.command_cursor % max(1, interval) == 0:
                manifest = self.commit_snapshot(result)
                report_entries.append(
                    {
                        "event": "snapshot",
                        "sequence": sequence,
                        "version": manifest["global_version"],
                    }
                )
        final_manifest = self.commit_snapshot(result)
        report_entries.append(
            {
                "event": "final_snapshot",
                "version": final_manifest["global_version"],
                "command_cursor": self.command_cursor,
            }
        )
        result["snapshot_manifest"] = final_manifest
        result["snapshot_events"] = report_entries
        return result

    # ------------------------------------------------------------------
    def export_state(self):
        """Export mutable state needed for a from-scratch full rebuild."""
        return {
            "threshold": self.threshold,
            "override_ledger": list(self.override_ledger),
            "command_cursor": self.command_cursor,
        }

    def load_state(self, state):
        self.threshold = float(state["threshold"])
        self.override_ledger = list(state["override_ledger"])
        self.command_cursor = int(state.get("command_cursor", 0))

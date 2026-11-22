"""说话人分离主管线：四阶段、缓存失效、并发与快照编排。"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

from .audio import read_wav_bytes
from .cache import STAGES, ContentCache, Snapshot, make_cache_key
from .clusters import run_attribution, run_clustering
from .codes import PipelineError
from .features import extract_features
from .injection import InjectionRegistry
from .segments import segment_audio
from .util import content_hash
from .verification import verify_audio


def _bytes(raw):
    # 字节哈希的确定性可序列化输入
    return list(raw)


class DiarizationPipeline:
    def __init__(self, input_dir, output_dir, config, truth, registry,
                 allow_edit=False, fresh=False):
        self.input_dir = input_dir
        self.config = config
        self.truth = truth
        self.registry = registry if isinstance(registry, InjectionRegistry) \
            else InjectionRegistry(registry)
        self.cache = ContentCache(os.path.join(output_dir, "cache"))
        self.snapshot = Snapshot(os.path.join(output_dir, "snapshot"))
        self.allow_edit = allow_edit
        self.fresh = fresh
        self.stats = {"compute": 0, "reuse": 0,
                      "by_stage": {stage: {"compute": 0, "reuse": 0}
                                   for stage in STAGES}}
        self.failures = []
        self.results = {}
        self.ledgers = []
        self._ledger_lock = threading.Lock()

    def _track_ledger(self, entry):
        with self._ledger_lock:
            self.ledgers.append(entry)
            self.stats["reuse" if entry["action"] == "reuse" else "compute"] += 1
            self.stats["by_stage"][entry["stage"]][entry["action"]] += 1

    def _fail(self, audio, error):
        with self._ledger_lock:
            record = error.to_failure(audio)
            if error.injection_id:
                item = next((i for i in self.registry.items
                             if i["injection_id"] == error.injection_id), None)
                if item:
                    record["injection"] = item
            self.failures.append(record)

    def _process_audio(self, audio, raw_bytes):
        prepared, audio_injection = self.registry.prepare_audio_bytes(audio, raw_bytes)
        audio_key = content_hash({"audio": audio, "bytes": _bytes(prepared)})
        try:
            audio_record = read_wav_bytes(prepared, audio_name=audio)

            feature_key = make_cache_key("features", audio_key, self.config)
            feature_product, _, feature_entry = self.cache.get_or_compute(
                feature_key,
                lambda: extract_features(audio_record, self.config, self.registry, audio),
                audio, "features",
            )
            self._track_ledger(feature_entry)

            frame_lookup = {f["frame_index"]: {"start_ms": f["start_ms"],
                                               "end_ms": f["end_ms"]}
                            for f in feature_product["frames"]}

            segment_key = make_cache_key("segments", feature_key, self.config)
            segment_product, _, segment_entry = self.cache.get_or_compute(
                segment_key,
                lambda: segment_audio(feature_product, self.config, self.registry, audio),
                audio, "segments",
            )
            self._track_ledger(segment_entry)
            segment_product["_frame_lookup"] = frame_lookup
            segment_product["_frames"] = feature_product["frames"]

            cluster_key = make_cache_key("clusters", segment_key, self.config)
            cluster_product, _, cluster_entry = self.cache.get_or_compute(
                cluster_key,
                lambda: run_clustering(segment_product, self.config, self.registry, audio),
                audio, "clusters",
            )
            self._track_ledger(cluster_entry)

            # 归属裁决独立缓存：overlap_rule 变更只令本阶段及核验失效
            attribution_key = make_cache_key("attribution", cluster_key, self.config)
            attribution_product, _, attribution_entry = self.cache.get_or_compute(
                attribution_key,
                lambda: run_attribution(segment_product, cluster_product,
                                        self.config, self.registry, audio),
                audio, "attribution",
            )
            self._track_ledger(attribution_entry)

            truth_entry = self.truth[audio]
            verification_key = make_cache_key(
                "verification", attribution_key, self.config,
                extra={"truth": content_hash(truth_entry)},
            )
            verification_product, _, verification_entry = self.cache.get_or_compute(
                verification_key,
                lambda: verify_audio(truth_entry, attribution_product, segment_product,
                                     self.config, self.registry, audio,
                                     frame_lookup, feature_product["n_frames"]),
                audio, "verification",
            )
            self._track_ledger(verification_entry)
        except PipelineError as error:
            if error.injection_id is None and audio_injection is not None:
                error.injection_id = audio_injection["injection_id"]
            self._fail(audio, error)
            return None

        return {
            "audio": audio,
            "keys": {
                "audio": audio_key,
                "features": feature_key,
                "segments": segment_key,
                "clusters": cluster_key,
                "attribution": attribution_key,
                "verification": verification_key,
            },
            "features": {
                "n_frames": feature_product["n_frames"],
                "quant_step": feature_product["quant_step"],
                "frac_bits": feature_product["frac_bits"],
            },
            "segments": [self._public_segment(seg) for seg in segment_product["segments"]],
            "boundary_records": segment_product["boundary_records"],
            "clusters": attribution_product["clusters"],
            "adjudications": attribution_product["adjudications"],
            "speaker_intervals": attribution_product["speaker_intervals"],
            "verification": verification_product,
        }

    @staticmethod
    def _public_segment(segment):
        return {key: segment[key] for key in (
            "segment_index", "start_frame", "end_frame", "start_ms", "end_ms",
            "overlap_candidate", "mean_cepstrum_q", "frame_set", "quant_step",
            "frac_bits",
        )}

    def run(self):
        audio_names = sorted(name for name in os.listdir(self.input_dir)
                             if name.lower().endswith(".wav"))
        catalog = {}
        raw_map = {}
        for name in audio_names:
            with open(os.path.join(self.input_dir, name), "rb") as fh:
                raw = fh.read()
            raw_map[name] = raw
            catalog[name] = {"size": len(raw), "sha256": content_hash(_bytes(raw))}

        config_hash = self.config.fingerprint()
        injection_hash = content_hash(self.registry.items)
        if self.fresh and self.snapshot.exists():
            self.snapshot.reset()
        manifest = self.snapshot.assert_resumable(
            config_hash, injection_hash, catalog, allow_edit=self.allow_edit,
        )
        if manifest is None:
            manifest = self.snapshot.initial_manifest(config_hash, injection_hash, catalog)

        # 所有音频都重新走缓存判定：未失效产物命中 reuse，失效或新增产物才 compute。
        # 已完成音频的历史结果索引仅用于快照恢复，不再跳过缓存检查。
        pending = list(audio_names)
        max_workers = max(1, int(self.config.max_workers))
        # 任务携带由文件名派生的稳定排序键；帧/区间序号在各阶段内部
        # 按 frame_index / segment_index 升序处理（见 features、segments、clusters）
        task_keys = {name: ("audio", name, 0, 0) for name in pending}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._process_audio, name, raw_map[name]): name
                       for name in pending}
            for future in sorted(futures, key=lambda fut: task_keys[futures[fut]]):
                audio = futures[future]
                result = future.result()
                if result is not None:
                    self.results[audio] = result
                    manifest["result_index"][audio] = {"keys": result["keys"]}
                manifest["completed_audio_cursor"] = sorted(
                    set(manifest["completed_audio_cursor"]) | {audio}
                )
                if len(manifest["completed_audio_cursor"]) % max(
                        1, self.config.snapshot_interval) == 0:
                    manifest["global_version"] += 1
                    self.snapshot.save(manifest)

        # 失败清单顺序与单线程按文件名顺序执行保持一致，与并发完成顺序无关
        self.failures.sort(key=lambda record: (record["audio"], record["stage"]))
        manifest["config_hash"] = config_hash
        manifest["injection_hash"] = injection_hash
        manifest["audio_catalog"] = catalog
        manifest["failure_index"] = self.failures
        manifest["completed_audio_cursor"] = sorted(
            set(manifest["completed_audio_cursor"]) | set(audio_names)
        )
        manifest["ledger"] = self.ledgers
        manifest["global_version"] += 1
        self.snapshot.save(manifest)
        return self._report(manifest)

    def _assemble_cached(self, audio, keys):
        feature_product = self.cache.get(keys["features"])
        segment_product = self.cache.get(keys["segments"])
        attribution_product = self.cache.get(keys["attribution"])
        verification_product = self.cache.get(keys["verification"])
        frame_lookup = {f["frame_index"]: {"start_ms": f["start_ms"], "end_ms": f["end_ms"]}
                        for f in feature_product["frames"]}
        return {
            "audio": audio,
            "keys": keys,
            "features": {"n_frames": feature_product["n_frames"],
                         "quant_step": feature_product["quant_step"],
                         "frac_bits": feature_product["frac_bits"]},
            "segments": [self._public_segment(seg) for seg in segment_product["segments"]],
            "boundary_records": segment_product["boundary_records"],
            "clusters": attribution_product["clusters"],
            "adjudications": attribution_product["adjudications"],
            "speaker_intervals": attribution_product["speaker_intervals"],
            "verification": verification_product,
            "frame_lookup": frame_lookup,
        }

    def _report(self, manifest):
        ordered_audio = sorted(self.results)
        return {
            "audio_order": ordered_audio,
            "results": {name: self.results[name] for name in ordered_audio},
            "failures": self.failures,
            "stats": self.stats,
            "ledger": self.ledgers,
            "snapshot": {
                "global_version": manifest["global_version"],
                "config_hash": manifest["config_hash"],
                "injection_hash": manifest["injection_hash"],
                "completed_audio_cursor": manifest["completed_audio_cursor"],
                "stage_versions": manifest["stage_versions"],
            },
        }

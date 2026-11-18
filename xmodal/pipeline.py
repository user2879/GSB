"""管线编排：加载校验 -> 特征抽取 -> 对齐打分 -> 格式转换 -> 检索。

特性：
  * 线程池并发，任务按 (样本名, 阶段名) 派生稳定排序键；
  * 缓存键驱动精确失效，未失效产物直接复用且不执行对应计算；
  * 阶段临时目录整体替换提交，支持信号优雅退出与强制终止后恢复；
  * 单样本异常隔离，失败样本进入失败清单且不影响其余样本。
"""

import json
import os
import shutil
import time

import numpy as np

from . import alignment as align_mod
from . import convert as convert_mod
from . import injection as inj_mod
from .cache import CachePlanner
from .config import config_fingerprint, load_config, spec_fingerprint
from .errors import ErrorCode, PipelineError, wrap_exception
from .features import (FeatureRecord, extract_sample, list_samples,
                       raw_input_fingerprints)
from .formats import (V1_VERSION, V2_VERSION, pack_fingerprint, read_pack,
                      read_v1, read_v2, write_pack)
from .hashing import combine, digest_json, stable_sort_key
from .report import build_report
from .snapshot import (ShutdownRequested, SignalGate, SnapshotStore,
                       new_manifest)

PACK_NAME = {V1_VERSION: "feature.xmf1", V2_VERSION: "feature.json"}


class Pipeline:
    def __init__(self, input_dir, work_dir, config_path, spec_path,
                 injection_path=None, format_version=None,
                 max_workers_override=None):
        self.input_dir = input_dir
        self.work_dir = work_dir
        self.cfg = load_config(config_path)
        if max_workers_override is not None:
            self.cfg = dict(self.cfg)
            self.cfg["max_workers"] = max_workers_override
        with open(spec_path, "rb") as fh:
            self.spec = json.loads(fh.read().decode("utf-8"))
        self.format_version = format_version or self.cfg.get(
            "format_version", V2_VERSION)
        if self.format_version not in (V1_VERSION, V2_VERSION):
            raise PipelineError(ErrorCode.SAMPLE_INVALID,
                                "格式版本未知: %r" % self.format_version)
        self.injections = inj_mod.load_injection_config(
            injection_path).get("injections", [])
        self.store = SnapshotStore(work_dir)
        self.planner = CachePlanner(
            self.cfg, self.spec, self.format_version, input_dir,
            self.cfg["image_size"])
        self.gate = SignalGate()
        self.failures = []
        self.injection_hits = []
        self.reuse_stats = {}
        self.stage_versions = {}
        self.shutdown_info = None

    # ------------------------------------------------------------------
    def _sorted_tasks(self, samples, stage):
        return sorted(samples, key=lambda s: stable_sort_key(s, stage))

    # ------------------------------------------------------------------
    def run(self):
        self.gate.install()
        removed_tmp = self.store.cleanup_tmp()
        previous = self.store.load_manifest()
        run_id = combine([
            config_fingerprint(self.cfg), spec_fingerprint(self.spec),
            str(self.format_version),
            digest_json(sorted(os.listdir(self.input_dir)), prefix="inputs"),
        ], prefix="run")[:20]
        if previous is None or previous.get("config_hash") != \
                config_fingerprint(self.cfg) or \
                previous.get("spec_hash") != spec_fingerprint(self.spec):
            manifest = new_manifest(
                config_fingerprint(self.cfg),
                spec_fingerprint(self.spec), run_id)
        else:
            manifest = previous
        manifest["run_id"] = run_id
        manifest["format_version"] = self.format_version
        self.manifest = manifest
        recovery = self.store.recovery_point()
        recovery["cleaned_tmp"] = removed_tmp

        all_samples = list_samples(self.input_dir)
        plan, current_keys = self.planner.plan(all_samples, previous)
        # 强制终止只可能留下完整提交的阶段目录；未提交阶段即使清单中残留
        # 键，也必须按重算处理，避免引用不存在的复用产物。
        committed_stages = set(recovery["committed_stages"])
        for stage in ("validate", "extract", "align", "convert"):
            if stage not in committed_stages:
                plan[stage]["recompute"] = sorted(
                    set(plan[stage]["recompute"])
                    | set(plan[stage]["reuse"]))
                plan[stage]["reuse"] = []
        if "retrieve" not in committed_stages:
            plan["retrieve"] = {"recompute": ["__index__"], "reuse": []}
        self.plan = plan
        manifest["keys"] = current_keys
        self.reuse_stats = {
            stage: {"recompute": len(plan[stage]["recompute"]),
                    "reuse": len(plan[stage]["reuse"])}
            for stage in plan
        }
        try:
            valid = self._stage_validate(plan, manifest)
            valid = self._stage_extract(plan, manifest, valid)
            self._stage_align(plan, manifest, valid)
            self._stage_convert(plan, manifest, valid)
            self._stage_retrieve(plan, manifest, valid)
        except ShutdownRequested as exc:
            self.shutdown_info = {
                "signal": exc.args[0] if exc.args else "SIGTERM",
                "exit_code": 0,
                "recovery_point": self.store.recovery_point(),
            }
            manifest["shutdown"] = self.shutdown_info
            self.store.write_manifest(manifest)
            self.gate.restore()
            return self._emit_report(recovery, interrupted=True)
        manifest["global_version"] = int(manifest.get("global_version", 0)) \
            + 1
        manifest.pop("shutdown", None)
        self.store.write_manifest(manifest)
        self.gate.restore()
        return self._emit_report(recovery, interrupted=False)

    # ------------------------------------------------------------------
    def _stage_validate(self, plan, manifest):
        stage = "validate"
        tmp = self.store.stage_tmp(stage)
        recompute = plan[stage]["recompute"]
        prior_valid = list(manifest.get("valid_samples", []))

        def task(sample):
            sample_tmp = os.path.join(tmp, sample)
            os.makedirs(sample_tmp, exist_ok=True)
            png = os.path.join(self.input_dir, sample, "image.png")
            text = os.path.join(self.input_dir, sample, "text.json")
            png_copy = os.path.join(sample_tmp, "image.png")
            text_copy = os.path.join(sample_tmp, "text.json")
            shutil.copyfile(png, png_copy)
            shutil.copyfile(text, text_copy)
            ids = []
            ids += inj_mod.apply_file_faults(
                self.injections, sample, stage, png_copy)
            ids += inj_mod.apply_file_faults(
                self.injections, sample, stage, text_copy)
            from .features import load_sample_files
            image, text_obj, regions = load_sample_files(
                self.input_dir, sample, self.cfg["image_size"],
                require_regions=False, base_dir=sample_tmp)
            with open(os.path.join(sample_tmp, "ok.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"sample": sample, "status": "ok"}, fh,
                          sort_keys=True)
            for inj_id in ids:
                self.injection_hits.append({
                    "id": inj_id, "type": next(
                        (i["type"] for i in self.injections
                         if i["id"] == inj_id), "unknown"),
                    "stage": stage, "sample": sample, "code": None})
            return sample

        valid_map = self._run_isolated(stage, recompute, task)
        valid_now = list(valid_map.keys())
        committed = self.store.stage_dir(stage)
        reused = []
        for sample in plan[stage]["reuse"]:
            if sample in prior_valid and \
                    os.path.isfile(os.path.join(committed, sample,
                                                "ok.json")):
                valid_now.append(sample)
                reused.append(sample)
        valid_now = sorted(set(valid_now))
        self._commit(stage, manifest, valid_now, reused)
        return valid_now

    def _stage_extract(self, plan, manifest, valid):
        stage = "extract"
        tmp = self.store.stage_tmp(stage)
        recompute = [s for s in plan[stage]["recompute"] if s in valid]
        records = {}

        def task(sample):
            record = extract_sample(
                self.input_dir, sample, self.cfg, source_version="raw")
            out = os.path.join(tmp, sample + ".json")
            payload = {
                "sample": record.sample,
                "source_version": record.source_version,
                "align_weights": record.align_weights,
                "image_blocks": record.image_blocks.astype(float).tolist(),
                "text_vector": record.text_vector.astype(float).tolist(),
                "region_truth": record.region_truth,
                "precision_labels": record.precision_labels,
            }
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, sort_keys=True,
                          separators=(",", ":"))
            return record

        done = self._run_isolated(stage, recompute, task)
        records.update(done)
        committed = self.store.stage_dir(stage)
        reused = []
        for sample in plan[stage]["reuse"]:
            if sample not in valid:
                continue
            path = os.path.join(committed, sample + ".json")
            if not os.path.isfile(path):
                raise PipelineError(ErrorCode.SAMPLE_INVALID,
                                    "复用产物缺失: " + path)
            records[sample] = self._load_extract_record(path)
            reused.append(sample)
        self._commit(stage, manifest, valid, reused)
        self.records = records
        return valid

    def _load_extract_record(self, path):
        with open(path, "rb") as fh:
            payload = json.loads(fh.read().decode("utf-8"))
        return FeatureRecord(
            sample=payload["sample"],
            image_blocks=np.array(payload["image_blocks"], dtype=np.float32),
            text_vector=np.array(payload["text_vector"], dtype=np.float32),
            align_weights=payload["align_weights"],
            source_version=payload["source_version"],
            precision_labels=payload.get("precision_labels", {}),
            region_truth=payload.get("region_truth"),
        )

    # ------------------------------------------------------------------
    def _stage_align(self, plan, manifest, valid):
        stage = "align"
        tmp = self.store.stage_tmp(stage)
        recompute = [s for s in plan[stage]["recompute"] if s in valid]
        scores = {}
        details = {}

        def task(sample):
            record = self.records[sample]
            score, detail = align_mod.alignment_score(
                record, self.cfg["orientation_bins"])
            with open(os.path.join(tmp, sample + ".json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"sample": sample, "score": score,
                           "components": detail}, fh, sort_keys=True)
            return score, detail

        done = self._run_isolated(stage, recompute, task)
        for sample, (score, detail) in done.items():
            scores[sample] = score
            details[sample] = detail
        committed = self.store.stage_dir(stage)
        reused = []
        for sample in plan[stage]["reuse"]:
            if sample not in valid:
                continue
            with open(os.path.join(committed, sample + ".json"),
                      "rb") as fh:
                payload = json.loads(fh.read().decode("utf-8"))
            scores[sample] = payload["score"]
            details[sample] = payload["components"]
            reused.append(sample)
        self.scores = scores
        self.score_details = details
        self._commit(stage, manifest, valid, reused)

    def _stage_convert(self, plan, manifest, valid):
        stage = "convert"
        tmp = self.store.stage_tmp(stage)
        recompute = [s for s in plan[stage]["recompute"] if s in valid]
        pack_name = PACK_NAME[self.format_version]
        deviations = []
        roundtrip = {}

        def task(sample):
            record = self.records[sample]
            record = FeatureRecord(
                record.sample, record.image_blocks, record.text_vector,
                record.align_weights, self.format_version,
                precision_labels=dict(record.precision_labels),
                region_truth=record.region_truth)
            sample_tmp = os.path.join(tmp, sample)
            os.makedirs(sample_tmp, exist_ok=True)
            pack_path = os.path.join(sample_tmp, pack_name)
            write_pack(pack_path, record, self.format_version)
            ids = inj_mod.apply_file_faults(
                self.injections, sample, stage, pack_path)
            reread = read_pack(pack_path, expected_sample=sample)
            for inj_id in ids:
                self.injection_hits.append({
                    "id": inj_id, "type": next(
                        (i["type"] for i in self.injections
                         if i["id"] == inj_id), "unknown"),
                    "stage": stage, "sample": sample,
                    "code": ErrorCode.SAMPLE_INVALID})
            roundtrip[sample] = {
                "pack_fingerprint": pack_fingerprint(pack_path),
                "image_blocks": reread.image_blocks.astype(float).tolist(),
                "text_vector": reread.text_vector.astype(float).tolist(),
                "align_weights": reread.align_weights,
                "source_version": reread.source_version,
            }
            if self.format_version == V1_VERSION:
                sidecar = convert_mod.convert_v1_to_v2(
                    pack_path, os.path.join(sample_tmp, "mirror.json"))
            else:
                v1_path = os.path.join(sample_tmp, "mirror.xmf1")
                sidecar = convert_mod.convert_v2_to_v1(pack_path, v1_path)
                mirror = read_v1(v1_path)
                deviations.append({
                    "sample": sample,
                    "image_max_abs": round(float(np.max(np.abs(
                        record.image_blocks.astype(np.float64)
                        - mirror.image_blocks.astype(np.float64)))), 12),
                    "text_max_abs": round(float(np.max(np.abs(
                        record.text_vector.astype(np.float64)
                        - mirror.text_vector.astype(np.float64)))), 12),
                })
            return sidecar

        done = self._run_isolated(stage, recompute, task)
        self.sidecars = done
        self.deviations = deviations
        committed = self.store.stage_dir(stage)
        reused = []
        for sample in plan[stage]["reuse"]:
            if sample not in valid:
                continue
            reused.append(sample)
            sample_dir = os.path.join(committed, sample)
            pack_path = os.path.join(sample_dir, pack_name)
            if not os.path.isfile(pack_path):
                raise PipelineError(ErrorCode.SAMPLE_INVALID,
                                    "复用产物缺失: " + pack_path)
            reread = read_pack(pack_path, expected_sample=sample)
            roundtrip[sample] = {
                "pack_fingerprint": pack_fingerprint(pack_path),
                "image_blocks": reread.image_blocks.astype(float).tolist(),
                "text_vector": reread.text_vector.astype(float).tolist(),
                "align_weights": reread.align_weights,
                "source_version": reread.source_version,
            }
            sidecar_path = convert_mod.sidecar_path(
                os.path.join(sample_dir,
                             "mirror.json" if self.format_version == V1_VERSION
                             else "mirror.xmf1"))
            if os.path.isfile(sidecar_path):
                with open(sidecar_path, "rb") as fh:
                    self.sidecars[sample] = json.loads(fh.read().decode(
                        "utf-8"))
            if self.format_version == V2_VERSION:
                mirror = read_v1(os.path.join(sample_dir, "mirror.xmf1"))
                record = self.records[sample]
                deviations.append({
                    "sample": sample,
                    "image_max_abs": round(float(np.max(np.abs(
                        record.image_blocks.astype(np.float64)
                        - mirror.image_blocks.astype(np.float64)))), 12),
                    "text_max_abs": round(float(np.max(np.abs(
                        record.text_vector.astype(np.float64)
                        - mirror.text_vector.astype(np.float64)))), 12),
                })
        self.roundtrip = roundtrip
        self.deviations = sorted(deviations, key=lambda d: d["sample"])
        self._commit(stage, manifest, valid, reused)

    def _stage_retrieve(self, plan, manifest, valid):
        stage = "retrieve"
        tmp = self.store.stage_tmp(stage)
        boundary = inj_mod.boundary_threshold_item(self.injections)
        threshold = float(self.cfg["retrieval_threshold"])
        if boundary:
            scores = sorted(self.scores.values())
            # 边界规则：阈值取次低分，使最低分查询严格低于阈值并触发
            # E_RETRIEVAL_LOW_SCORE，而边界样本恰好等于阈值（可审计）。
            rank = int(boundary.get("rank", 1))
            rank = max(1, min(rank, len(scores) - 1))
            threshold = float(scores[rank])
            self.injection_hits.append({
                "id": boundary["id"], "type": boundary["type"],
                "stage": stage, "sample": None,
                "code": ErrorCode.RETRIEVAL_LOW_SCORE})
        score_map = {s: self.scores[s] for s in sorted(self.scores)}
        results = []
        for query_sample in sorted(score_map):
            result = align_mod.search(
                score_map, self.cfg["top_k"], threshold,
                query_sample=query_sample)
            result["query"] = query_sample
            results.append(result)
        with open(os.path.join(tmp, "retrieval.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"threshold": threshold, "results": results}, fh,
                      sort_keys=True, separators=(",", ":"))
        self.retrieval_threshold = threshold
        self.retrieval = results
        self._commit(stage, manifest, valid, [])

    # ------------------------------------------------------------------
    def _record_failure(self, sample, stage, exc, injection_ids):
        if isinstance(exc, PipelineError):
            err = exc
        else:
            err = wrap_exception(ErrorCode.SAMPLE_INVALID, exc)
        self.failures.append(err.to_record(sample, stage))
        for inj_id in injection_ids:
            self.injection_hits.append({
                "id": inj_id, "type": next(
                    (i["type"] for i in self.injections
                     if i["id"] == inj_id), "unknown"),
                "stage": stage, "sample": sample, "code": err.code})

    # ------------------------------------------------------------------
    def _run_isolated(self, stage, samples, func):
        """并发执行样本任务，单样本异常隔离并记录失败清单。"""
        done = {}
        interval = max(1, int(self.cfg["snapshot_commit_interval"]))
        processed = 0

        def guarded(sample):
            return sample, func(sample)

        ordered = self._sorted_tasks(samples, stage)
        workers = max(1, int(self.cfg["max_workers"]))
        delay = inj_mod.delay_item(self.injections, stage)
        if delay:
            time.sleep(float(delay.get("seconds", 0.5)))
            self.injection_hits.append({
                "id": delay["id"], "type": delay["type"],
                "stage": stage, "sample": None, "code": None})
            self.gate.checkpoint()
        if workers == 1 or len(ordered) <= 1:
            iterator = ((s, guarded) for s in ordered)
            for sample, _ in iterator:
                self.gate.checkpoint()
                try:
                    name, value = guarded(sample)
                    done[name] = value
                except Exception as exc:  # noqa: BLE001 - 隔离边界
                    self._record_failure(sample, stage, exc,
                                         self._active_injection_ids(
                                             sample, stage))
                processed += 1
                if processed % interval == 0:
                    self.store.write_manifest(self.manifest)
        else:
            from concurrent.futures import ThreadPoolExecutor as Pool
            with Pool(max_workers=workers) as pool:
                future_map = {pool.submit(guarded, s): s for s in ordered}
                for future in future_map:
                    self.gate.checkpoint()
                    sample = future_map[future]
                    try:
                        name, value = future.result()
                        done[name] = value
                    except Exception as exc:  # noqa: BLE001
                        self._record_failure(sample, stage, exc,
                                             self._active_injection_ids(
                                                 sample, stage))
                    processed += 1
                    if processed % interval == 0:
                        self.store.write_manifest(self.manifest)
        return done

    def _active_injection_ids(self, sample, stage):
        return [item["id"] for item in inj_mod.select(
            self.injections, sample, stage)
            if item["type"] != "stage_delay"]

    def _commit(self, stage, manifest, valid_samples, reuse_samples=None):
        manifest["stage_versions"][stage] = combine(
            [str(stage), manifest.get("run_id", ""),
             digest_json(sorted(valid_samples), prefix=stage)],
            prefix="stage-ver")[:20]
        manifest["valid_samples"] = sorted(set(valid_samples))
        manifest["completed_samples"] = sorted(
            set(manifest.get("completed_samples", []))
            | set(valid_samples))
        reuse_samples = reuse_samples or []
        committed = self.store.stage_dir(stage)
        tmp = self.store.stage_tmp(stage)
        if os.path.isdir(committed):
            reuse_names = set(reuse_samples)
            reuse_files = {name + ".json" for name in reuse_samples}
            for name in os.listdir(committed):
                if name in reuse_names or name in reuse_files:
                    src = os.path.join(committed, name)
                    dst = os.path.join(tmp, name)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copyfile(src, dst)
        self.store.commit_stage(stage)
        self.store.write_manifest(manifest)
        self.gate.checkpoint()
        self.stage_versions[stage] = manifest["stage_versions"][stage]

    def _emit_report(self, recovery, interrupted):
        report = build_report(
            cfg=self.cfg,
            spec_hash=spec_fingerprint(self.spec),
            config_hash=config_fingerprint(self.cfg),
            format_version=self.format_version,
            manifest=self.manifest,
            plan=self.plan,
            reuse_stats=self.reuse_stats,
            failures=self.failures,
            injection_hits=self.injection_hits,
            injections=self.injections,
            scores=getattr(self, "scores", {}),
            score_details=getattr(self, "score_details", {}),
            retrieval=getattr(self, "retrieval", []),
            threshold=getattr(self, "retrieval_threshold",
                              self.cfg["retrieval_threshold"]),
            deviations=getattr(self, "deviations", []),
            sidecars=getattr(self, "sidecars", {}),
            roundtrip=getattr(self, "roundtrip", {}),
            recovery=recovery,
            interrupted=interrupted,
            shutdown_info=self.shutdown_info,
        )
        report_path = os.path.join(self.work_dir, "report.json")
        tmp = report_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, sort_keys=True, ensure_ascii=False,
                      indent=2, allow_nan=False)
        os.replace(tmp, report_path)
        return report

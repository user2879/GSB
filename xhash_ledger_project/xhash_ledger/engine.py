"""跨模态哈希签名与近重复簇账本的增量持久化引擎。

同一份引擎代码同时服务两种运行方式：

* 增量运行：在已有快照上逐条应用编辑命令，按缓存键命中情况精确复用；
* 全量重建（``rebuild=True``）：忽略旧缓存，从空白状态对最终样本集合与配置
  执行同一套纯函数流水线。

两条路径共享特征/量化/签名/分桶/聚类的全部确定性函数，因此逐字段一致；
增量路径额外产出逐条失效复用清单，全量路径的清单全部标记为 recomputed。
"""
from __future__ import annotations

import os
import shutil
from typing import Any

from . import cache as cache_mod
from . import features as features_mod
from .clustering import build_cluster_ledger, clusters_to_dict, reassign_cluster_ids
from .config import Config, parse_command
from .errors import E_CONFIG_INVALID, Failure, make_failure
from .features import compose_weighted, extract_raw_features
from .fixedpoint import quantize_vector
from .hashing import fingerprint
from .scheduler import plan_execution, run_thread_pool, total_cost
from .signature import Projection, build_projection, generate_signature, layout_for
from .state import SnapshotStore, read_json, write_json_atomic
from .validation import SampleBundle, load_sample

FIXED_SIZE = (64, 64)


class Engine:
    def __init__(self, workspace: str, rebuild: bool = False):
        self.workspace = os.path.abspath(workspace)
        self.sample_source = os.path.join(self.workspace, "samples")
        self.state_dir = os.path.join(self.workspace, "state")
        self.rebuild = rebuild
        self.store = SnapshotStore(self.state_dir)
        self.config = self._load_initial_config()
        self.bundles: dict[str, SampleBundle] = {}
        self.content_fps: dict[str, str] = {}
        self.signatures: dict[str, Any] = {}
        self.clusters: dict[str, Any] = {}
        self.reassign_records: list[dict[str, str]] = []
        self.reuse_log: list[dict[str, Any]] = []
        self.failures: list[Failure] = []
        self.pending: list[str] = []
        self.version = 0
        self.budget_consumed = 0
        self.feature_cache: dict[str, dict] = {}
        self.signature_cache: dict[str, dict] = {}
        self.bucket_cache: dict[str, dict] = {}
        self.cluster_cache: dict[str, dict] = {}
        self.projection: Projection | None = None
        self.layout: features_mod.FeatureLayout | None = None
        self._hydrate()

    # ------------------------------------------------------------------
    # 配置与状态初始化
    # ------------------------------------------------------------------
    def _load_initial_config(self) -> Config:
        raw = read_json(os.path.join(self.workspace, "config.json"), None)
        if raw is None:
            return Config()
        config, failures = self._parse_config(raw)
        if failures:
            raise ValueError(f"初始配置非法: {failures}")
        return config

    @staticmethod
    def _parse_config(raw: dict[str, Any]):
        from .config import parse_config
        return parse_config(raw)

    def _hydrate(self) -> None:
        """从快照恢复样本集合、缓存与清单；rebuild 模式从空白开始。"""
        if self.rebuild:
            return
        manifest = self.store.read_manifest()
        if manifest is None:
            return
        self.version = manifest["version"]
        self.budget_consumed = manifest["budget_consumed"]
        self.pending = list(manifest.get("pending", []))
        self.config, _ = self._parse_config(manifest["config"])
        for sid in manifest.get("completed_samples", []):
            bundle, failures = load_sample(os.path.join(self.state_dir, "samples", sid), sid, FIXED_SIZE)
            if failures:
                self.failures.extend(failures)
                continue
            self.bundles[sid] = bundle
        for sid in list(self.bundles):
            fp_doc = read_json(os.path.join(self.state_dir, "samples", sid, "content.json"), {})
            self.content_fps[sid] = fp_doc.get("content_fingerprint", "")
        self.feature_cache = self._load_cache_dir("feature")
        self.signature_cache = self._load_cache_dir("signature")
        self.bucket_cache = self._load_cache_dir("bucket")
        # 恢复签名内存视图，保证续跑后删除/聚类路径无悬空成员
        for sid, sig_doc in self.signature_cache.items():
            if sid not in self.bundles:
                continue
            bdoc = self.bucket_cache.get(sid)
            if bdoc:
                self.signatures[sid] = _SignatureView(sig_doc, bdoc)
        cluster_doc = read_json(
            os.path.join(self.state_dir, "cache", "cluster", "ledger.json"), {}
        )
        self.cluster_cache = cluster_doc if isinstance(cluster_doc, dict) and "ledger" in cluster_doc else {}
        if self.cluster_cache:
            self.clusters = self._clusters_from_cache()

    def _load_cache_dir(self, stage: str) -> dict[str, dict]:
        directory = os.path.join(self.state_dir, "cache", stage)
        result = {}
        if not os.path.isdir(directory):
            return result
        for name in sorted(os.listdir(directory)):
            if name.endswith(".json"):
                result[name[:-5]] = read_json(os.path.join(directory, name), {})
        return result

    # ------------------------------------------------------------------
    # 样本内容存储（复制进 state，保证自包含可恢复）
    # ------------------------------------------------------------------
    def _install_sample(self, sid: str) -> SampleBundle | None:
        """从样本源目录校验并安装单个样本到状态目录。"""
        source_dir = self.sample_source
        replacement_dir = os.path.join(self.workspace, "replacements")
        if os.path.exists(os.path.join(replacement_dir, f"{sid}.png")):
            source_dir = replacement_dir
        bundle, failures = load_sample(source_dir, sid, FIXED_SIZE)
        self.failures.extend(failures)
        if bundle is None:
            return None
        target = os.path.join(self.state_dir, "samples", sid)
        if os.path.exists(target):
            shutil.rmtree(target)
        os.makedirs(target, exist_ok=True)
        for src in (bundle.png_path, bundle.assertion_path, bundle.regions_path):
            shutil.copy2(src, os.path.join(target, os.path.basename(src)))
        content_fp = cache_mod.sample_content_fingerprint(
            {
                "png": bundle.png_path,
                "assertion": bundle.assertion_path,
                "regions": bundle.regions_path,
            }
        )
        write_json_atomic(os.path.join(target, "content.json"), {"content_fingerprint": content_fp})
        self.bundles[sid] = bundle
        self.content_fps[sid] = content_fp
        if source_dir != self.sample_source:
            for suffix in (".png", ".assertion.json", ".regions.json"):
                staged = os.path.join(source_dir, f"{sid}{suffix}")
                if os.path.exists(staged):
                    os.remove(staged)
        return bundle

    def _delete_sample_files(self, sid: str) -> None:
        target = os.path.join(self.state_dir, "samples", sid)
        if os.path.exists(target):
            shutil.rmtree(target)
        self.bundles.pop(sid, None)
        self.content_fps.pop(sid, None)
        self.feature_cache.pop(sid, None)
        self.signature_cache.pop(sid, None)
        self.bucket_cache.pop(sid, None)

    # ------------------------------------------------------------------
    # 命令应用（仅修改样本集合/配置，计算在 run_pipeline 中统一完成）
    # ------------------------------------------------------------------
    def apply_command(self, command: dict[str, Any]) -> bool:
        failures = parse_command(command, self.config.declared_seed_versions)
        if failures:
            self.failures.extend(failures)
            return False
        cmd = command["cmd"]
        sid = command.get("sample_id")
        if cmd == "add":
            if sid in self.bundles:
                self.failures.append(
                    make_failure(E_CONFIG_INVALID, "command", sid, "cmd", f"样本 {sid} 已存在，add 冲突")
                )
                return False
            return self._install_sample(sid) is not None
        if cmd == "replace":
            if sid not in self.bundles:
                self.failures.append(
                    make_failure(E_CONFIG_INVALID, "command", sid, "cmd", f"样本 {sid} 不存在，无法 replace")
                )
                return False
            old_fp = self.content_fps.get(sid)
            installed = self._install_sample(sid)
            if installed is not None:
                # 替换即内容指纹变化：旧三级缓存键必然失配，显式清除以触发重算
                self.feature_cache.pop(sid, None)
                self.signature_cache.pop(sid, None)
                self.bucket_cache.pop(sid, None)
                self.signatures.pop(sid, None)
                self._invalidate_sample(sid, old_fp)
            return installed is not None
        if cmd == "delete":
            if sid not in self.bundles:
                self.failures.append(
                    make_failure(E_CONFIG_INVALID, "command", sid, "cmd", f"样本 {sid} 不存在，无法 delete")
                )
                return False
            old_clusters = self._clusters_from_cache()
            self._delete_sample_files(sid)
            self.signatures.pop(sid, None)
            self.pending = [item for item in self.pending if item != sid]
            new_clusters = self._recompute_clusters_lite(old_clusters, {sid})
            self.reassign_records.extend(
                reassign_cluster_ids(old_clusters, new_clusters, {sid})
            )
            self.clusters = new_clusters
            return True
        if cmd == "set_config":
            new_config, cfg_failures = self._parse_config(command["config"])
            if cfg_failures:
                self.failures.extend(cfg_failures)
                return False
            self._apply_config_change(new_config)
            return True
        return False

    def _invalidate_sample(self, sid: str, old_fp: str | None) -> None:
        """样本替换后按内容指纹差异标记分量级失效（缓存键变化即自然失效，
        这里额外记录受影响分量编号到复用清单）。"""
        if old_fp is None or old_fp == self.content_fps.get(sid):
            return
        # 图像与断言文件都可能变化，统一按全分量保守标注；缓存键本身仍精确区分
        self._pending_invalidation = getattr(self, "_pending_invalidation", {})
        self._pending_invalidation[sid] = "sample_replaced"

    def _apply_config_change(self, new_config: Config) -> None:
        old = self.config
        self.config = new_config
        # 分桶位数变化只失效 bucket/cluster；签名侧参数变化失效 signature 及下游
        if (old.signature_bits, old.seed_version, old.frac_bits, old.tile_size,
                old.image_weight, old.text_weight) != (
            new_config.signature_bits, new_config.seed_version, new_config.frac_bits,
            new_config.tile_size, new_config.image_weight, new_config.text_weight,
        ):
            self.signature_cache.clear()
            self.bucket_cache.clear()
        elif old.bucket_bits != new_config.bucket_bits:
            self.bucket_cache.clear()
        if (old.frac_bits, old.tile_size, old.image_weight, old.text_weight) != (
            new_config.frac_bits, new_config.tile_size, new_config.image_weight, new_config.text_weight,
        ):
            self.feature_cache.clear()
        self.cluster_cache = {}

    # ------------------------------------------------------------------
    # 簇账本的缓存重建辅助（删除命令路径使用）
    # ------------------------------------------------------------------
    def _clusters_from_cache(self):
        from .clustering import Cluster, MemberRecord, MergeRecord

        clusters: dict[str, Cluster] = {}
        ledger = {}
        if isinstance(self.cluster_cache, dict):
            inner = self.cluster_cache.get("ledger", self.cluster_cache)
            if isinstance(inner, dict):
                ledger = inner
        for cid, doc in ledger.items():
            members = [MemberRecord(**m) for m in doc.get("members", [])]
            logs = [MergeRecord(**m) for m in doc.get("merge_log", [])]
            clusters[cid] = Cluster(cluster_id=doc.get("cluster_id", cid), members=members, merge_log=logs)
        return clusters

    def _recompute_clusters_lite(self, old_clusters, deleted_ids: set[str]):
        """删除样本后基于剩余签名重算簇（纯函数），并清理悬空成员。"""
        surviving = {sid: sig for sid, sig in self.signatures.items() if sid not in deleted_ids}
        if not surviving:
            return {}
        return build_cluster_ledger(
            surviving, self.config.signature_bits, self.config.similarity_permille
        )

    # ------------------------------------------------------------------
    # 特征侧配置指纹与投影
    # ------------------------------------------------------------------
    def _feature_config_fingerprint(self) -> str:
        scope = {
            "frac_bits": self.config.frac_bits,
            "tile_size": self.config.tile_size,
            "image_weight": self.config.image_weight,
            "text_weight": self.config.text_weight,
        }
        return fingerprint(scope)

    def _ensure_projection(self) -> tuple[Projection, features_mod.FeatureLayout]:
        any_bundle = next(iter(self.bundles.values()))
        layout = layout_for(
            any_bundle,
            self.config.tile_size,
            self.config.image_weight,
            self.config.text_weight,
        )
        seed = self.config.projection_seed(layout.dim)
        projection = build_projection(
            seed, self.config.signature_bits, layout.dim, self.config.frac_bits
        )
        return projection, layout

    # ------------------------------------------------------------------
    # 单样本三级流水线（线程池 worker，必须保持纯函数式、无共享写）
    # ------------------------------------------------------------------
    def _process_sample(self, sid: str) -> dict[str, Any]:
        bundle = self.bundles[sid]
        content_fp = self.content_fps[sid]
        feat_key = cache_mod.feature_cache_key(content_fp, self._feature_config_fingerprint())
        sig_key = cache_mod.signature_cache_key(
            feat_key, self.config.seed_version, self.config.signature_bits
        )
        bkt_key = cache_mod.bucket_cache_key(sig_key, self.config.bucket_bits)
        result: dict[str, Any] = {"sample_id": sid}

        cached_feature = None if self.rebuild else self.feature_cache.get(sid)
        if cached_feature and cached_feature.get("cache_key") == feat_key:
            result["feature"] = cached_feature
            result["feature_hit"] = True
            quantization = self._quantization_from_doc(cached_feature)
            raw_components = cached_feature.get("affected_components", [])
        else:
            raw = extract_raw_features(bundle, self.layout)
            weighted = compose_weighted(raw, self.layout)
            quantization = quantize_vector(weighted, self.config.frac_bits)
            affected = self._invalidation_components(sid)
            feature_doc = {
                "sample_id": sid,
                "cache_key": feat_key,
                "upstream_content_fingerprint": content_fp,
                "quantization": quantization.to_dict(),
                "affected_components": affected,
            }
            result["feature"] = feature_doc
            result["feature_hit"] = False
            raw_components = affected
        result["feature_reason_affected"] = raw_components

        cached_signature = None if self.rebuild else self.signature_cache.get(sid)
        if cached_signature and cached_signature.get("cache_key") == sig_key:
            result["signature"] = cached_signature
            result["signature_hit"] = True
        else:
            sig = generate_signature(sid, quantization, self.projection, self.config.bucket_bits)
            signature_doc = sig.to_dict()
            signature_doc["cache_key"] = sig_key
            signature_doc["upstream_feature_key"] = feat_key
            signature_doc["referenced_components"] = sig.referenced_components
            result["signature"] = signature_doc
            result["signature_hit"] = False

        cached_bucket = None if self.rebuild else self.bucket_cache.get(sid)
        if cached_bucket and cached_bucket.get("cache_key") == bkt_key:
            result["bucket"] = cached_bucket
            result["bucket_hit"] = True
        else:
            signature_doc = result["signature"]
            result["bucket"] = {
                "sample_id": sid,
                "cache_key": bkt_key,
                "upstream_signature_key": sig_key,
                "bucket_keys": signature_doc["bucket_keys"],
                "bucket_bit_sets": signature_doc["bucket_bit_sets"],
            }
            result["bucket_hit"] = False
        return result

    @staticmethod
    def _quantization_from_doc(feature_doc: dict):
        from .fixedpoint import Quantization

        qdoc = feature_doc["quantization"]
        return Quantization(
            frac_bits=qdoc["frac_bits"],
            qstep_exp=qdoc["qstep_exp"],
            values=list(qdoc["values"]),
            residuals=list(qdoc["residuals"]),
        )

    def _invalidation_components(self, sid: str) -> list[int]:
        pending = getattr(self, "_pending_invalidation", {})
        if sid in pending:
            lo, hi = self.layout.image_component_range
            tlo, thi = self.layout.text_component_range
            return list(range(lo, hi)) + list(range(tlo, thi))
        return []

    # ------------------------------------------------------------------
    # 主流水线：调度 -> 并发执行 -> 簇账本 -> 快照提交
    # ------------------------------------------------------------------
    def run_pipeline(self, command: dict[str, Any] | None) -> dict[str, Any]:
        self.reuse_log = []
        self.failures = [f for f in self.failures]  # 保留命令校验失败
        if not self.bundles:
            self.signatures = {}
            self.clusters = {}
            return self._snapshot_and_report(command, plan=None, results=[])

        self.projection, self.layout = self._ensure_projection()
        # 只对缺少完整三级缓存的样本计费；已命中缓存的样本直接复用，不消耗点预算。
        def _needs_compute(sid: str) -> bool:
            if self.rebuild:
                return True
            fdoc, sdoc, bdoc = (self.feature_cache.get(sid),
                                self.signature_cache.get(sid),
                                self.bucket_cache.get(sid))
            return not (fdoc and sdoc and bdoc)

        candidate_ids = sorted(sid for sid in self.bundles if _needs_compute(sid))
        # global_budget 是每次运行重新发放的点预算；budget_consumed 仅累计已完成
        # 样本的历史点数，用于清单复核，不挤占本轮预算。
        budget_remaining = self.config.global_budget
        per_command_quota = self._quota_for_command(command)
        plan = plan_execution(
            candidate_ids=candidate_ids,
            pending_carry=[] if self.rebuild else [sid for sid in self.pending if sid in self.bundles],
            quota=per_command_quota,
            budget_remaining=max(0, budget_remaining),
            feature_dim=self.layout.dim,
            signature_bits=self.config.signature_bits,
            bucket_bits=self.config.bucket_bits,
        )
        results = run_thread_pool(plan.ready, self._process_sample, self.config.max_workers)
        self._record_cached_only(sid for sid in sorted(self.bundles) if sid not in plan.ready
                                 and sid not in plan.pending and sid not in plan.quota_failed)
        self._merge_results(results, plan)
        self._build_and_cache_clusters(plan)
        report = self._snapshot_and_report(command, plan, results)
        # 命令处理完毕，清理待失效标记
        self._pending_invalidation = {}
        return report

    def _quota_for_command(self, command: dict[str, Any] | None) -> int:
        """单样本配额：命令可通过顶层 per_sample_quota 临时收紧以演示配额中止；
        默认配额等于单样本满额成本，正常命令不会触发配额错误。"""
        if command and isinstance(command.get("per_sample_quota"), int):
            return int(command["per_sample_quota"])
        if self.layout is not None:
            return total_cost(
                self.layout.dim, self.config.signature_bits, self.config.bucket_bits
            )
        return 10**12

    def _record_cached_only(self, sids) -> None:
        """全缓存命中、本轮不耗点的样本，仍在复用清单中逐条登记命中理由。"""
        from . import cache as cache_mod
        for sid in sids:
            fdoc = self.feature_cache.get(sid)
            sdoc = self.signature_cache.get(sid)
            bdoc = self.bucket_cache.get(sid)
            if not (fdoc and sdoc and bdoc):
                continue
            reason = "缓存键命中：上游内容指纹与派生版本均未变化，直接复用（本轮零计算点）"
            self.reuse_log.append(cache_mod.reuse_entry("feature", sid, fdoc["cache_key"], True, reason))
            self.reuse_log.append(cache_mod.reuse_entry("signature", sid, sdoc["cache_key"], True, reason))
            self.reuse_log.append(cache_mod.reuse_entry("bucket", sid, bdoc["cache_key"], True, reason))
            # 重建内存签名视图，供聚类使用
            self.signatures[sid] = _SignatureView(sdoc, bdoc)

    def _merge_results(self, results: list[dict[str, Any]], plan) -> None:
        for doc in results:
            sid = doc["sample_id"]
            self.feature_cache[sid] = doc["feature"]
            self.signature_cache[sid] = doc["signature"]
            self.bucket_cache[sid] = doc["bucket"]
            self._append_reuse(sid, doc, plan)
            self.signatures[sid] = _SignatureView(doc["signature"], doc["bucket"])
        self.failures.extend(plan.failures)
        if not self.rebuild:
            self.pending = list(plan.pending)
        self.budget_consumed += plan.consumed

    def _append_reuse(self, sid: str, doc: dict[str, Any], plan) -> None:
        stages = [
            ("feature", doc["feature_hit"], doc["feature"]["cache_key"],
             doc.get("feature_reason_affected", [])),
            ("signature", doc["signature_hit"], doc["signature"]["cache_key"], []),
            ("bucket", doc["bucket_hit"], doc["bucket"]["cache_key"], []),
        ]
        for stage, hit, key, affected in stages:
            if hit:
                reason = "缓存键命中：上游内容指纹与派生版本均未变化，直接复用"
            else:
                reason = self._miss_reason(stage, sid)
            self.reuse_log.append(
                cache_mod.reuse_entry(stage, sid, key, hit, reason, affected)
            )

    def _miss_reason(self, stage: str, sid: str) -> str:
        pending = getattr(self, "_pending_invalidation", {})
        if sid in pending:
            return f"样本 {sid} 内容替换，上游内容指纹变化，受影响分量需重算"
        if stage == "signature":
            return "投影种子版本或签名位宽变化，投影矩阵重新派生，签名位重算"
        if stage == "bucket":
            return "分桶位数或上游签名变化，桶归属重新切分"
        return "特征侧配置或样本内容变化，量化特征重算"

    def _build_and_cache_clusters(self, plan) -> None:
        if plan is not None and plan.pending:
            # 有待办时簇账本仅覆盖已完成样本，并在报告中显式标注
            pass
        sig_views = {sid: self.signatures[sid] for sid in sorted(self.signatures)
                     if sid not in (plan.pending if plan else [])}
        self.clusters = build_cluster_ledger(
            sig_views, self.config.signature_bits, self.config.similarity_permille
        )
        bucket_map = {sid: view.bucket_keys for sid, view in sig_views.items()}
        cluster_key = cache_mod.cluster_cache_key(
            bucket_map, self.config.similarity_permille, self.config.signature_bits
        )
        self.cluster_cache = {
            "cache_key": cluster_key,
            "ledger": clusters_to_dict(self.clusters),
        }
        self.reuse_log.append(
            cache_mod.reuse_entry(
                "cluster",
                "ledger",
                cluster_key,
                False,
                "簇账本随桶归属/阈值确定性重建，并与上一版本快照核对复用依据",
            )
        )

    @property
    def _completed_ids(self) -> list[str]:
        return sorted(
            sid for sid in self.bundles
            if sid in self.feature_cache and sid in self.signature_cache
            and sid in self.bucket_cache and sid not in self.pending
        )

    # ------------------------------------------------------------------
    # 阶段快照整体替换与报告
    # ------------------------------------------------------------------
    def _snapshot_and_report(self, command, plan, results: list[dict[str, Any]]) -> dict[str, Any]:
        self.version += 1
        stage_payloads = {
            "feature": {sid: doc for sid, doc in self.feature_cache.items()},
            "signature": {sid: doc for sid, doc in self.signature_cache.items()},
            "bucket": {sid: doc for sid, doc in self.bucket_cache.items()},
            "cluster": self.cluster_cache,
        }
        for stage, payload in stage_payloads.items():
            stage_dir = self.store.stage_dir(self.version, stage)
            if stage == "cluster":
                write_json_atomic(os.path.join(stage_dir, "ledger.json"), payload)
            else:
                for sid, doc in payload.items():
                    write_json_atomic(os.path.join(stage_dir, f"{sid}.json"), doc)
            self.store.commit_stage(self.version, stage)
        self.store.cleanup_tmp(self.version)

        for sid, doc in self.feature_cache.items():
            write_json_atomic(os.path.join(self.state_dir, "cache", "feature", f"{sid}.json"), doc)
        for sid, doc in self.signature_cache.items():
            write_json_atomic(os.path.join(self.state_dir, "cache", "signature", f"{sid}.json"), doc)
        for sid, doc in self.bucket_cache.items():
            write_json_atomic(os.path.join(self.state_dir, "cache", "bucket", f"{sid}.json"), doc)
        write_json_atomic(
            os.path.join(self.state_dir, "cache", "cluster", "ledger.json"), self.cluster_cache
        )
        write_json_atomic(os.path.join(self.state_dir, "pending.json"), {"pending": self.pending})

        manifest = {
            "version": self.version,
            "config_hash": self.config.config_hash(),
            "config": self.config.to_dict(),
            "seed_version": self.config.seed_version,
            "completed_samples": self._completed_ids,
            "installed_samples": sorted(self.bundles),
            "budget_consumed": self.budget_consumed,
            "budget_remaining": max(0, self.config.global_budget - self.budget_consumed),
            "pending": list(self.pending),
        }
        self.store.write_manifest(manifest)
        return self._build_report(command, plan)

    def _build_report(self, command, plan) -> dict[str, Any]:
        return {
            "command": command,
            "manifest": {
                "version": self.version,
                "config_hash": self.config.config_hash(),
                "seed_version": self.config.seed_version,
                "completed_samples": self._completed_ids,
                "installed_samples": sorted(self.bundles),
                "budget_consumed": self.budget_consumed,
                "budget_remaining": max(0, self.config.global_budget - self.budget_consumed),
            },
            "signatures": {
                sid: {
                    "bits": doc["bits"],
                    "signature_int": doc["signature_int"],
                    "tie_breaks": doc["tie_breaks"],
                    "bucket_keys": doc["bucket_keys"],
                    "bucket_bit_sets": doc["bucket_bit_sets"],
                    "cache_key": self.signature_cache[sid]["cache_key"],
                    "quantization": doc["quantization"],
                }
                for sid, doc in sorted(self.signature_cache.items())
                if sid in self.bundles
            },
            "cluster_ledger": (self.cluster_cache or {}).get("ledger", {}),
            "cluster_reassignments": list(self.reassign_records),
            "reuse_log": list(self.reuse_log),
            "pending_queue": list(self.pending),
            "point_deductions": list(plan.point_deductions) if plan else [],
            "run_consumed": plan.consumed if plan else 0,
            "failures": [f.to_dict() for f in self.failures],
        }

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def execute_stream(self, commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for command in commands:
            self.failures = []
            applied = self.apply_command(command)
            if not applied:
                # 非法命令仍产出一份稳定报告，不改变账本状态
                reports.append(self._stable_failure_report(command))
                continue
            reports.append(self.run_pipeline(command))
        return reports

    def apply_final_config(self, commands: list[dict[str, Any]]) -> None:
        """回放命令流中的最后一次 set_config，得到最终配置（不触碰缓存）。"""
        for command in commands:
            if command.get("cmd") == "set_config":
                new_config, failures = self._parse_config(command["config"])
                if failures:
                    raise ValueError(f"全量重建遇到非法配置: {failures}")
                self.config = new_config

    def apply_batch(self, commands: list[dict[str, Any]]) -> dict[str, Any]:
        """在一次运行（一次预算发放）内应用一批样本编辑命令，再统一跑流水线。

        用于待办/预算场景：批内新增样本共享同一份全局点预算，
        预算不足的样本按稳定顺序进入待办队列。
        """
        self.failures = []
        for command in commands:
            self.apply_command(command)
        return self.run_pipeline({"cmd": "batch", "items": commands})

    def execute_rebuild(self, final_sample_ids: list[str]) -> dict[str, Any]:
        """全量重建路径：从空白对最终样本集合一次性计算。"""
        for sid in sorted(final_sample_ids):
            self._install_sample(sid)
        return self.run_pipeline(None)

    def _stable_failure_report(self, command) -> dict[str, Any]:
        return {
            "command": command,
            "manifest": {
                "version": self.version,
                "config_hash": self.config.config_hash(),
                "seed_version": self.config.seed_version,
                "completed_samples": self._completed_ids,
                "installed_samples": sorted(self.bundles),
                "budget_consumed": self.budget_consumed,
                "budget_remaining": max(0, self.config.global_budget - self.budget_consumed),
            },
            "signatures": {},
            "cluster_ledger": (self.cluster_cache or {}).get("ledger", {}),
            "cluster_reassignments": list(self.reassign_records),
            "reuse_log": [],
            "pending_queue": list(self.pending),
            "point_deductions": [],
            "run_consumed": 0,
            "failures": [f.to_dict() for f in self.failures],
        }


class _SignatureView:
    """聚类阶段所需的签名视图。"""

    def __init__(self, signature_doc: dict, bucket_doc: dict):
        self.signature_int = signature_doc["signature_int"]
        self.bits = signature_doc["bits"]
        self.bucket_keys = list(bucket_doc["bucket_keys"])

"""工作台引擎：加载、命令应用、阶段缓存流水线、快照与诊断报告。

同一份引擎同时支持两种执行路径，并保证逐字段一致：
- incremental：基于 Ledger 的精确失效 + 复用；
- full：从空白状态全量重建（用于等价性校验与验收）。
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from . import features as features_mod
from .alignment import run_query
from .concurrency import BoundedPool, Task, stable_task_key
from .config import Config, Command
from .errors import E_INTERACT_BUDGET_EXCEEDED, E_SNAPSHOT_STALE
from .fingerprint import digest_many, digest_obj
from .fixedpoint import QuantSpec
from .ledger import Ledger, LedgerEntry, make_cache_key
from .samples import SampleSet, scan_samples
from .snapshot import SnapshotStore, initial_manifest
from .strategy import StrategyDecision, select_strategy


@dataclass
class RuntimeParams:
    """可被命令流修改的运行期参数（含在参数指纹内）。"""

    weights: dict[str, float]
    channels_enabled: dict[str, bool]
    frac_bits: int
    tier: str
    active_tier: int

    def quant(self) -> QuantSpec:
        return QuantSpec(self.frac_bits, self.tier)

    def fingerprint(self) -> str:
        return digest_obj({
            "weights": self.weights,
            "channels_enabled": self.channels_enabled,
            "quant": [self.frac_bits, self.tier],
            "active_tier": self.active_tier,
        })


@dataclass
class StageArtifacts:
    visual: dict[str, Any] = field(default_factory=dict)
    text: dict[str, Any] = field(default_factory=dict)
    alignment: dict[str, Any] = field(default_factory=dict)
    recall: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandOutcome:
    seq: int
    op: str
    budget_exceeded: bool
    elapsed_ms: float
    affected_samples: list[str]
    completed_samples: int
    channel_candidate_counts: dict[str, int]
    reused: list[str]
    recomputed: list[str]
    stale: bool
    report: dict[str, Any]


class Engine:
    def __init__(self, config: Config, sample_root: str, snapshot_root: str | None = None):
        self.config = config
        self.sample_root = sample_root
        self.params = RuntimeParams(
            weights=dict(config.weights),
            channels_enabled=dict(config.channels_enabled),
            frac_bits=config.frac_bits,
            tier=config.tier,
            active_tier=config.active_tier,
        )
        self.config_fingerprint = digest_obj(config.to_fingerprint_obj())
        self.ledger = Ledger()
        self.artifacts = StageArtifacts()
        self.sample_set = SampleSet()
        self.decision: StrategyDecision | None = None
        self.last_report: dict[str, Any] | None = None
        self.last_summary: dict[str, Any] | None = None
        snapshot_root = snapshot_root or config.snapshot_dir
        self.snapshot = SnapshotStore(snapshot_root, config.temp_dir,
                                      self._snapshot_config_hash())
        self.manifest = initial_manifest(self._snapshot_config_hash())
        self.peak_concurrency = 0
        self._last_pipeline: dict[str, Any] | None = None
        self._dirty = False
        self._last_commit_cursor = 0

    # ------------------------------------------------------------------ #
    # 配置/参数指纹
    # ------------------------------------------------------------------ #
    def _snapshot_config_hash(self) -> str:
        return digest_many("snapshot-config", self.config_fingerprint)

    def _effective_config(self) -> Config:
        """把运行期参数投影回 Config 视图，供特征/对齐模块使用。"""

        view = copy.copy(self.config)
        view.weights = dict(self.params.weights)
        view.channels_enabled = dict(self.params.channels_enabled)
        view.frac_bits = self.params.frac_bits
        view.tier = self.params.tier
        view.active_tier = self.params.active_tier
        return view

    def _param_fingerprint(self, scope: str) -> str:
        scoped = {
            "visual": {"quant": [self.params.frac_bits, self.params.tier]},
            "text": {"quant": [self.params.frac_bits, self.params.tier]},
            "alignment": {
                "quant": [self.params.frac_bits, self.params.tier],
                "weights": self.params.weights,
            },
            "recall": {
                "quant": [self.params.frac_bits, self.params.tier],
                "weights": self.params.weights,
                "channels_enabled": self.params.channels_enabled,
                "channel_weights": self.config.channel_weights,
                "fusion_temperature": self.config.fusion_temperature,
            },
            "diagnosis": {"full": self.params.fingerprint()},
        }[scope]
        return digest_many("param", scope, digest_obj(scoped),
                           self.config_fingerprint)

    # ------------------------------------------------------------------ #
    # 命令应用（仅改参数/触发样本替换标记，不直接计算）
    # ------------------------------------------------------------------ #
    def apply_command(self, command: Command) -> set[str]:
        payload = command.payload
        affected: set[str] = set()
        if command.op == "set_scale_tier":
            self.params.active_tier = int(payload["value"])
            # 特征本身与规模档位无关，仅对齐/召回/诊断需要重算（策略可能改变）
            self.ledger.invalidate("alignment")
        elif command.op == "set_weight":
            self.params.weights[payload["channel"]] = float(payload["value"])
            self.ledger.invalidate("alignment")
        elif command.op == "set_quant":
            self.params.frac_bits = int(payload["frac_bits"])
            self.params.tier = str(payload["tier"])
            self.ledger.invalidate("visual")
            self.ledger.invalidate("text")
        elif command.op in ("replace_image", "replace_text"):
            sample_name = payload["sample"]
            affected = self._reload_sample(sample_name, command.op)
        elif command.op == "disable_channel":
            self.params.channels_enabled[payload["channel"]] = False
            self.ledger.invalidate("recall")
        elif command.op == "enable_channel":
            self.params.channels_enabled[payload["channel"]] = True
            self.ledger.invalidate("recall")
        return affected

    def _reload_sample(self, sample_name: str, command_op: str) -> set[str]:
        """替换样本文件后重新扫描，更新样本库并精确失效。"""

        from .samples import load_sample

        self._materialize_replace(sample_name, command_op)
        image_path = os.path.join(self.sample_root, f"{sample_name}.png")
        text_path = os.path.join(self.sample_root, f"{sample_name}.text.json")
        region_path = os.path.join(self.sample_root, f"{sample_name}.regions.json")
        sample, failure = load_sample(
            sample_name, image_path, text_path, region_path,
            self.config.resolution, self.config.vocabulary
        )
        prior_failure = any(item.sample == sample_name
                            for item in self.sample_set.failures)
        self.sample_set.failures = [
            item for item in self.sample_set.failures if item.sample != sample_name
        ]
        self.sample_set.skipped.discard(sample_name)
        if failure is not None:
            self.sample_set.failures.append(failure)
            self.sample_set.skipped.add(sample_name)
            self.sample_set.samples.pop(sample_name, None)
            self.ledger.invalidate("visual", {sample_name})
            self.ledger.invalidate("text", {sample_name})
            self.artifacts.visual.pop(sample_name, None)
            self.artifacts.text.pop(sample_name, None)
            return {sample_name}

        old_sample = self.sample_set.samples.get(sample_name)
        old_image_fp = old_sample.image_fingerprint if old_sample else None
        old_text_fp = old_sample.text_fingerprint if old_sample else None
        self.sample_set.samples[sample_name] = sample
        # 按编辑类型与对应内容指纹精确失效
        if command_op == "replace_image" and (prior_failure or old_image_fp != sample.image_fingerprint):
            self.ledger.invalidate("visual", {sample_name})
        if command_op == "replace_text" and (prior_failure or old_text_fp != sample.text_fingerprint):
            self.ledger.invalidate("text", {sample_name})
        return {sample_name}

    def _materialize_replace(self, sample_name: str, op: str) -> None:
        """用固定种子在原文件上生成替换内容，保证可复现且真实改变内容指纹。

        图像替换与文本替换使用不同派生键，避免两次替换生成同一内容。
        """

        import json as _json

        from PIL import Image

        from .generator import generate_sample
        from .rng import child_rng, root_rng

        if sample_name not in self.sample_set.samples:
            return
        rng = child_rng(
            root_rng(20261123),
            f"replace:{op}:{sample_name}:{self.params.fingerprint()}",
        )
        payload = generate_sample(sample_name, self.config.raw, rng)
        if op == "replace_image":
            image_path = os.path.join(self.sample_root, f"{sample_name}.png")
            Image.fromarray(payload["image"], mode="RGB").save(
                image_path, format="PNG"
            )
            with open(os.path.join(self.sample_root,
                                   f"{sample_name}.regions.json"), "w",
                      encoding="utf-8") as handle:
                _json.dump(payload["regions"], handle, ensure_ascii=False, indent=2)
        else:
            with open(os.path.join(self.sample_root,
                                   f"{sample_name}.text.json"), "w",
                      encoding="utf-8") as handle:
                _json.dump(payload["text"], handle, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # 样本装载与规模档位取数
    # ------------------------------------------------------------------ #
    def load_samples(self) -> None:
        self.sample_set = scan_samples(
            self.sample_root, self.config.resolution, self.config.vocabulary
        )

    def _active_names(self) -> list[str]:
        names = sorted(self.sample_set.samples)
        return names[: self.params.active_tier]

    # ------------------------------------------------------------------ #
    # 阶段流水线
    # ------------------------------------------------------------------ #
    def run_pipeline(self, pool: BoundedPool | None = None) -> dict[str, Any]:
        pool = pool or BoundedPool(self.config.thread_pool_size,
                                   self.config.work_ms_per_task)
        config_view = self._effective_config()
        spec = self.params.quant()
        active_names = self._active_names()
        active_samples = [self.sample_set.samples[name] for name in active_names]
        self.decision = select_strategy(config_view, active_samples,
                                        self.params.active_tier)
        target_assertion = max(
            (sample.assertion_len() for sample in active_samples), default=0
        ) or self.config.default_assertion_count

        visual_tasks: list[Task] = []
        text_tasks: list[Task] = []
        for name in active_names:
            sample = self.sample_set.samples[name]
            visual_tasks.append(Task(
                stable_task_key(name, "visual"),
                lambda s=sample: features_mod.extract_visual(
                    s, config_view, self.config_fingerprint, spec,
                    self.decision.feature_precision
                ),
                f"visual:{name}",
            ))
            text_tasks.append(Task(
                stable_task_key(name, "text"),
                lambda s=sample: features_mod.extract_text(
                    s, config_view, self.config_fingerprint, spec,
                    target_assertion
                ),
                f"text:{name}",
            ))
        visual_results = pool.map_ordered(visual_tasks)
        text_results = pool.map_ordered(text_tasks)
        self.peak_concurrency = max(self.peak_concurrency, pool.peak_concurrency)

        reused: list[str] = []
        recomputed: list[str] = []
        visual_store: dict[str, Any] = {}
        text_store: dict[str, Any] = {}
        desired: dict[tuple[str, str], str] = {}

        for (_, visual) in visual_results:
            source_fp = self.sample_set.samples[visual.sample].image_fingerprint
            desired_key = make_cache_key(
                "visual", visual.sample,
                [visual.feature_fingerprint, source_fp],
                self._param_fingerprint("visual"), spec.key(),
                visual.quant["q"].tolist(),
            )
            desired[("visual", visual.sample)] = desired_key
            current = self.ledger.get("visual", visual.sample)
            if current is not None and current.cache_key == desired_key:
                visual_store[visual.sample] = self.artifacts.visual[visual.sample]
                reused.append(f"visual:{visual.sample}")
            else:
                entry = LedgerEntry(
                    stage="visual", sample=visual.sample, cache_key=desired_key,
                    upstream_keys=[visual.feature_fingerprint],
                    referenced_tiles=list(visual.referenced_tiles),
                    payload_fingerprint=digest_obj(visual.quant["q"].tolist()),
                )
                self.ledger.upsert(entry)
                self.artifacts.visual[visual.sample] = visual
                visual_store[visual.sample] = visual
                recomputed.append(f"visual:{visual.sample}")

        for (_, text) in text_results:
            source_fp = self.sample_set.samples[text.sample].text_fingerprint
            desired_key = make_cache_key(
                "text", text.sample,
                [text.feature_fingerprint, source_fp],
                self._param_fingerprint("text"), spec.key(),
                text.quant["q"].tolist(),
            )
            desired[("text", text.sample)] = desired_key
            current = self.ledger.get("text", text.sample)
            if current is not None and current.cache_key == desired_key:
                text_store[text.sample] = self.artifacts.text[text.sample]
                reused.append(f"text:{text.sample}")
            else:
                entry = LedgerEntry(
                    stage="text", sample=text.sample, cache_key=desired_key,
                    upstream_keys=[text.feature_fingerprint],
                    referenced_assertions=[ref["index"] for ref in text.assertion_refs],
                    payload_fingerprint=digest_obj(text.quant["q"].tolist()),
                )
                self.ledger.upsert(entry)
                self.artifacts.text[text.sample] = text
                text_store[text.sample] = text
                recomputed.append(f"text:{text.sample}")

        # 对齐阶段：逐查询计算，先判定账本命中；命中则直接复用历史对齐产物
        align_tasks: list[Task] = []
        for query_name in active_names:
            align_tasks.append(Task(
                stable_task_key(query_name, "alignment"),
                lambda q=query_name: run_query(
                    q, visual_store, text_store, self.sample_set.samples,
                    config_view, spec, self.decision, active_names
                ),
                f"alignment:{query_name}",
            ))
        align_results = pool.map_ordered(align_tasks)
        self.peak_concurrency = max(self.peak_concurrency, pool.peak_concurrency)

        reports: dict[str, Any] = {}
        for (_, query_result) in sorted(align_results, key=lambda item: item[0]):
            query_name = query_result.query
            alignment_payload = query_result.alignment_details
            upstream_visual = [visual_store[n].feature_fingerprint
                               for n in active_names]
            upstream_text = [text_store[n].feature_fingerprint for n in active_names]
            align_key = make_cache_key(
                "alignment", query_name,
                upstream_visual + upstream_text,
                self._param_fingerprint("alignment"), spec.key(),
                alignment_payload,
            )
            desired[("alignment", query_name)] = align_key
            align_entry = self.ledger.get("alignment", query_name)
            if align_entry is not None and align_entry.cache_key == align_key:
                reused.append(f"alignment:{query_name}")
                alignment_payload = self.artifacts.alignment[query_name]
            else:
                recomputed.append(f"alignment:{query_name}")
                self.artifacts.alignment[query_name] = alignment_payload
            self.ledger.upsert(LedgerEntry(
                stage="alignment", sample=query_name, cache_key=align_key,
                upstream_keys=[visual_store[n].feature_fingerprint for n in active_names]
                + [text_store[n].feature_fingerprint for n in active_names],
                referenced_tiles=[
                    tuple(t) for t in alignment_payload.get(
                        next(iter(alignment_payload)), {}
                    ).get("used_tiles", [])
                ] if alignment_payload else [],
                payload_fingerprint=digest_obj(alignment_payload),
            ))

            # 召回阶段：对齐未命中或召回键失效时才融合；否则复用历史报告
            base_payload = query_result.to_dict()
            base_payload["alignment_trace"] = {
                candidate: detail["tile_similarities"]
                for candidate, detail in alignment_payload.items()
            }
            base_payload["used_tiles"] = {
                candidate: detail["used_tiles"]
                for candidate, detail in alignment_payload.items()
            }
            recall_key = make_cache_key(
                "recall", query_name, [align_key],
                self._param_fingerprint("recall"), spec.key(), base_payload
            )
            desired[("recall", query_name)] = recall_key
            recall_entry = self.ledger.get("recall", query_name)
            if recall_entry is not None and recall_entry.cache_key == recall_key \
                    and query_name in self.artifacts.recall:
                reused.append(f"recall:{query_name}")
                reports[query_name] = self.artifacts.recall[query_name]
            else:
                recomputed.append(f"recall:{query_name}")
                self.artifacts.recall[query_name] = base_payload
                reports[query_name] = base_payload
            self.ledger.upsert(LedgerEntry(
                stage="recall", sample=query_name, cache_key=recall_key,
                upstream_keys=[align_key],
                payload_fingerprint=digest_obj(reports[query_name]),
            ))

        # 诊断汇总（阶段键依赖全部召回结果）
        diagnosis = self._build_diagnosis(reports, active_names)
        diag_key = make_cache_key(
            "diagnosis", "<global>",
            [reports[name]["fingerprint"] for name in active_names],
            self._param_fingerprint("diagnosis"), spec.key(), diagnosis
        )
        desired[("diagnosis", "<global>")] = diag_key
        diag_entry = self.ledger.get("diagnosis", "<global>")
        if diag_entry is not None and diag_entry.cache_key == diag_key:
            reused.append("diagnosis:<global>")
        else:
            recomputed.append("diagnosis:<global>")
        self.ledger.upsert(LedgerEntry(
            stage="diagnosis", sample="<global>", cache_key=diag_key,
            upstream_keys=[reports[name]["fingerprint"] for name in active_names],
            payload_fingerprint=digest_obj(diagnosis),
        ))

        return {
            "reports": reports,
            "diagnosis": diagnosis,
            "reused": sorted(set(reused)),
            "recomputed": sorted(set(recomputed)),
            "active_names": active_names,
            "desired": desired,
            "pool_peak": pool.peak_concurrency,
        }

    # ------------------------------------------------------------------ #
    # 诊断与边界样本
    # ------------------------------------------------------------------ #
    def _build_diagnosis(self, reports: dict[str, Any],
                         active_names: list[str]) -> dict[str, Any]:
        boundary: list[dict[str, Any]] = []
        for name in active_names:
            candidates = reports[name]["candidates"]
            if len(candidates) >= 2:
                top, second = candidates[0], candidates[1]
                gap = top["fused_q"] - second["fused_q"]
                scale = max(1, abs(top["fused_q"]))
                relative_gap = round(gap / scale, 6)
                # 边界样本判定：相对差距 <=15%、完全同分、或逐块对齐符号翻转
                tile_scores = reports[name].get("alignment_trace", {}).get(
                    top["sample"], []
                )
                sign_flip = any(row["score_q"] < 0 for row in tile_scores)
                if relative_gap <= 0.15 or gap == 0 or sign_flip:
                    boundary.append({
                        "query": name,
                        "top": top["sample"],
                        "second": second["sample"],
                        "gap_q": gap,
                        "relative_gap": relative_gap,
                        "reason": "tie" if gap == 0 else
                        ("sign_flip" if sign_flip else "narrow_gap"),
                    })
            if any(sum(item["channel_contrib"].values()) == 0 for item in candidates):
                boundary.append({"query": name, "reason": "zero_contribution_candidate"})
        quant_records = []
        for name in active_names[:2]:
            visual = self.artifacts.visual.get(name)
            if visual is not None:
                quant_records.append({
                    "sample": name,
                    "step": visual.quant["step"],
                    "tier": visual.quant["tier"],
                    "frac_bits": visual.quant["frac_bits"],
                    "residual_max_abs": float(
                        abs(visual.quant["residual"]).max()
                    ),
                    "clipped": visual.quant["clipped"],
                })
        return {
            "boundary_samples": sorted(boundary, key=lambda row: json.dumps(row, sort_keys=True)),
            "quantization_audit": quant_records,
            "decision": self.decision.to_dict() if self.decision else None,
            "failures": [item.to_dict() for item in self.sample_set.failures],
        }

    # ------------------------------------------------------------------ #
    # 报告与交互
    # ------------------------------------------------------------------ #
    def baseline_report(self, pool: BoundedPool | None = None) -> dict[str, Any]:
        """初始装载后的首次完整运行，建立账本与快照基线。"""

        pool = pool or BoundedPool(self.config.thread_pool_size,
                                   self.config.work_ms_per_task)
        self._last_pipeline = self.run_pipeline(pool)
        report = self.build_report(self._last_pipeline, None, False, 0.0, [], 0)
        self._dirty = True
        self._maybe_snapshot(0, report)
        return report

    def _channel_candidate_counts(self, reports: dict[str, Any],
                                  active_names: list[str]) -> dict[str, int]:
        counts = {channel: 0 for channel in self.decision.channels}
        for name in active_names:
            for candidate in reports[name]["candidates"]:
                for channel, contribution in candidate["channel_contrib"].items():
                    if contribution:
                        counts[channel] = counts.get(channel, 0) + 1
        return counts

    def build_report(self, pipeline: dict[str, Any], command: Command | None,
                     budget_exceeded: bool, elapsed_ms: float,
                     affected: list[str], cursor: int) -> dict[str, Any]:
        active_names = pipeline["active_names"]
        reports = pipeline["reports"]
        diagnosis = pipeline["diagnosis"]
        summary = {
            "completed_samples": len(active_names),
            "channel_candidate_counts": self._channel_candidate_counts(reports, active_names),
            "affected_samples": sorted(affected),
            "active_tier": self.params.active_tier,
            "strategy": self.decision.strategy_id if self.decision else None,
        }
        report = {
            "schema_version": 1,
            "command_cursor": cursor,
            "command": None if command is None else {"seq": command.seq, "op": command.op},
            "budget": {
                "limit_ms": self.config.latency_budget_ms,
                "elapsed_ms": round(elapsed_ms, 3),
                "exceeded": budget_exceeded,
                "code": E_INTERACT_BUDGET_EXCEEDED if budget_exceeded else None,
            },
            "config_fingerprint": self.config_fingerprint,
            "param_fingerprint": self.params.fingerprint(),
            "quant": {"frac_bits": self.params.frac_bits, "tier": self.params.tier},
            "summary": summary,
            "decision": diagnosis["decision"],
            "per_query": [reports[name] for name in active_names],
            "boundary_samples": diagnosis["boundary_samples"],
            "quantization_audit": diagnosis["quantization_audit"],
            "failures": diagnosis["failures"],
            "ledger": {
                "reused": pipeline["reused"],
                "recomputed": pipeline["recomputed"],
                "entries": self.ledger.export(),
            },
            "concurrency": {
                "max_workers": self.config.thread_pool_size,
                "observed_peak": pipeline["pool_peak"],
            },
        }
        self.last_report = report
        self.last_summary = summary
        return report

    def handle_command(self, command: Command, cursor: int,
                       pool: BoundedPool | None = None) -> CommandOutcome:
        """命令触达：预算内返回最新完整结果，否则返回过期摘要。"""

        start = time.perf_counter()
        affected = sorted(self.apply_command(command))
        pool = pool or BoundedPool(self.config.thread_pool_size,
                                   self.config.work_ms_per_task)
        pipeline = self.run_pipeline(pool)
        self._last_pipeline = pipeline
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        budget_exceeded = elapsed_ms > self.config.latency_budget_ms

        if budget_exceeded and self.last_summary is not None:
            stale_report = self._stale_report(command, elapsed_ms, affected, cursor)
            return CommandOutcome(
                seq=command.seq, op=command.op, budget_exceeded=True,
                elapsed_ms=elapsed_ms, affected_samples=affected,
                completed_samples=self.last_summary["completed_samples"],
                channel_candidate_counts=self.last_summary["channel_candidate_counts"],
                reused=pipeline["reused"], recomputed=pipeline["recomputed"],
                stale=True, report=stale_report,
            )

        report = self.build_report(pipeline, command, budget_exceeded,
                                   elapsed_ms, affected, cursor)
        self._dirty = True
        if cursor % self.config.commit_interval_commands == 0:
            self._maybe_snapshot(cursor, report)
        else:
            self.manifest = self.snapshot.update_cursor(self.manifest, cursor)
        return CommandOutcome(
            seq=command.seq, op=command.op, budget_exceeded=budget_exceeded,
            elapsed_ms=elapsed_ms, affected_samples=affected,
            completed_samples=report["summary"]["completed_samples"],
            channel_candidate_counts=report["summary"]["channel_candidate_counts"],
            reused=pipeline["reused"], recomputed=pipeline["recomputed"],
            stale=False, report=report,
        )

    def _stale_report(self, command: Command, elapsed_ms: float,
                      affected: list[str], cursor: int) -> dict[str, Any]:
        report = copy.deepcopy(self.last_report)
        report["command_cursor"] = cursor
        report["command"] = {"seq": command.seq, "op": command.op}
        report["budget"] = {
            "limit_ms": self.config.latency_budget_ms,
            "elapsed_ms": round(elapsed_ms, 3),
            "exceeded": True,
            "code": E_INTERACT_BUDGET_EXCEEDED,
        }
        report["stale"] = True
        report["summary"]["affected_samples"] = sorted(
            set(report["summary"].get("affected_samples", [])) | set(affected)
        )
        return report

    # ------------------------------------------------------------------ #
    # 快照
    # ------------------------------------------------------------------ #
    def _maybe_snapshot(self, cursor: int, report: dict[str, Any]) -> None:
        for stage in ("visual", "text", "alignment", "recall", "diagnosis"):
            if stage == "diagnosis":
                payload = {
                    "boundary_samples": report["boundary_samples"],
                    "quantization_audit": report["quantization_audit"],
                    "decision": report["decision"],
                    "failures": report["failures"],
                }
            elif stage == "recall":
                payload = report["per_query"]
            elif stage == "alignment":
                payload = self.artifacts.alignment
            else:
                payload = {
                    name: {
                        "fingerprint": self.artifacts.__dict__[stage][name].feature_fingerprint,
                        "q": self.artifacts.__dict__[stage][name].quant["q"].tolist(),
                    }
                    for name in self._active_names()
                    if name in self.artifacts.__dict__[stage]
                }
            prior_hash = self.manifest.stage_hashes.get(stage, "")
            changed = digest_obj(payload) != prior_hash
            self.manifest, _ = self.snapshot.commit_stage(
                stage, payload, self.manifest, changed
            )
        # 命令游标每条命令都推进；commit_interval_commands 决定阶段提交节奏
        self.manifest = self.snapshot.update_cursor(self.manifest, cursor)
        self._dirty = False
        self._last_commit_cursor = cursor

    def resume_or_rebuild(self) -> tuple[str | None, int]:
        """启动恢复判定，返回 (错误码, 续跑游标)。"""

        manifest, code = self.snapshot.resume_manifest()
        if code == E_SNAPSHOT_STALE:
            self.snapshot.reset()
            return E_SNAPSHOT_STALE, 0
        if manifest is not None:
            self.manifest = manifest
            return None, manifest.command_cursor
        return None, 0

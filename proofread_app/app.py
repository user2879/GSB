"""应用装配与交互命令：增量失效、重算、快照与报告。"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cache import ContentCache
from .config import AppConfig, load_config
from .errors import ProofreadError
from .loader import (
    load_samples,
    load_rules,
    load_scoring_params,
    validate_rule_hints,
)
from .models import Rule, Sample
from .pipeline import Pipeline
from .report import build_report, write_corrections, write_report
from .signals import SignalController
from .snapshot import SnapshotStore


class ProofreadApp:
    """装配全部组件并维护可交互调参的应用状态。"""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.config_path = self.base_dir / "config" / "proofread.yaml"
        self.samples_path = self.base_dir / "samples" / "sentences.jsonl"
        self.rules_path = self.base_dir / "resources" / "rules.json"
        self.scoring_path = self.base_dir / "resources" / "scoring.json"
        self.state_dir = self.base_dir / "state"
        self.output_dir = self.base_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config: AppConfig = load_config(self.config_path)
        self.resource = load_rules(self.rules_path)
        self.scoring = load_scoring_params(self.scoring_path)
        self.samples, self.failures = load_samples(self.samples_path)
        self.failures.extend(validate_rule_hints(
            self.samples, self.resource, str(self.samples_path)))
        self.samples.sort(key=lambda item: item.sample_id)

        self.snapshots = SnapshotStore(self.state_dir / "snapshots")
        self.cache = ContentCache(self.state_dir / "cache.json")
        self.controller = SignalController()
        self._pipeline: Optional[Pipeline] = None
        self.last_results: List = []
        self.last_manifest = None
        self.invalidated_keys: List[str] = []

    # ---- 装配 ----
    def _build_pipeline(self) -> Pipeline:
        # 复用 app 上安装了信号处理器的同一个 SignalController，
        # 保证 SIGTERM/SIGINT 状态在多次构建间共享
        pipeline = Pipeline(
            config=self.config,
            resource=self.resource,
            scoring_fingerprint=self.scoring.source_fingerprint,
            cache=self.cache,
            snapshots=self.snapshots,
            signal_controller=self.controller,
        )
        pipeline.bind_scoring_params(self.scoring)
        return pipeline

    def check_empty_state(self) -> None:
        manifest = self.snapshots.manifest_path
        if manifest.exists():
            raise ProofreadError(
                "E_STATE_NOT_EMPTY",
                "清单目录非空：首次全量运行前请清理 state/ 或使用 --resume",
                source=str(self.state_dir))

    # ---- 运行 ----
    def run(self, resume: bool = False) -> Dict[str, Any]:
        existing = None
        if resume:
            existing = self.snapshots.resume_guard(self._config_hash())
        elif self.snapshots.manifest_exists():
            raise ProofreadError(
                "E_STATE_NOT_EMPTY",
                "已存在快照，首次全量运行请加 --clean 或使用 --resume",
                source=str(self.state_dir))
        self.cache.reset_counters()
        pipeline = self._build_pipeline()
        results, manifest, exited = pipeline.run_batch(self.samples, existing)
        self.last_results = results
        self.last_manifest = manifest
        recovery = self.snapshots.recovery_point()
        report = build_report(
            results, self.failures, manifest,
            self._config_hash(), self.resource.source_fingerprint,
            self.cache.counters(), recovery)
        write_report(self.output_dir / "diagnostics.json", report)
        write_corrections(self.output_dir / "corrected.jsonl", results)
        report["graceful_exit"] = exited
        return report

    def _config_hash(self) -> str:
        from .fingerprint import content_fingerprint
        return content_fingerprint(self.config.to_fingerprint_obj())

    # ---- 交互命令 ----
    def apply_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """执行单条交互命令并返回受影响/失效摘要。"""
        name = command["cmd"]
        before_hits = self.cache.hits
        if name == "set_threshold":
            self.config = self.config.with_updates(
                similarity_threshold=float(command["value"]))
            affected = self.cache.invalidate_stages(
                ["detect", "score", "aggregate", "decision"])
            reason = "相似度阈值变化：检测及其下游失效，切分阶段复用"
        elif name == "set_quant_levels":
            self.config = self.config.with_updates(
                quant_levels=int(command["value"]))
            affected = self.cache.invalidate_stages(
                ["score", "aggregate", "decision"])
            reason = "量化档位变化：分项打分及其下游失效，切分/检测复用"
        elif name == "set_aggregation_weights":
            self.config = self.config.with_updates(
                aggregation_weights=dict(command["value"]))
            affected = self.cache.invalidate_stages(["aggregate", "decision"])
            reason = "聚合权重变化：仅聚合与决策阶段失效，上游复用"
        elif name == "add_confusion_pair":
            affected = self._add_confusion_pair(command["rule"])
            reason = "新增混淆字对：规则资源指纹变化，匹配该字对的样本重算"
        elif name == "remove_confusion_pair":
            rule_id = command["rule_id"]
            affected = self.cache.invalidate_rule(rule_id)
            self.resource = self._without_rule(rule_id)
            reason = f"删除混淆字对 {rule_id}：引用该规则的样本及其下游失效"
        elif name == "replace_sample":
            affected = self._replace_sample(command["sample"])
            reason = "替换单条样本：仅该样本全部阶段失效，其他样本缓存复用"
        else:
            raise ProofreadError("E_CONFIG_INVALID", f"未知交互命令: {name}")

        self.invalidated_keys = sorted(
            {self._coordinate_to_string(item) for item in affected})
        return {
            "cmd": name,
            "reason": reason,
            "invalidated_keys": self.invalidated_keys,
            "invalidated_count": len(self.invalidated_keys),
            "cache_hits_before": before_hits,
        }

    def _coordinate_to_string(self, coordinate: tuple) -> str:
        sample_id, stage, key = coordinate
        return f"{sample_id}:{stage}:{key}"

    def _invalidate_all_samples(self) -> List[tuple]:
        keys: List[tuple] = []
        for sample in self.samples:
            keys.extend(self.cache.invalidate_sample(sample.sample_id))
        return keys

    def _add_confusion_pair(self, rule: Dict[str, Any]) -> List[str]:
        new_rule = Rule(
            rule_id=rule["rule_id"], etype="wrong_char",
            pattern=rule["wrong"], replacement=rule["correct"],
            edit_type="replace", weight=float(rule.get("weight", 1.0)))
        for char in (new_rule.pattern, new_rule.replacement):
            if char not in self.resource.char_table:
                raise ProofreadError(
                    "E_RESOURCE_INVALID",
                    f"混淆字对字符 {char} 不在受控字表内")
        rules = sorted([*self.resource.rules, new_rule], key=lambda r: r.rule_id)
        weights = dict(self.resource.rule_weights)
        weights[new_rule.rule_id] = new_rule.weight
        self.resource = replace(
            self.resource, rules=rules, rule_weights=weights,
            source_fingerprint=self._recompute_rule_fp(rules, weights))
        # 新规则可能命中任意含该字的样本：保守失效 pattern 字符命中的样本
        affected: List[tuple] = []
        for sample in self.samples:
            if new_rule.pattern in sample.text:
                affected.extend(self.cache.invalidate_sample(sample.sample_id))
        return affected

    def _without_rule(self, rule_id: str) -> Any:
        rules = [rule for rule in self.resource.rules if rule.rule_id != rule_id]
        weights = {key: value for key, value in self.resource.rule_weights.items()
                   if key != rule_id}
        return replace(
            self.resource, rules=rules, rule_weights=weights,
            source_fingerprint=self._recompute_rule_fp(rules, weights))

    def _recompute_rule_fp(self, rules: List[Rule],
                           weights: Dict[str, float]) -> str:
        from .fingerprint import content_fingerprint
        return content_fingerprint({
            "char_table": sorted(self.resource.char_table),
            "rules": [rule.to_fingerprint_obj() for rule in rules],
            "rule_weights": weights,
        })

    def _replace_sample(self, sample_obj: Dict[str, Any]) -> List[str]:
        new_sample = Sample(
            sample_id=sample_obj["sample_id"],
            text=sample_obj["text"],
            context=sample_obj.get("context"),
            truth=sample_obj.get("truth"),
            line_no=-1,
            rule_hints=list(sample_obj.get("rule_hints", [])),
        )
        index = next((i for i, item in enumerate(self.samples)
                      if item.sample_id == new_sample.sample_id), None)
        if index is None:
            self.samples.append(new_sample)
        else:
            self.samples[index] = new_sample
        self.samples.sort(key=lambda item: item.sample_id)
        return self.cache.invalidate_sample(new_sample.sample_id)

    # ---- 回放 ----
    def replay(self, commands: List[Dict[str, Any]],
               run_after_each: bool = False) -> List[Dict[str, Any]]:
        traces = []
        for command in commands:
            summary = self.apply_command(command)
            trace = {"command": command, "summary": summary}
            if run_after_each:
                trace["report_fingerprint"] = self._run_fingerprint()
            traces.append(trace)
        return traces

    def _run_fingerprint(self) -> str:
        from .fingerprint import content_fingerprint
        report = self.run(resume=False) if not self.snapshots.manifest_exists() \
            else self._rerun_incremental()
        return content_fingerprint({
            "corrected": [r.corrected_text for r in
                          sorted(self.last_results, key=lambda x: x.sample.sample_id)],
            "cache": report["cache"],
        })

    def _rerun_incremental(self) -> Dict[str, Any]:
        # 交互后重跑：缓存中未失效部分直接复用
        return self.run(resume=False)

    def clean_state(self) -> None:
        self.snapshots.reset()
        if self.cache.path and self.cache.path.exists():
            self.cache.path.unlink()
        self.cache = ContentCache(self.state_dir / "cache.json")

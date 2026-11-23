"""说话人音色分析流水线：五阶段串联、精确增量、预算并发与快照恢复。"""

import os

import numpy as np

from . import aggregate as aggregate_mod
from . import attach as attach_mod
from . import audio as audio_mod
from . import cache as cache_mod
from . import features as features_mod
from . import quantize as quantize_mod
from .errors import PipelineError, fail_record
from .budget import BudgetExecutor, BudgetTask
from .snapshot import SnapshotStore, STAGES, build_manifest
from .utils import diff_summary, round_fixed


class PipelineState:
    """单音频在多次命令间持续演进的全部状态。"""

    def __init__(self, audio_id):
        self.audio_id = audio_id
        self.samples = None
        self.segments = []
        self.features = {}
        self.quantized = {}
        self.attach = None
        self.aggregate = None
        self.failures = []
        self.injection_hits = []
        self.previous_vectors = {}
        self.agg_diffs = {}


class TimbrePipeline:
    def __init__(self, config, injection_book=None, snapshot_store=None,
                 deterministic=False, time_func=None):
        self.config = config
        self.injections = injection_book
        self.snapshot = snapshot_store
        self.deterministic = deterministic
        self.time_func = time_func
        self.ledger = cache_mod.CacheLedger()
        self.states = {}
        self.global_version = 0
        self.stage_versions = {stage: 0 for stage in STAGES}
        self.commands = []
        self.fallback_events = []
        self.latest_summary = None

    # ------------------------------------------------------------------
    # 阶段 1：切分
    # ------------------------------------------------------------------
    def process_segmentation(self, wav_path):
        audio = audio_mod.read_wav_strict(wav_path)
        audio_id = audio["audio_id"]
        state = self.states.setdefault(audio_id, PipelineState(audio_id))
        state.samples = audio["samples"]
        result = audio_mod.segment_audio(self.config, audio)
        state.segments = result["segments"]
        key = cache_mod.make_cache_key(
            "segment", audio_id, self.config.stage_fingerprint("segment"),
            [segment["content_fingerprint"] for segment in result["segments"]])
        self.ledger.put(key, result)
        self.ledger.record_reference(audio_id, "segment", key, [])
        return state, result, key

    # ------------------------------------------------------------------
    # 阶段 2/3：特征与量化（可并发）
    # ------------------------------------------------------------------
    def _compute_feature(self, state, segment, injection):
        feature = features_mod.compute_segment_feature(
            self.config, segment, state.samples, injection)
        if segment.get("feature_override"):
            override = segment["feature_override"]
            name = override["dimension"]
            if name in feature["descriptor_inputs"]:
                feature["descriptor_inputs"][name] = round_fixed(float(override["value"]))
            feature["feature_fingerprint"] = features_mod.feature_fingerprint(
                segment["content_fingerprint"],
                self.config.stage_fingerprint("feature"), feature)
        return feature

    def _feature_task(self, state, segment, command_id, injection):
        def runner():
            key = cache_mod.make_cache_key(
                "feature", segment["content_fingerprint"],
                self.config.stage_fingerprint("feature"))
            cached = self.ledger.get(key)
            if cached is not None:
                self.ledger.record_event(command_id, segment["segment_id"], "feature",
                                         "reuse", key, "cache_hit")
                return key, cached
            value = self._compute_feature(state, segment, injection)
            self.ledger.put(key, value)
            self.ledger.record_reference(segment["segment_id"], "feature", key,
                                         [self._segment_reference_key(segment)])
            self.ledger.record_event(command_id, segment["segment_id"], "feature",
                                     "recompute", key, "cache_miss")
            return key, value
        return BudgetTask(state.audio_id, segment["segment_index"], "feature", runner)

    def _quantize_task(self, segment, feature, command_id, injection, schemas, schema_key):
        def runner():
            key = cache_mod.make_cache_key(
                "quantize", feature["feature_fingerprint"], schema_key)
            cached = self.ledger.get(key)
            if cached is not None and injection is None:
                self.ledger.record_event(command_id, segment["segment_id"], "quantize",
                                         "reuse", key, "cache_hit")
                return key, cached
            value = quantize_mod.quantize_with_schema(feature, schemas, injection)
            if injection is None:
                self.ledger.put(key, value)
            self.ledger.record_reference(segment["segment_id"], "quantize", key,
                                         [cache_mod.make_cache_key(
                                             "feature", segment["content_fingerprint"],
                                             self.config.stage_fingerprint("feature"))])
            self.ledger.record_event(command_id, segment["segment_id"], "quantize",
                                     "recompute", key,
                                     injection.get("id", "cache_miss") if injection else "cache_miss")
            return key, value
        return BudgetTask(segment["segment_id"].split("#")[0], segment["segment_index"],
                          "quantize", runner)

    def _build_quant_schemas(self, state):
        rows = [feature["descriptor_inputs"]
                for segment in sorted(state.segments, key=lambda item: item["segment_index"])
                if (feature := state.features.get(segment["segment_id"])) is not None]
        schemas = quantize_mod.build_dimension_schemas(self.config, rows)
        schema_key = cache_mod.make_cache_key(
            "schema", self.config.stage_fingerprint("quantize"),
            quantize_mod.schema_fingerprint(schemas))
        return schemas, schema_key

    def _segment_reference_key(self, segment):
        return cache_mod.make_cache_key(
            "segment", segment["segment_id"].split("#")[0],
            self.config.stage_fingerprint("segment"),
            [segment["content_fingerprint"]])

    # ------------------------------------------------------------------
    # 阶段 4/5：归属与聚合（同音频顺序执行）
    # ------------------------------------------------------------------
    def _truth_map(self, state, truth_labels):
        mapping = {}
        if not truth_labels:
            return mapping
        audio_labels = truth_labels.get(state.audio_id, {})
        for item in audio_labels.get("segments", []):
            mapping[int(item["segment_index"])] = item["speaker"]
        return mapping

    def run_attach_aggregate(self, state, command_id, truth_labels=None, attach_injection=None):
        ordered = sorted(
            (segment for segment in state.segments
             if segment["segment_id"] in state.quantized),
            key=lambda item: item["segment_index"])
        if not ordered:
            raise PipelineError("E_STAGE_FAILED",
                                "音频 {} 没有可归属的描述向量".format(state.audio_id),
                                "attach", state.audio_id)
        quant_list = [state.quantized[segment["segment_id"]] for segment in ordered]
        truth_map = self._truth_map(state, truth_labels)
        attach_key = cache_mod.make_cache_key(
            "attach", [item["quant_fingerprint"] for item in quant_list],
            self.config.stage_fingerprint("attach"))
        cached_attach = self.ledger.get(attach_key) if attach_injection is None else None
        if cached_attach is not None:
            state.attach = cached_attach
            self.ledger.record_event(command_id, state.audio_id, "attach",
                                     "reuse", attach_key, "cache_hit")
        else:
            state.attach = attach_mod.attach_segments(
                self.config, state.audio_id, quant_list, truth_map, attach_injection)
            if attach_injection is None:
                self.ledger.put(attach_key, state.attach)
            self.ledger.record_event(command_id, state.audio_id, "attach",
                                     "recompute", attach_key,
                                     attach_injection.get("id", "cache_miss") if attach_injection else "cache_miss")
        aggregate_key = cache_mod.make_cache_key(
            "aggregate", state.attach["attach_fingerprint"],
            self.config.stage_fingerprint("aggregate"))
        feature_list = [state.features[segment["segment_id"]] for segment in ordered]
        cached_aggregate = self.ledger.get(aggregate_key)
        before_vectors = {speaker["speaker_id"]: speaker["integer_vector"]
                          for speaker in state.previous_vectors.get(state.audio_id, [])}
        if cached_aggregate is not None and not attach_injection:
            state.aggregate = cached_aggregate
            self.ledger.record_event(command_id, state.audio_id, "aggregate",
                                     "reuse", aggregate_key, "cache_hit")
        else:
            state.aggregate = aggregate_mod.aggregate_speakers(
                quant_list, feature_list, state.attach)
            if not attach_injection:
                self.ledger.put(aggregate_key, state.aggregate)
            self.ledger.record_event(command_id, state.audio_id, "aggregate",
                                     "recompute", aggregate_key, "cache_miss")
        state.agg_diffs = {}
        for speaker in state.aggregate["speakers"]:
            state.agg_diffs[speaker["speaker_id"]] = diff_summary(
                before_vectors.get(speaker["speaker_id"]), speaker["integer_vector"])
        state.previous_vectors[state.audio_id] = [
            {"speaker_id": s["speaker_id"], "integer_vector": list(s["integer_vector"])}
            for s in state.aggregate["speakers"]]
        state.aggregate["diff_summary"] = state.agg_diffs
        for assignment in state.attach["assignments"]:
            if assignment["boundary_sample"]:
                for quant in quant_list:
                    if quant["segment_id"] == assignment["segment_id"]:
                        quant["boundary_sample"] = True

    # ------------------------------------------------------------------
    # 全量运行
    # ------------------------------------------------------------------
    def run_full(self, wav_paths, truth_labels=None):
        """全量处理全部 WAV；单音频故障隔离并进入失败清单。"""
        for wav_path in wav_paths:
            audio_id = os.path.splitext(os.path.basename(wav_path))[0]
            try:
                state, _, _ = self.process_segmentation(wav_path)
            except PipelineError as exc:
                state = self.states.setdefault(audio_id, PipelineState(audio_id))
                state.failures.append(exc.to_record())
                continue
            range_failures = self._range_injection_failures(state)
            state.failures.extend(range_failures)
            tasks = self._build_feature_quant_tasks(state, "full-000000")
            executor = self._executor()
            if self.deterministic:
                results, pending, fallback = executor.run_sequential_under_budget(
                    tasks, self.latest_summary)
            else:
                results, pending, fallback = executor.run(tasks, self.latest_summary)
            if fallback:
                self.fallback_events.append(fallback)
            if pending:
                state.failures.append(fail_record(
                    "E_LATENCY_FALLBACK", "全量阶段仍有待办任务", "budget", audio_id,
                    pending=[task.sort_key for task in pending]))
                continue
            self._absorb_task_results(state, results)
            quant_tasks = self._build_quant_tasks(state, "full-000000")
            if self.deterministic:
                quant_results, quant_pending, quant_fallback = executor.run_sequential_under_budget(
                    quant_tasks, self.latest_summary)
            else:
                quant_results, quant_pending, quant_fallback = executor.run(quant_tasks, self.latest_summary)
            if quant_fallback:
                self.fallback_events.append(quant_fallback)
            if quant_pending:
                state.failures.append(fail_record(
                    "E_LATENCY_FALLBACK", "全量量化阶段仍有待办任务", "budget", audio_id,
                    pending=[task.sort_key for task in quant_pending]))
                continue
            self._absorb_task_results(state, quant_results)
            try:
                self._run_attach_with_injection(state, "full-000000", truth_labels)
            except PipelineError as exc:
                state.failures.append(exc.to_record())
                state.injection_hits.append(exc.to_record())
        self.global_version += 1
        self._bump_stages(STAGES)
        self.latest_summary = self.build_summary()
        self._commit_snapshot()
        return self.build_report()

    def _range_injection_failures(self, state):
        if not self.injections:
            return []
        records = self.injections.validate_segment_range(
            state.audio_id, len(state.segments))
        for record in records:
            state.injection_hits.append(record)
        return records

    def _build_feature_quant_tasks(self, state, command_id):
        tasks = []
        for segment in sorted(state.segments, key=lambda item: item["segment_index"]):
            injection = None
            if self.injections:
                injection = self.injections.for_target(
                    state.audio_id, segment["segment_index"], "feature")
            tasks.append(self._feature_task(state, segment, command_id, injection))
        # 量化任务依赖特征结果，分两批提交；这里先构造特征批。
        return tasks

    def _build_quant_tasks(self, state, command_id):
        schemas, schema_key = self._build_quant_schemas(state)
        tasks = []
        for segment in sorted(state.segments, key=lambda item: item["segment_index"]):
            if segment["segment_id"] not in state.features:
                continue
            injection = None
            if self.injections:
                injection = self.injections.for_target(
                    state.audio_id, segment["segment_index"], "quantize")
            feature = state.features[segment["segment_id"]]
            tasks.append(self._quantize_task(
                segment, feature, command_id, injection, schemas, schema_key))
        return tasks

    def _executor(self):
        return BudgetExecutor(self.config.get("max_workers"),
                              self.config.get("first_response_budget_ms"))

    def _absorb_task_results(self, state, results):
        for item in sorted(results.values(), key=lambda value: value["sort_key"]):
            payload = item["value"]
            if isinstance(payload, tuple) and payload and payload[0] == "__error__":
                exc = payload[1]
                record = exc.to_record() if hasattr(exc, "to_record") else {
                    "code": "E_STAGE_FAILED", "message": str(exc),
                    "stage": item["stage"], "target": item["sort_key"]}
                state.failures.append(record)
                state.injection_hits.append(record)
                continue
            key, value = payload
            if item["stage"] == "feature":
                state.features[value["segment_id"]] = value
            elif item["stage"] == "quantize":
                state.quantized[value["segment_id"]] = value

    def _run_attach_with_injection(self, state, command_id, truth_labels):
        injection = None
        if self.injections:
            injection = self.injections.threshold_item(state.audio_id)
        self.run_attach_aggregate(state, command_id, truth_labels, injection)
        if injection:
            state.injection_hits.append({
                "code": "INJECTION_APPLIED",
                "injection_id": injection["id"],
                "stage": "attach",
                "target": state.audio_id,
            })

    # ------------------------------------------------------------------
    # 增量编辑
    # ------------------------------------------------------------------
    def apply_edit_commands(self, commands, truth_labels=None):
        """逐条应用编辑命令，只重算受影响段及归属/聚合下游。"""
        reports = []
        for command in commands:
            command_id = command["command_id"]
            self.commands.append(command)
            self.ledger.command_cursor = len(self.commands)
            audio_id = command["audio"]
            state = self.states.get(audio_id)
            if state is None:
                raise PipelineError("E_STAGE_FAILED", "未知音频: {}".format(audio_id),
                                    "edit", audio_id)
            affected_segments = self._invalidate_for_command(state, command)
            self._recompute_affected(state, command_id, affected_segments, truth_labels)
            self.global_version += 1
            self._bump_stages(self._stages_for_command(command))
            self.latest_summary = self.build_summary([command_id])
            if self.global_version % max(1, int(self.config.get("snapshot_commit_every"))) == 0:
                self._commit_snapshot()
            reports.append(self.build_report(command_id=command_id))
        self._commit_snapshot()
        return reports

    def _invalidate_for_command(self, state, command):
        kind = command["type"]
        if kind in ("split", "merge"):
            targets = cache_mod.apply_command(self.config, state.segments, command)
            downstream = set()
            for segment in state.segments:
                downstream.add(segment["segment_id"])
            self.ledger.invalidate_targets(
                command["command_id"], downstream,
                ("feature", "quantize"), "segmentation_edit:{}".format(kind))
            self.ledger.invalidate_targets(
                command["command_id"], {state.audio_id},
                ("attach", "aggregate"), "downstream_of_segmentation_edit")
            state.features = {key: value for key, value in state.features.items()
                              if key in {s["segment_id"] for s in state.segments}}
            state.quantized = {key: value for key, value in state.quantized.items()
                               if key in {s["segment_id"] for s in state.segments}}
            return [segment["segment_id"] for segment in state.segments]
        if kind == "modify_feature":
            targets = cache_mod.apply_command(self.config, state.segments, command)
            downstream = self.ledger.downstream_targets(set(targets))
            self.ledger.invalidate_targets(
                command["command_id"], downstream,
                ("feature", "quantize"), "feature_override")
            self.ledger.invalidate_targets(
                command["command_id"], {state.audio_id},
                ("attach", "aggregate"), "downstream_of_feature_edit")
            return targets
        raise PipelineError("E_STAGE_FAILED", "不支持的编辑命令: {}".format(kind),
                            "edit", state.audio_id)

    def _recompute_affected(self, state, command_id, affected_segments, truth_labels):
        feature_tasks = []
        for segment in state.segments:
            if segment["segment_id"] not in affected_segments:
                continue
            injection = None
            if self.injections:
                injection = self.injections.for_target(
                    state.audio_id, segment["segment_index"], "feature")
            feature_tasks.append(self._feature_task(state, segment, command_id, injection))
        executor = self._executor()
        if self.deterministic:
            feature_results, pending, fallback = executor.run_sequential_under_budget(
                feature_tasks, self.latest_summary)
        else:
            feature_results, pending, fallback = executor.run(feature_tasks, self.latest_summary)
        if fallback:
            self.fallback_events.append(fallback)
        self._absorb_task_results(state, feature_results)
        if pending:
            state.failures.append(fail_record(
                "E_LATENCY_FALLBACK", "预算耗尽，待办保留", "budget", state.audio_id,
                pending=[task.sort_key for task in pending]))
            return
        quant_tasks = self._build_quant_tasks(state, command_id)
        if self.deterministic:
            quant_results, pending, fallback = executor.run_sequential_under_budget(
                quant_tasks, self.latest_summary)
        else:
            quant_results, pending, fallback = executor.run(quant_tasks, self.latest_summary)
        if fallback:
            self.fallback_events.append(fallback)
        self._absorb_task_results(state, quant_results)
        if pending:
            state.failures.append(fail_record(
                "E_LATENCY_FALLBACK", "预算耗尽，量化待办保留", "budget", state.audio_id,
                pending=[task.sort_key for task in pending]))
            return
        try:
            self._run_attach_with_injection(state, command_id, truth_labels)
        except PipelineError as exc:
            state.failures.append(exc.to_record())
            state.injection_hits.append(exc.to_record())

    def _stages_for_command(self, command):
        if command["type"] in ("split", "merge"):
            return STAGES
        return ("feature", "quantize", "attach", "aggregate")

    def _bump_stages(self, stages):
        for stage in stages:
            self.stage_versions[stage] += 1

    # ------------------------------------------------------------------
    # 配置切换（位宽/阈值）
    # ------------------------------------------------------------------
    def apply_config(self, new_config, command_id, truth_labels=None):
        """切换量化档位或相似度阈值：按阶段配置指纹精确失效。"""
        old_config = self.config
        self.config = new_config
        self.commands.append({"command_id": command_id, "type": "config_switch"})
        self.ledger.command_cursor = len(self.commands)
        for state in self.states.values():
            if not state.segments:
                continue
            if old_config.stage_fingerprint("quantize") != new_config.stage_fingerprint("quantize"):
                self.ledger.invalidate_targets(
                    command_id, {s["segment_id"] for s in state.segments},
                    ("quantize",), "quantization_config_changed")
                self.ledger.invalidate_targets(
                    command_id, {state.audio_id}, ("attach", "aggregate"),
                    "downstream_of_quantization_config")
            if old_config.stage_fingerprint("attach") != new_config.stage_fingerprint("attach"):
                self.ledger.invalidate_targets(
                    command_id, {state.audio_id}, ("attach", "aggregate"),
                    "threshold_config_changed")
            affected = [s["segment_id"] for s in state.segments]
            self._recompute_affected(state, command_id, affected, truth_labels)
        self.global_version += 1
        self._bump_stages(("feature", "quantize", "attach", "aggregate"))
        self.latest_summary = self.build_summary([command_id])
        self._commit_snapshot()
        return self.build_report(command_id=command_id)

    # ------------------------------------------------------------------
    # 快照与恢复
    # ------------------------------------------------------------------
    def _commit_snapshot(self):
        if self.snapshot is None:
            return
        levels = {}
        for audio_id, state in self.states.items():
            levels[audio_id] = {
                segment_id: [dim["selected_level"] for dim in item["dimensions"]]
                for segment_id, item in state.quantized.items()}
        manifest = build_manifest(
            self.global_version, self.config, self.ledger.command_cursor,
            self.stage_versions, levels)
        payload = {
            "states": {
                audio_id: {
                    "segments": state.segments,
                    "features": state.features,
                    "quantized": state.quantized,
                    "attach": state.attach,
                    "aggregate": state.aggregate,
                    "failures": state.failures,
                    "injection_hits": state.injection_hits,
                    "previous_vectors": state.previous_vectors,
                    "agg_diffs": state.agg_diffs,
                }
                for audio_id, state in self.states.items()
            },
            "ledger": self.ledger.snapshot(),
            "commands": self.commands,
            "fallback_events": self.fallback_events,
            "stage_versions": self.stage_versions,
            "global_version": self.global_version,
        }
        for stage in STAGES:
            stage_payload = self._stage_payload(stage, payload)
            self.snapshot.commit_stage(stage, stage_payload, manifest)
        ledger_manifest = dict(manifest)
        ledger_manifest["stage_versions"] = dict(self.stage_versions)
        self.snapshot.commit_stage("ledger", payload, ledger_manifest)

    def _stage_payload(self, stage, full_payload):
        if stage == "segment":
            return {audio_id: state["segments"]
                    for audio_id, state in full_payload["states"].items()}
        if stage == "feature":
            return {audio_id: state["features"]
                    for audio_id, state in full_payload["states"].items()}
        if stage == "quantize":
            return {audio_id: state["quantized"]
                    for audio_id, state in full_payload["states"].items()}
        if stage == "attach":
            return {audio_id: state["attach"]
                    for audio_id, state in full_payload["states"].items()}
        return {audio_id: state["aggregate"]
                for audio_id, state in full_payload["states"].items()}

    def restore_from_snapshot(self):
        """按版本号与配置哈希判定续跑，恢复阶段产物与缓存账本。"""
        decision = self.snapshot.recovery_decision(self.config.hash)
        if decision["action"] == "cold_start":
            return decision
        for stage in STAGES:
            stage_payload = self.snapshot.load_stage(stage)
            if stage_payload is None:
                continue
            for audio_id, value in stage_payload.items():
                state = self.states.setdefault(audio_id, PipelineState(audio_id))
                if stage == "segment":
                    state.segments = value
                elif stage == "feature":
                    state.features = dict(value)
                elif stage == "quantize":
                    state.quantized = dict(value)
                elif stage == "attach":
                    state.attach = value
                elif stage == "aggregate":
                    state.aggregate = value
        manifest = decision["manifest"]
        self.global_version = int(manifest["global_version"])
        self.stage_versions = dict(manifest["stage_versions"])
        for audio_id, state in self.states.items():
            if not state.segments and not state.failures:
                state.failures.append(fail_record(
                    "E_AUDIO_FORMAT", "恢复快照中该音频无切分产物，维持失败清单",
                    "segment", audio_id))
        ledger_path = os.path.join(self.snapshot.root, "ledger_stage.json")
        if os.path.exists(ledger_path):
            import json
            with open(ledger_path, "r", encoding="utf-8") as handle:
                self.ledger.restore(json.load(handle)["ledger"])
        return decision

    # ------------------------------------------------------------------
    # 摘要与报告
    # ------------------------------------------------------------------
    def build_summary(self, recent_commands=None):
        completed_segments = sum(
            len(state.quantized) for state in self.states.values())
        speaker_count = sum(
            len(state.aggregate["speakers"])
            for state in self.states.values() if state.aggregate)
        dim_count = len(self.config.dimensions())
        affected = []
        for state in self.states.values():
            affected.extend(sorted(state.quantized))
        progress = {
            "global_version": self.global_version,
            "completed_segments": completed_segments,
            "current_speakers": speaker_count,
            "vector_dimensions": dim_count,
            "affected_segments": affected,
            "progress_ratio": self._progress_ratio(),
            "recent_commands": list(recent_commands or []),
            "command_cursor": self.ledger.command_cursor,
        }
        return progress

    def _progress_ratio(self):
        total = sum(max(1, len(state.segments)) for state in self.states.values())
        done = sum(len(state.quantized) for state in self.states.values())
        return round_fixed(done / total) if total else 0.0

    def build_report(self, command_id=None):
        audio_reports = {}
        for audio_id in sorted(self.states):
            state = self.states[audio_id]
            audio_reports[audio_id] = {
                "segment_count": len(state.segments),
                "segments": [
                    {
                        "segment_id": segment["segment_id"],
                        "start_sample": segment["start_sample"],
                        "end_sample": segment["end_sample"],
                        "frame_range": [segment["frame_start"], segment["frame_end"]],
                        "content_fingerprint": segment["content_fingerprint"],
                        "origin": segment["origin"],
                    }
                    for segment in state.segments
                ],
                "features": {
                    segment_id: {
                        "valid_frames": feature["valid_frames"],
                        "total_frames": feature["total_frames"],
                        "f0_stats": feature["f0_stats"],
                        "formant_stats": feature["formant_stats"],
                        "syllable_peak_count": feature["syllable_peak_count"],
                        "descriptor_inputs": feature["descriptor_inputs"],
                        "feature_fingerprint": feature["feature_fingerprint"],
                    }
                    for segment_id, feature in state.features.items()
                },
                "descriptors": {
                    segment_id: {
                        "integer_vector": item["integer_vector"],
                        "descriptor_vector": item["descriptor_vector"],
                        "dimensions": item["dimensions"],
                        "boundary_sample": item["boundary_sample"],
                        "quant_fingerprint": item["quant_fingerprint"],
                    }
                    for segment_id, item in state.quantized.items()
                },
                "attach": state.attach,
                "aggregate": state.aggregate,
                "failures": state.failures,
                "injection_hits": state.injection_hits,
            }
        return {
            "command_id": command_id,
            "global_version": self.global_version,
            "config_hash": self.config.hash,
            "stage_versions": dict(self.stage_versions),
            "summary": self.build_summary(),
            "audio_reports": audio_reports,
            "ledger_events": self.ledger.events,
            "fallback_events": self.fallback_events,
            "failure_list": [
                record for state in self.states.values() for record in state.failures
            ],
        }

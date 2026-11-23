"""语音证据链定点校准主流水线。

阶段顺序：加载校验 -> 并发帧特征/证据强度 -> 逐轮定点校准 ->
说话人归属轨迹 -> 账本失效/复用 -> 预算批次调度 -> 快照提交/恢复。
所有输出按轮次升序、情感/说话人字典序固定，不依赖全局随机状态。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import features as features_mod
from .audio import (AudioClip, load_wav, scan_wav_files, validate_wav)
from .calibration import CalibrationRow, calibrate_round
from .config import CalibConfig, load_config
from .concurrency import TaskEnvelope, effective_workers, run_tasks
from .errors import ErrorCollector, ErrorCode
from .evidence import (EMOTIONS, StrengthArtifact, build_strength)
from .features import FEATURE_DIM, FrameFeatures, extract_features, make_sort_key
from .fixedpoint import FixedPoint
from .hashing import fingerprint, fingerprint_bytes, short_key
from .injection import (Injection, apply_sample_injections,
                        build_injections, check_evidence_injection,
                        check_gain_injection, check_replace_round,
                        has_nonfinite_samples, injections_for_round)
from .ledger import (Ledger, make_attr_entry, make_calib_entry,
                     make_evidence_entry)
from .scheduler import Scheduler
from .scriptdata import ScriptData, load_evidence_jsonl, load_script
from .snapshot import load_snapshot, commit_snapshot
from .speakers import (AttributionRow, Trajectory, attribute_round,
                       build_trajectories, trajectory_session_key)

STAGE_EVIDENCE = "evidence"
STAGE_CALIBRATION = "calibration"
STAGE_ATTRIBUTION = "attribution"
STAGE_TRAJECTORY = "trajectory"


@dataclass
class LoadedInputs:
    config: CalibConfig | None
    script: ScriptData | None
    clips: dict[int, AudioClip]
    evidence_summary: dict[int, dict]
    errors: ErrorCollector
    valid_rounds: list[int]
    injections: list[Injection]


def _round_from_name(path: Path) -> int | None:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def load_inputs(sample_dir: str | Path, config_name: str = "calibration_config.json",
                script_name: str = "dialogue_script.json",
                evidence_name: str = "acoustic_evidence.jsonl",
                injection_file: str | None = None) -> LoadedInputs:
    sample_dir = Path(sample_dir)
    errors = ErrorCollector()

    script = load_script(sample_dir / script_name, errors)
    valid_script_rounds = {r.round for r in script.rounds} if script else set()
    config = load_config(sample_dir / config_name, errors,
                         expected_rounds=len(valid_script_rounds) or None)

    injections: list[Injection] = []
    if config:
        injections.extend(build_injections(config.injections))
    if injection_file:
        from .config import load_injection_file
        extra = load_injection_file(sample_dir / injection_file, errors,
                                    max(len(valid_script_rounds), 1))
        injections.extend(build_injections(extra))

    # 扫描目录下全部 WAV，逐个做 RIFF/PCM 校验。
    expected_files = {r.audio_file for r in script.rounds} if script else set()
    clips: dict[int, AudioClip] = {}
    for wav_path in scan_wav_files(sample_dir):
        rnd = _round_from_name(wav_path)
        if not validate_wav(wav_path, errors, rnd):
            continue
        if wav_path.name not in expected_files:
            errors.add(ErrorCode.E_SCRIPT_STRUCTURE, "audio",
                       f"wav {wav_path.name} not bound to any script round",
                       round=rnd, file=wav_path.name)
            continue
        raw_bytes = wav_path.read_bytes()
        content_hash = fingerprint_bytes(raw_bytes)
        match = next(r for r in script.rounds if r.audio_file == wav_path.name)
        clips[match.round] = load_wav(wav_path, match.round, content_hash)

    # 脚本引用但目录缺失的音频 -> E_AUDIO_MISSING。
    present = {path.name for path in scan_wav_files(sample_dir)}
    for row in script.rounds if script else []:
        if row.audio_file not in present:
            errors.add(ErrorCode.E_AUDIO_MISSING, "audio",
                       f"script references missing audio {row.audio_file}",
                       round=row.round, file=row.audio_file)

    evidence_summary = load_evidence_jsonl(
        sample_dir / evidence_name, errors, valid_script_rounds)
    valid_rounds = sorted(
        rnd for rnd in valid_script_rounds
        if rnd in clips and rnd in evidence_summary
    )
    return LoadedInputs(config, script, clips, evidence_summary, errors,
                        valid_rounds, injections)


@dataclass
class PipelineResult:
    evidence: dict[int, StrengthArtifact] = field(default_factory=dict)
    calibrations: dict[int, CalibrationRow] = field(default_factory=dict)
    attributions: dict[int, AttributionRow] = field(default_factory=dict)
    trajectories: list[Trajectory] = field(default_factory=list)
    ledger: Ledger = field(default_factory=Ledger)
    errors: ErrorCollector = field(default_factory=ErrorCollector)
    batches: list[dict] = field(default_factory=list)
    valid_rounds: list[int] = field(default_factory=list)
    pending_rounds: list[int] = field(default_factory=list)
    snapshot_manifest: dict | None = None
    snapshot_restored: bool = False
    reuse_hits: int = 0
    recompute_rounds: list[int] = field(default_factory=list)
    injections_hit: list[dict] = field(default_factory=list)

    def add_injection_hit(self, injection_id: str, rnd: int, stage: str) -> None:
        marker = (injection_id, rnd, stage)
        existing = {(i["injection_id"], i["round"], i["stage"])
                    for i in self.injections_hit}
        if marker not in existing:
            self.injections_hit.append(
                {"injection_id": injection_id, "round": rnd, "stage": stage})

    def to_report_dict(self) -> dict:
        return {
            "rounds": self.valid_rounds,
            "emotions_order": list(EMOTIONS),
            "evidence": [self.evidence[r].to_dict()
                         for r in self.evidence],
            "calibration": [self.calibrations[r].to_dict()
                            for r in sorted(self.calibrations)],
            "attribution": [self.attributions[r].to_dict()
                            for r in sorted(self.attributions)],
            "trajectories": [t.to_dict() for t in self.trajectories],
            "batches": self.batches,
            "pending_rounds": self.pending_rounds,
            "reuse_hits": self.reuse_hits,
            "recompute_rounds": self.recompute_rounds,
            "ledger_decisions": self.ledger.decisions,
            "injections_hit": self.injections_hit,
            "failures": self.errors.sorted_dict_list(),
            "snapshot": self.snapshot_manifest,
            "snapshot_restored": self.snapshot_restored,
        }


def _task_extract(round_index: int, file_name: str, samples: np.ndarray,
                  config: CalibConfig, sort_key: str) -> FrameFeatures:
    return extract_features(round_index, file_name, samples,
                            config.frame_length_ms, config.frame_shift_ms,
                            sort_key)


def _task_strength(features: FrameFeatures, audio_hash: str,
                   config: CalibConfig) -> StrengthArtifact:
    return build_strength(features, audio_hash, config)


def run_pipeline(inputs: LoadedInputs, output_dir: str | Path,
                 max_batches: int | None = None,
                 resume: bool = True) -> PipelineResult:
    if inputs.config is None or inputs.script is None:
        result = PipelineResult(errors=inputs.errors)
        return result

    config = inputs.config
    script = inputs.script
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "ledger.json"
    result = PipelineResult(errors=inputs.errors,
                            valid_rounds=list(inputs.valid_rounds))

    # ---- 快照恢复判定 ----
    ledger = Ledger()
    persisted: dict = {}
    if resume:
        snap = load_snapshot(output_dir, STAGE_TRAJECTORY, config.config_hash,
                             config.injection_hash, inputs.errors)
        if snap is not None and (snap.path / "ledger.json").exists():
            try:
                ledger = Ledger.load(snap.path / "ledger.json")
                # 决策清单按“单次运行”统计，恢复后清空历史判定，
                # 缓存条目本身保留用于命中比对。
                ledger.decisions = []
                persisted = json.loads(
                    (snap.path / "artifacts.json").read_text(encoding="utf-8"))
                result.snapshot_restored = True
                result.snapshot_manifest = snap.manifest
            except (OSError, json.JSONDecodeError):
                ledger = Ledger()
                persisted = {}
        elif ledger_path.exists() and not inputs.errors.has(
                ErrorCode.E_SNAPSHOT_STALE):
            try:
                ledger = Ledger.load(ledger_path)
                ledger.decisions = []
            except (OSError, json.JSONDecodeError):
                ledger = Ledger()
        if inputs.errors.has(ErrorCode.E_SNAPSHOT_STALE):
            # 版本/配置/注入哈希不符：拒绝旧快照与旧账本，彻底冷启动。
            ledger = Ledger()
            persisted = {}
            result.snapshot_restored = False

    # 内存账本与结果对象关联，保证报告读取到同一份决策清单。
    result.ledger = ledger

    # 音频删除：移除该轮及其下游缓存。
    removed = ledger.prune_missing_rounds(set(inputs.valid_rounds))
    for rnd in removed:
        ledger.record_decision(rnd, "evidence", "recompute",
                               "audio_deleted_downstream_invalidated", None)

    current_audio = {rnd: (inputs.clips[rnd].path.name,
                           inputs.clips[rnd].content_hash)
                     for rnd in inputs.valid_rounds}
    plan = ledger.plan_round_status(
        current_audio,
        param_fingerprint=config.evidence_param_fingerprint,
        frac_bits=config.frac_bits,
        quant_gear=config.quant_gear,
        calib_param_fingerprint=config.param_fingerprint,
    )

    scheduler = Scheduler(config.batch_segment_budget)
    total_rounds = len(inputs.valid_rounds)
    batch_index = 0
    completed: set[int] = set()

    scheduled: set[int] = set()       # 已进入调度器判定（复用或占预算）的轮次
    blocked: set[int] = set()         # 证据/校准阶段失败的轮次

    while len(completed) + len(blocked) < total_rounds:
        if max_batches is not None and batch_index >= max_batches:
            break
        batch_index += 1
        batch = scheduler.start_batch()
        pending_after: list[int] = []

        # 待办队列优先：整体仍按轮次升序，仅遍历尚未调度且未阻断的轮次。
        queued = [rnd for rnd in inputs.valid_rounds
                  if rnd not in scheduled and rnd not in blocked]
        feature_tasks: list[TaskEnvelope] = []
        strength_tasks: list[TaskEnvelope] = []
        batch_rounds: list[int] = []

        for rnd in queued:
            actions = plan.get(rnd, {})
            evidence_action = actions.get(STAGE_EVIDENCE,
                                          "recompute:no_cache_entry")
            calib_action = actions.get(STAGE_CALIBRATION,
                                      "recompute:no_cache_entry")
            attr_action = actions.get(STAGE_ATTRIBUTION,
                                     "recompute:no_cache_entry")
            full_reuse = evidence_action.startswith("reuse") and \
                calib_action.startswith("reuse") and attr_action.startswith("reuse")
            # 片段预算只约束特征/证据提取；证据命中轮的校准重算不占片段预算。
            if full_reuse:
                budget_decision = "reuse"
            elif evidence_action.startswith("reuse"):
                # 证据命中、校准需重算：不占片段预算，也不计复用命中。
                budget_decision = "compute"
            else:
                budget_decision = scheduler.take(rnd, is_reuse=False)
            decision = "reuse" if full_reuse else budget_decision
            if decision == "reuse":
                result.reuse_hits += 1
                batch.reused += 1
                scheduled.add(rnd)
                completed.add(rnd)
                ledger.record_decision(rnd, STAGE_EVIDENCE, "reuse",
                                       evidence_action,
                                       ledger.key_for(STAGE_EVIDENCE, rnd))
                ledger.record_decision(rnd, STAGE_CALIBRATION, "reuse",
                                       calib_action,
                                       ledger.key_for(STAGE_CALIBRATION, rnd))
                ledger.record_decision(rnd, STAGE_ATTRIBUTION, "reuse",
                                       attr_action,
                                       ledger.key_for(STAGE_ATTRIBUTION, rnd))
                _restore_round_artifacts(rnd, persisted, result, config, script)
                continue
            # 至少一个阶段需要重算：记录各阶段动作，证据命中则恢复证据产物。
            ledger.record_decision(
                rnd, STAGE_EVIDENCE,
                "reuse" if evidence_action.startswith("reuse") else "recompute",
                evidence_action, ledger.key_for(STAGE_EVIDENCE, rnd))
            ledger.record_decision(
                rnd, STAGE_CALIBRATION,
                "reuse" if calib_action.startswith("reuse") else "recompute",
                calib_action, ledger.key_for(STAGE_CALIBRATION, rnd))
            ledger.record_decision(
                rnd, STAGE_ATTRIBUTION,
                "reuse" if attr_action.startswith("reuse") else "recompute",
                attr_action, ledger.key_for(STAGE_ATTRIBUTION, rnd))
            if decision == "pending":
                pending_after.append(rnd)
                _add_error_once(inputs.errors, ErrorCode.E_CALIB_BUDGET_PENDING,
                                STAGE_EVIDENCE,
                                "segment budget exhausted, retained for next batch",
                                rnd)
                continue
            scheduled.add(rnd)
            batch_rounds.append(rnd)
            if evidence_action.startswith("reuse"):
                # 证据指纹命中：仅校准/归属重算，直接恢复证据产物且不消耗特征计算。
                _restore_round_artifacts(rnd, persisted, result, config, script)
                continue
            clip = inputs.clips[rnd]
            round_injections = injections_for_round(inputs.injections, rnd)
            samples, hit_ids = apply_sample_injections(
                clip.samples, round_injections)
            for hid in hit_ids:
                result.add_injection_hit(hid, rnd, STAGE_EVIDENCE)
            sort_key = make_sort_key(clip.path.name, STAGE_EVIDENCE)
            feature_tasks.append(TaskEnvelope(
                sort_key=f"{sort_key}|frames", round=rnd,
                func=_task_extract,
                kwargs=dict(round_index=rnd, file_name=clip.path.name,
                            samples=samples, config=config, sort_key=sort_key)))

        # 并发帧特征提取（并发度取配置且不超过任务数）。
        evidence_eligible = set(batch_rounds)
        workers = effective_workers(config.max_workers, len(feature_tasks))
        feature_results = run_tasks(feature_tasks, workers) if feature_tasks else []

        for feat in feature_results:
            rnd = feat.round
            round_injections = injections_for_round(inputs.injections, rnd)
            if has_nonfinite_samples(feat.vector) or has_nonfinite_samples(
                    feat.frame_energy):
                inputs.errors.add(ErrorCode.E_NONFINITE_SAMPLE, STAGE_EVIDENCE,
                                  "non-finite frame feature detected", round=rnd,
                                  file=feat.file,
                                  injection_id=next(
                                      (i.injection_id for i in round_injections
                                       if i.type == "nonfinite_sample"), None))
                ledger.mark_failed(STAGE_EVIDENCE, rnd,
                                   ErrorCode.E_NONFINITE_SAMPLE)
                blocked.add(rnd)
                continue
            ok, code, inj_id = check_evidence_injection(
                feat.vector, round_injections)
            if not ok:
                inputs.errors.add(code, STAGE_EVIDENCE,
                                  "evidence dimensionality injection triggered",
                                  round=rnd, file=feat.file, injection_id=inj_id)
                result.add_injection_hit(inj_id, rnd, STAGE_EVIDENCE)
                ledger.mark_failed(STAGE_EVIDENCE, rnd, code)
                blocked.add(rnd)
                continue
            strength_tasks.append(TaskEnvelope(
                sort_key=make_sort_key(feat.file, "strength"), round=rnd,
                func=_task_strength,
                kwargs=dict(features=feat,
                            audio_hash=inputs.clips[rnd].content_hash,
                            config=config)))

        strength_results = run_tasks(strength_tasks, effective_workers(
            config.max_workers, len(strength_tasks))) if strength_tasks else []
        for artifact in strength_results:
            result.evidence[artifact.round] = artifact
            ledger.put(make_evidence_entry(
                artifact.round, artifact,
                config.evidence_param_fingerprint, config))

        # ---- 串行逐轮定点校准（仅处理本批拿到证据的轮次）----
        prev_row: CalibrationRow | None = None
        # 串行前驱必须是最后一个已完成轮次的真实校准行（含复用恢复的行）。
        done_rounds = sorted(result.calibrations)
        if done_rounds:
            prev_row = result.calibrations[done_rounds[-1]]

        for rnd in inputs.valid_rounds:
            if rnd in completed:
                prev_row = result.calibrations.get(rnd, prev_row)
                continue
            if rnd not in result.evidence:
                if rnd in evidence_eligible:
                    # 本批证据阶段失败：该轮阻断，后续轮次以最后成功前驱继续。
                    _add_error_once(inputs.errors, ErrorCode.E_UPSTREAM_BLOCKED,
                                    STAGE_CALIBRATION,
                                    "upstream evidence unavailable", rnd)
                    ledger.record_decision(rnd, STAGE_CALIBRATION, "recompute",
                                           "upstream_blocked", None)
                    blocked.add(rnd)
                # 预算待办轮次不在本批处理，保留到下一批。
                continue
            script_row = next(s for s in script.rounds if s.round == rnd)
            round_injections = injections_for_round(inputs.injections, rnd)
            ok, code, inj_id = check_gain_injection(config.gain_q, round_injections)
            ok_replace, rep_code, rep_id = check_replace_round(
                rnd, total_rounds, round_injections)
            if not ok:
                inputs.errors.add(code, STAGE_CALIBRATION,
                                  "zero gain injection triggered", round=rnd,
                                  injection_id=inj_id)
                result.add_injection_hit(inj_id, rnd, STAGE_CALIBRATION)
                ledger.mark_failed(STAGE_CALIBRATION, rnd, code)
                blocked.add(rnd)
                continue
            if not ok_replace:
                inputs.errors.add(rep_code, STAGE_CALIBRATION,
                                  "replacement round out of range", round=rnd,
                                  injection_id=rep_id)
                result.add_injection_hit(rep_id, rnd, STAGE_CALIBRATION)
                ledger.mark_failed(STAGE_CALIBRATION, rnd, rep_code)
                blocked.add(rnd)
                continue
            row = calibrate_round(rnd, script_row.speaker, script_row.link,
                                  result.evidence[rnd], prev_row, config)
            result.calibrations[rnd] = row
            ledger.put(make_calib_entry(rnd, row, config))
            ledger.record_decision(rnd, STAGE_CALIBRATION, "recompute",
                                   "serial_fixed_point_step", row.cache_key)

            attribution = attribute_round(rnd, result.evidence[rnd], row, config)
            result.attributions[rnd] = attribution
            ledger.put(make_attr_entry(rnd, attribution, config))
            ledger.record_decision(rnd, STAGE_ATTRIBUTION, "recompute",
                                   "derived_from_calibration",
                                   attribution.cache_key)
            prev_row = row
            completed.add(rnd)

        finished_batch = scheduler.finish_batch()
        if pending_after:
            finished_batch.pending_rounds = sorted(set(pending_after))
        result.batches.append(finished_batch.to_dict())
        # 待办只保留真正尚未完成的轮次。
        result.pending_rounds = sorted(
            rnd for rnd in set(pending_after) if rnd not in completed)
        if not batch_rounds and not result.pending_rounds:
            break

    # 重算清单按账本实际动作汇总（含仅校准/归属重算的下游轮次）。
    valid_set = set(inputs.valid_rounds)
    result.recompute_rounds = sorted({
        d["round"] for d in ledger.decisions
        if d["action"] == "recompute" and d["round"] in valid_set
    })

    # ---- 轨迹重建（全部已完成归属轮次）----
    ordered_attrs = [result.attributions[rnd]
                     for rnd in sorted(result.attributions)]
    _, trajectories = build_trajectories(ordered_attrs, config)
    result.trajectories = trajectories

    ledger.save(ledger_path)

    # ---- 阶段快照（按间隔轮次提交，最终必提交）----
    artifacts = {
        "evidence": [result.evidence[r].to_dict()
                     for r in sorted(result.evidence)],
        "calibration": [result.calibrations[r].to_dict()
                        for r in sorted(result.calibrations)],
        "attribution": [result.attributions[r].to_dict()
                        for r in sorted(result.attributions)],
        "trajectories": [t.to_dict() for t in trajectories],
    }
    cursor = max(result.calibrations.keys(), default=-1)
    handle = commit_snapshot(
        output_dir, STAGE_TRAJECTORY, cursor, config.config_hash,
        config.injection_hash, config.frac_bits, config.quant_gear,
        artifacts, ledger,
        extra={"snapshot_interval_rounds": config.snapshot_interval_rounds,
               "trajectory_session_key": trajectory_session_key(config)})
    result.snapshot_manifest = handle.manifest
    return result



def _add_error_once(collector: ErrorCollector, code: str, stage: str,
                    message: str, rnd: int) -> None:
    if any(record.code == code and record.stage == stage
           and record.round == rnd for record in collector.records):
        return
    collector.add(code, stage, message, round=rnd)


def _restore_round_artifacts(rnd: int, persisted: dict,
                             result: PipelineResult, config: CalibConfig,
                             script: ScriptData) -> None:
    """增量复用时从快照重建产物对象，数值字段逐字段还原。"""
    if rnd in result.evidence:
        return
    ev_list = {item["round"]: item for item in persisted.get("evidence", [])}
    cal_list = {item["round"]: item for item in persisted.get("calibration", [])}
    attr_list = {item["round"]: item for item in persisted.get("attribution", [])}
    if rnd not in ev_list:
        return
    ev = ev_list[rnd]
    artifact = StrengthArtifact(
        round=rnd,
        file=ev["file"],
        raw_scores=list(ev["raw_scores"]),
        softmax_float=list(ev["softmax_float64"]),
        strength_q=list(ev["strength_q"]),
        quant_residuals=list(ev["quant_residuals"]),
        feature_vector=list(ev["feature_vector"]),
        fp_metadata=dict(ev["fixed_point"]),
        cache_key=ev["cache_key"],
        audio_hash=ev["audio_hash"],
        sort_key=f"restore|{ev['file']}",
    )
    result.evidence[rnd] = artifact
    cal = cal_list[rnd]
    result.calibrations[rnd] = CalibrationRow(
        round=rnd,
        speaker=cal["speaker"],
        link=cal["link"],
        link_weight_q=cal["link_weight_q"],
        gain_q=cal["gain_q"],
        decay_q=cal["decay_q"],
        evidence_q=list(cal["evidence_q"]),
        calib_q=list(cal["calib_q"]),
        calib_float64=list(cal["calib_float64"]),
        emotion=cal["emotion"],
        prev_round=cal["prev_round"],
        cache_key=cal["cache_key"],
        evidence_key=cal["evidence_key"],
    )
    attr = attr_list[rnd]
    result.attributions[rnd] = AttributionRow(
        round=rnd,
        speaker=attr["speaker"],
        candidates=list(attr["candidates"]),
        distance_hz=dict(attr["distance_hz"]),
        segment_id=attr.get("segment_id"),
        break_reason=attr.get("break_reason"),
        calib_round=attr["calib_round"],
        f0_hz=attr["f0_hz"],
        evidence_component=attr["evidence_component"],
        cache_key=attr["cache_key"],
    )

"""链路调度器：加载校验 -> VAD -> 分析窗 -> 定点特征 -> 后验 -> 命令结论。

维护分层缓存键与跨窗回执账本，支持片段替换/档位修改的精确失效、
单批重算窗点数预算、待办优先、线程池并发与阶段化快照。
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from dataclasses import asdict, dataclass, field

from .audio import Fragment, load_fragments
from .classifier import WindowPosterior, build_projection, classify_window
from .commands import CommandConclusion, summarize_commands
from .config import PipelineConfig
from .errors import BusinessError, E_WINDOW_BUDGET_PENDING
from .features import (
    WindowFeature,
    assemble_global_audio,
    extract_window_feature,
    fragment_feature_key,
    fragment_offsets_total,
    mel_filterbank,
)
from .fingerprint import stable_key_for_window
from .injection import InjectionConfig, InjectionEngine
from .snapshot import load_snapshot, write_snapshot
from .vad import SpeechInterval, detect_fragment
from .windowing import AnalysisWindow, build_windows


@dataclass
class BatchResult:
    batch_id: int
    recomputed: list[int] = field(default_factory=list)
    reused: list[int] = field(default_factory=list)
    pending: list[int] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    budget_remaining: int = 0
    conclusions: list[dict] = field(default_factory=list)
    budget_pending_records: list[dict] = field(default_factory=list)


class Pipeline:
    def __init__(self, cfg: PipelineConfig, base_dir: str,
                 injection_config: InjectionConfig | None = None,
                 snapshot: bool = True):
        self.cfg = cfg
        self.base_dir = base_dir
        self.engine = InjectionEngine(injection_config)
        self.use_snapshot = snapshot
        self.fragments: dict[int, Fragment] = {}
        self.fragment_keys: dict[int, str] = {}
        self.intervals: dict[int, list[SpeechInterval]] = {}
        self.windows: dict[int, AnalysisWindow] = {}
        self.features: dict[int, WindowFeature] = {}
        self.posteriors: dict[int, WindowPosterior] = {}
        self.conclusions: list[CommandConclusion] = []
        self.invalid_windows: list[int] = []
        self.pending: list[int] = []
        self.completed_fragments: list[int] = []
        self.budget_consumed_total = 0
        self.failed_windows: set[int] = set()
        self.failed_fragments: set[int] = set()
        self.global_version = 0
        self.stage_versions = {
            "vad": 0, "window_feature": 0,
            "posterior": 0, "command_summary": 0,
        }
        self.reuse_ledger: list[dict] = []
        self.failures: list[dict] = []
        self.bank = mel_filterbank(cfg.mel, cfg.sample.sample_rate)
        self.projection = build_projection(cfg)
        self._global_audio = None

    # ---------- 加载与校验 ----------
    def load_all(self) -> list[dict]:
        from .audio import load_manifest, read_wav_bytes

        vocab_words = set(self.cfg.vocab.values())
        manifest_path = f"{self.base_dir}/{self.cfg.paths.manifest_file}"
        wav_root = f"{self.base_dir}/{self.cfg.paths.data_dir}"
        entries = load_manifest(manifest_path, wav_root, vocab_words)
        load_failures: list[dict] = []
        for entry in entries:
            seq = entry["seq"]
            try:
                raw = open(entry["wav_path"], "rb").read()
                raw = self.engine.mutate_wav_bytes(seq, raw)
                injection_id = self.engine.wants_non_finite(seq)
                samples = read_wav_bytes(
                    raw,
                    entry["wav_path"],
                    inject_non_finite=injection_id is not None,
                )
                fragment = Fragment(
                    seq=seq,
                    wav_file=entry["wav_file"],
                    arrival_ms=entry["arrival_ms"],
                    capture_ms=entry["capture_ms"],
                    command_word=entry.get("command_word"),
                    samples=samples,
                    sample_count=len(samples),
                )
                self.ingest_fragment(fragment)
            except BusinessError as error:
                if error.fragment_seq is None:
                    error.fragment_seq = seq
                if not error.stage:
                    error.stage = "audio_load"
                if error.injection_id is None:
                    injected = self.engine._find(
                        "truncate_wav", fragment_seq=seq
                    ) or self.engine._find("non_finite_sample", fragment_seq=seq)
                    if injected is not None:
                        error.injection_id = injected.id
                load_failures.append(error.to_record())
                self.failures.append(error.to_record())
                self.engine.record_failure(error)
                self.failed_fragments.add(seq)
        return load_failures

    def ingest_fragment(self, fragment: Fragment) -> None:
        self.fragments[fragment.seq] = fragment
        self.completed_fragments = sorted(set(self.completed_fragments + [fragment.seq]))
        self._rebuild_front_stage()
        self._invalidate_for_fragment(fragment.seq)

    def replace_fragment(self, fragment: Fragment) -> list[int]:
        old_key = self.fragment_keys.get(fragment.seq)
        self.fragments[fragment.seq] = fragment
        self.completed_fragments = sorted(set(self.completed_fragments + [fragment.seq]))
        self._rebuild_front_stage()
        new_key = fragment_feature_key(fragment, self.cfg)
        affected: list[int] = []
        if old_key != new_key:
            affected = self._invalidate_for_fragment(fragment.seq)
        return sorted(set(affected))

    def update_quant_level(self, quant_level: int) -> list[int]:
        allowed = self.cfg.quant.allowed_levels
        if quant_level not in allowed:
            raise BusinessError(
                "E_QUANT_LEVEL_INVALID",
                f"量化档位 {quant_level} 不在允许枚举 {allowed} 内",
                stage="config",
            )
        old_fp = self.cfg.config_fingerprint()
        self.cfg.quant.quant_level = quant_level
        if self.cfg.config_fingerprint() != old_fp:
            self.features.clear()
            self.posteriors.clear()
            self.invalid_windows = sorted(self.windows)
            for window_index in self.invalid_windows:
                self._mark_invalid(window_index, reason="quant_level_changed")
        return list(self.invalid_windows)

    # ---------- 前端：VAD 与分析窗 ----------
    def _rebuild_front_stage(self) -> None:
        ordered = [self.fragments[seq] for seq in sorted(self.fragments)]
        self._global_audio = assemble_global_audio(ordered)
        self.intervals = {}
        for fragment in ordered:
            self.fragment_keys[fragment.seq] = fragment_feature_key(fragment, self.cfg)
            self.intervals[fragment.seq] = detect_fragment(
                fragment.samples, fragment.seq, self.cfg.vad
            )
        all_intervals = [
            interval
            for seq in sorted(self.intervals)
            for interval in self.intervals[seq]
        ]
        new_windows = {
            window.window_index: window
            for window in build_windows(all_intervals, ordered, self.cfg.window)
        }
        keep = {
            index: window
            for index, window in self.windows.items()
            if index in new_windows and new_windows[index] == window
        }
        self.windows = new_windows
        self.features = {k: v for k, v in self.features.items() if k in keep}
        self.posteriors = {k: v for k, v in self.posteriors.items() if k in keep}
        self.stage_versions["vad"] += 1
        missing = sorted(index for index in self.windows if index not in self.features)
        for index in missing:
            if index not in self.invalid_windows and index not in self.pending:
                self._mark_invalid(index, reason="new_window")

    def _invalidate_for_fragment(self, fragment_seq: int) -> list[int]:
        affected = sorted(
            index
            for index, window in self.windows.items()
            if fragment_seq in window.fragment_seqs
        )
        for index in affected:
            self._mark_invalid(index, reason=f"fragment_{fragment_seq}_changed")
        return affected

    def _mark_invalid(self, window_index: int, *, reason: str) -> None:
        self.features.pop(window_index, None)
        self.posteriors.pop(window_index, None)
        if window_index not in self.invalid_windows:
            self.invalid_windows.append(window_index)
        self.reuse_ledger.append(
            {"window_index": window_index, "action": "invalidated", "reason": reason}
        )

    def _stable_order(self, indices: list[int]) -> list[int]:
        return sorted(
            indices,
            key=lambda index: stable_key_for_window(
                self.windows[index].fragment_seqs, index
            ),
        )

    # ---------- 单窗处理（含注入） ----------
    def _process_window(self, window_index: int) -> None:
        window = self.windows[window_index]
        _, total = fragment_offsets_total(list(self.fragments.values()))
        self.engine.mutate_span(window_index, (window.start_sample, window.end_sample), total)
        self.engine.mutate_quant_level(
            self.cfg.quant.quant_level,
            self.cfg.quant.allowed_levels,
            window_index=window_index,
        )
        feature = extract_window_feature(
            window, self._global_audio, self.cfg, self.bank, self.fragment_keys
        )
        projection = self.projection
        self.engine.mutate_dimension(window_index, projection.shape[1])
        posterior = classify_window(feature, self.cfg, projection)
        self.features[window_index] = feature
        self.posteriors[window_index] = posterior

    def _worker(self, window_index: int) -> tuple[int, object]:
        try:
            self._process_window(window_index)
            return window_index, None
        except BusinessError as error:
            if error.window_index is None:
                error.window_index = window_index
            return window_index, error

    # ---------- 批调度 ----------
    def run_batch(self, budget: int | None = None) -> BatchResult:
        if budget is None:
            budget = self.cfg.runtime.default_budget
        self.global_version += 1
        batch_id = self.global_version
        result = BatchResult(batch_id=batch_id, budget_remaining=budget)
        queue = self._stable_order(self.pending + self.invalid_windows)
        queued = set(queue)
        self.pending = [index for index in self.pending if index not in queued]
        self.invalid_windows = [
            index for index in self.invalid_windows if index not in queued
        ]
        selected: list[int] = []
        for window_index in queue:
            if len(selected) >= budget:
                break
            selected.append(window_index)
        deferred = [index for index in queue if index not in selected]
        if budget == 0:
            selected, deferred = [], queue

        workers = min(self.cfg.runtime.max_workers, max(len(selected), 1))
        if selected:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._worker, index) for index in selected]
                worker_results = [future.result() for future in futures]
        else:
            worker_results = []
        for window_index, error in sorted(worker_results, key=lambda item: item[0]):
            if error is None:
                result.recomputed.append(window_index)
                self.budget_consumed_total += 1
                result.budget_remaining -= 1
                self.reuse_ledger.append(
                    {"window_index": window_index, "action": "recomputed"}
                )
            else:
                result.failures.append(error.to_record())
                self.failures.append(error.to_record())
                self.engine.record_failure(error)
                self.failed_windows.add(window_index)
                self.reuse_ledger.append(
                    {
                        "window_index": window_index,
                        "action": "failed",
                        "code": error.code,
                    }
                )

        for window_index in self._stable_order(deferred):
            self.pending.append(window_index)
            record = BusinessError(
                E_WINDOW_BUDGET_PENDING,
                f"窗 {window_index} 因单批预算耗尽保留待办",
                stage="scheduler",
                window_index=window_index,
            ).to_record()
            result.budget_pending_records.append(record)

        reused = sorted(
            index
            for index in self.windows
            if index in self.posteriors
            and index not in result.recomputed
            and index not in self.failed_windows
        )
        result.reused = reused
        for index in reused:
            self.reuse_ledger.append({"window_index": index, "action": "reused"})

        self._refresh_conclusions(result)
        self.stage_versions["window_feature"] += 1
        self.stage_versions["posterior"] += 1
        self.stage_versions["command_summary"] += 1
        result.pending = list(self.pending)
        if self.use_snapshot:
            self.persist()
        return result

    def _refresh_conclusions(self, result: BatchResult) -> None:
        if self.pending or self.invalid_windows:
            return
        if any(index not in self.posteriors for index in self.windows):
            return
        spans = {
            index: (window.start_sample, window.end_sample)
            for index, window in self.windows.items()
        }
        self.conclusions = summarize_commands(
            [self.posteriors[index] for index in sorted(self.posteriors)],
            spans, self.cfg,
        )
        result.conclusions = [asdict(item) for item in self.conclusions]

    # ---------- 快照 ----------
    def persist(self) -> None:
        state = {
            "fragments": {
                str(seq): {
                    "seq": fragment.seq,
                    "wav_file": fragment.wav_file,
                    "arrival_ms": fragment.arrival_ms,
                    "capture_ms": fragment.capture_ms,
                    "command_word": fragment.command_word,
                    "sample_count": fragment.sample_count,
                    "fragment_key": self.fragment_keys[seq],
                }
                for seq, fragment in self.fragments.items()
            },
            "windows": [asdict(window) for window in self.windows.values()],
            "features": {
                str(index): self._feature_to_dict(feature)
                for index, feature in self.features.items()
            },
            "posteriors": {
                str(index): asdict(posterior)
                for index, posterior in self.posteriors.items()
            },
            "conclusions": [asdict(item) for item in self.conclusions],
            "invalid_windows": list(self.invalid_windows),
            "pending": list(self.pending),
            "completed_fragments": list(self.completed_fragments),
            "budget_consumed_total": self.budget_consumed_total,
            "global_version": self.global_version,
            "stage_versions": self.stage_versions,
            "reuse_ledger": self.reuse_ledger,
            "failures": self.failures,
            "failed_windows": sorted(self.failed_windows),
        }
        write_snapshot(
            f"{self.base_dir}/{self.cfg.paths.snapshot_dir}",
            state, self.cfg, self.stage_versions, self.global_version,
        )

    @staticmethod
    def _feature_to_dict(feature: WindowFeature) -> dict:
        return {
            "window_index": feature.window_index,
            "fragment_seqs": list(feature.fragment_seqs),
            "span": list(feature.span),
            "frame_count": feature.frame_count,
            "fixed_point": list(feature.fixed_point),
            "cache_key": feature.cache_key,
            "fragment_key": feature.fragment_key,
        }

    def resume_if_possible(self) -> str:
        decision = load_snapshot(
            f"{self.base_dir}/{self.cfg.paths.snapshot_dir}", self.cfg
        )
        if decision is None or decision["action"] == "rebuild":
            return "rebuild"
        state = decision["state"]
        self.global_version = state["global_version"]
        self.stage_versions = state["stage_versions"]
        self.pending = list(state["pending"])
        self.invalid_windows = list(state["invalid_windows"])
        self.completed_fragments = list(state["completed_fragments"])
        self.budget_consumed_total = state["budget_consumed_total"]
        self.reuse_ledger = list(state["reuse_ledger"])
        self.failures = list(state["failures"])
        self.failed_windows = set(state.get("failed_windows", []))
        self.windows = {
            item["window_index"]: AnalysisWindow(
                window_index=item["window_index"],
                start_sample=item["start_sample"],
                end_sample=item["end_sample"],
                fragment_seqs=tuple(item["fragment_seqs"]),
                speech_region=tuple(item["speech_region"]),
            )
            for item in state["windows"]
        }
        self.features = {
            int(index): WindowFeature(
                window_index=item["window_index"],
                fragment_seqs=tuple(item["fragment_seqs"]),
                span=tuple(item["span"]),
                frame_count=item["frame_count"],
                frame_features=(),
                fixed_point=tuple(item["fixed_point"]),
                cache_key=item["cache_key"],
                fragment_key=item["fragment_key"],
            )
            for index, item in state["features"].items()
        }
        self.posteriors = {
            int(index): WindowPosterior(
                window_index=item["window_index"],
                scores=tuple(item["scores"]),
                posteriors=tuple(item["posteriors"]),
                predicted_code=item["predicted_code"],
                predicted_word=item["predicted_word"],
                boundary=item["boundary"],
                cache_key=item["cache_key"],
                feature_key=item["feature_key"],
            )
            for index, item in state["posteriors"].items()
        }
        self.conclusions = [
            CommandConclusion(
                conclusion_id=item["conclusion_id"],
                code=item["code"],
                word=item["word"],
                window_indices=tuple(item["window_indices"]),
                sample_span=tuple(item["sample_span"]),
                boundary=item["boundary"],
                cache_key=item["cache_key"],
                posterior_keys=tuple(item["posterior_keys"]),
            )
            for item in state["conclusions"]
        ]
        # 仅依赖已持久化快照：按快照片段清单重新装载 WAV，重建音频内存态
        from .audio import read_wav
        from pathlib import Path
        wav_root = Path(self.base_dir) / self.cfg.paths.data_dir
        for seq_text, item in sorted(state["fragments"].items(), key=lambda kv: int(kv[0])):
            seq = int(seq_text)
            samples = read_wav(wav_root / item["wav_file"])
            self.fragments[seq] = Fragment(
                seq=seq,
                wav_file=item["wav_file"],
                arrival_ms=item["arrival_ms"],
                capture_ms=item["capture_ms"],
                command_word=item["command_word"],
                samples=samples,
                sample_count=item["sample_count"],
            )
            self.fragment_keys[seq] = item["fragment_key"]
        self._global_audio = assemble_global_audio(list(self.fragments.values()))
        return "resume"

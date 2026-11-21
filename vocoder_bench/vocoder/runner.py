"""Concurrent pipeline runner with deterministic stable-key ordering."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import apply_edit, config_fingerprint, phase_profile
from .engine import VocoderEngine, stable_task_key
from .hashing import canonical_hash
from .ledger import Ledger
from .loader import load_inputs
from .snapshot import SnapshotStore
from .wav_io import write_wav


class PipelineRunner:
    def __init__(self, input_dir: str | Path, config: dict, work_dir: str | Path):
        self.input_dir = Path(input_dir)
        self.config = config
        self.work_dir = Path(work_dir)
        self.snapshot_store = SnapshotStore(self.work_dir / "snapshots")
        self.output_dir = self.work_dir / "waveforms"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger()
        self.version = self.snapshot_store.latest_complete_version()
        self.command_cursor = 0
        self._lock = threading.Lock()
        self._last_summary: dict | None = None
        self._current_future = None
        self._executor = ThreadPoolExecutor(max_workers=int(config["max_workers"]))
        self.stop_requested = False

    # ------------------------------------------------------------------ load
    def reload_inputs(self):
        self.utterances, self.failures = load_inputs(
            self.input_dir, int(self.config["harmonic_limit"])
        )
        self._snapshot_stage("load", {
            "utterances": [u.utterance_id for u in self.utterances],
            "failures": [asdict(f) for f in self.failures],
        })
        return self.utterances, self.failures

    # --------------------------------------------------------------- execute
    def _ordered_tasks(self):
        tasks = []
        for utter in self.utterances:
            for frame in utter.frames:
                tasks.append((stable_task_key(utter.utterance_id, frame["frame_index"]),
                              utter, frame["frame_index"]))
        tasks.sort(key=lambda item: item[0])
        return tasks

    def run_full(self) -> dict:
        if not hasattr(self, "utterances"):
            self.reload_inputs()
        self.ledger.reset_hits()
        engine = VocoderEngine(self.config, self.ledger)
        ordered = self._ordered_tasks()
        utterance_ids = sorted({utter.utterance_id for _, utter, _ in ordered})

        results = {}
        # Concurrent utterance pipelines; inside each utterance frame tasks use
        # stable keys, and final outputs are re-sorted before assembly.
        future_map = {
            utter.utterance_id: self._executor.submit(engine.run_utterance, utter)
            for utter in self.utterances
        }
        for utter_id in utterance_ids:
            results[utter_id] = future_map[utter_id].result()

        report = self._assemble_report(results)
        self._persist_results(results)
        self._snapshot_pipeline(report)
        with self._lock:
            self._last_summary = self._summary(report)
        return report

    def run_incremental(self, command: dict) -> dict:
        changed = apply_edit(self.config, command) if command["type"] != "snapshot" else set()
        self.command_cursor = int(command.get("seq", self.command_cursor))
        # Harmonic truncation is applied inside the engine; the on-disk frame
        # width never changes, so no reload is required (cache keys differ
        # because the config fingerprint includes harmonic_limit).
        return self.run_full()

    # -------------------------------------------------------------- snapshots
    def _snapshot_stage(self, stage: str, artifacts: dict) -> None:
        self.version += 1
        completed = [
            f"{u.utterance_id}:{f['frame_index']}"
            for u in getattr(self, "utterances", []) for f in u.frames
        ] if stage != "load" else []
        self.snapshot_store.commit_stage(
            stage=stage,
            version=self.version,
            config_hash=config_fingerprint(self.config),
            phase_bits=int(self.config["phase_bits"]),
            phase_quant_step=int(self.config["phase_quant_step"]),
            completed_frames=completed,
            command_cursor=self.command_cursor,
            artifacts=artifacts,
        )

    def _snapshot_pipeline(self, report: dict) -> None:
        for stage, artifacts in (
            ("phase", {"report": report}),
            ("harmonic", {"report": report}),
            ("overlap_add", {"report": report}),
            ("metric", {"report": report}),
        ):
            self._snapshot_stage(stage, artifacts)

    def _persist_results(self, results: dict) -> None:
        for utter_id, result in results.items():
            write_wav(self.output_dir / f"{utter_id}.fixed.wav",
                      result.wave_pcm.astype(np.float64), self.config["sample_rate"])
            write_wav(self.output_dir / f"{utter_id}.float64.wav",
                      result.wave_float64_pcm.astype(np.float64), self.config["sample_rate"])

    def _assemble_report(self, results: dict) -> dict:
        items = []
        for utter_id in sorted(results):
            result = results[utter_id]
            items.append({
                "utterance_id": utter_id,
                "alignment": result.alignment,
                "metrics": result.metrics,
                "attribution": result.attribution,
                "truncation": result.truncation,
                "frame_summaries": result.frame_summaries,
                "recomputed_cache_keys": result.recompute_keys,
                "referenced_feature_frames": self.ledger.frame_sets[utter_id],
            })
        return {
            "version": self.version,
            "command_cursor": self.command_cursor,
            "config_hash": config_fingerprint(self.config),
            "phase_profile": phase_profile(self.config),
            "config": dict(sorted(self.config.items())),
            "utterances": items,
            "reuse": self.ledger.counters(),
            "reuse_evidence": self.ledger.hit_evidence(),
            "failures": [asdict(f) for f in self.failures],
        }

    def _summary(self, report: dict) -> dict:
        done = len(report["utterances"])
        if done:
            ncc = float(np.mean([u["metrics"]["normalized_cross_correlation"]
                                 for u in report["utterances"]]))
            rmse = float(np.mean([u["metrics"]["waveform_rmse"]
                                  for u in report["utterances"]]))
        else:
            ncc = rmse = 0.0
        return {
            "completed_utterances": done,
            "mean_ncc": round(ncc, 6),
            "mean_rmse": round(rmse, 6),
            "version": report["version"],
            "command_cursor": report["command_cursor"],
            "reuse": report["reuse"],
            "progress": 1.0,
        }

    def latest_summary(self) -> dict | None:
        with self._lock:
            return None if self._last_summary is None else dict(self._last_summary)

    def interactive_dispatch(self, command: dict) -> dict:
        budget = float(self.config["interactive_budget_ms"]) / 1000.0
        start = time.perf_counter()
        try:
            report = self.run_incremental(command)
        except Exception as exc:  # pragma: no cover - defensive
            return {"code": "E_INTERACT_BUDGET_PENDING", "error": str(exc)}
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > budget:
            return {
                "code": "E_INTERACT_BUDGET_PENDING",
                "elapsed_ms": round(elapsed_ms, 3),
                "budget_ms": budget * 1000.0,
                "latest_completed_summary": self.latest_summary(),
                "affected_frames": self._affected_frames(command),
            }
        return {"code": "OK", "elapsed_ms": round(elapsed_ms, 3),
                "summary": self._summary(report), "report": report}

    # Edit type -> pipeline stages whose cache keys it invalidates.
    AFFECTED_STAGES = {
        "f0_ratio": ("phase", "harmonic", "overlap_add", "metric"),
        "phase_bits": ("phase", "harmonic", "overlap_add", "metric"),
        "phase_quant_step": ("phase", "harmonic", "overlap_add", "metric"),
        "harmonic_limit": ("harmonic", "overlap_add", "metric"),
        "window_type": ("overlap_add", "metric"),
        "energy_rounding": ("overlap_add", "metric"),
        "snapshot": (),
    }

    def affected_stages(self, command: dict) -> tuple[str, ...]:
        return self.AFFECTED_STAGES.get(command["type"], ())

    def _affected_frames(self, command: dict) -> list[str]:
        stages = self.affected_stages(command)
        if not stages or not hasattr(self, "utterances"):
            return []
        # Overlap-add / metric artifacts are whole-utterance; represent them by
        # every frame feeding that utterance, which is the recompute set.
        return [
            f"{u.utterance_id}:{f['frame_index']}"
            for u in self.utterances for f in u.frames
        ]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def save_report(self, report: dict, path: str | Path) -> None:
        Path(path).write_text(json.dumps(report, indent=2, sort_keys=True),
                               encoding="utf-8")

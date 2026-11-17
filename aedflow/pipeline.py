"""End-to-end staged pipeline with snapshots, recovery and graceful exit."""

from __future__ import annotations

import hashlib
import json
import signal
import threading
from pathlib import Path
from typing import Any

import yaml

from .boundary import classify_frames, decide_boundaries
from .config import config_fingerprint
from .loader import load_shards
from .matching import evaluate
from .report import atomic_write_json, build_report
from .snapshot import SnapshotStore, STAGE_ORDER, assert_or_reset
from .trajectory import build_trajectories, smooth_boundaries


class GracefulExit(Exception):
    pass


class Pipeline:
    def __init__(self, workspace: Path, config: dict[str, Any], injection: dict[str, Any] | None = None):
        self.workspace = workspace
        self.config = config
        self.injection = injection or {}
        self.fingerprint = config_fingerprint(config)
        self.exit_event = threading.Event()
        self._install_signal_handlers()
        self.context: dict[str, Any] = {
            "failures": [],
            "stage_status": {},
            "exit": {"requested": False, "completed": False, "exit_code": None},
        }

    def _install_signal_handlers(self) -> None:
        def handler(signum: int, frame: Any) -> None:  # pragma: no cover - OS dependent
            self.exit_event.set()

        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                # Non-main threads or unsupported platforms keep running.
                pass

    def request_exit(self) -> None:
        """Test/demo hook equivalent to receiving SIGTERM between stages."""

        self.exit_event.set()

    def _maybe_exit(self, stage: str) -> None:
        trigger_after = self.injection.get("exit_after_stage")
        if self.exit_event.is_set() or trigger_after == stage:
            self.exit_event.set()
            self.context["exit"] = {
                "requested": True,
                "completed": True,
                "exit_code": 0,
                "last_completed_stage": stage,
            }
            raise GracefulExit(stage)

    def _paths(self, key: str) -> Path:
        return self.workspace / self.config["paths"][key]

    def _load_inputs(self) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
        with self._paths("manifest").open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
        with self._paths("truth").open("r", encoding="utf-8") as handle:
            truth = yaml.safe_load(handle)
        return manifest, truth["events"], str(manifest["sample_id"])

    def _store(self, header_version_set: list[str] | None = None) -> SnapshotStore:
        identity = {
            "config_fingerprint": self.fingerprint,
            "header_version_set": header_version_set or [],
            "global_version": 1,
        }
        return SnapshotStore(self._paths("snapshot_dir"), identity)

    def run(self) -> dict[str, Any]:
        manifest, truth_events, sample_id = self._load_inputs()
        pre_store = self._store()
        manifest_file = pre_store.root / "snapshot_manifest.json"
        if pre_store.root.exists() and manifest_file.exists():
            with manifest_file.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
        else:
            existing = None
        resume_stage = existing["stage"] if existing else None

        # Derive the current input identity first, then validate any snapshot.
        fresh_loaded = load_shards(self._paths("shard_dir"), manifest, self.config)
        header_set = fresh_loaded["header_version_set"]
        store = self._store(header_set)
        if existing is not None:
            # Strict whole-snapshot identity gate; stale inputs must not resume.
            assert_or_reset(store)

        # Stage 1: loading and alignment.
        if existing and STAGE_ORDER.index(existing["stage"]) >= STAGE_ORDER.index("load_align"):
            load_artifacts = store.load_json("load_artifacts")
            grid = store.load_array("global_grid")
            self.context["failures"].extend(load_artifacts["failures"])
            completed_shards = load_artifacts["completed_shard_ids"]
        else:
            loaded = fresh_loaded
            grid = loaded["global_grid"]
            self.context["failures"].extend(loaded["failures"])
            completed_shards = loaded["completed_shard_ids"]
            load_artifacts = {
                "failures": loaded["failures"],
                "compat_records": loaded["compat_records"],
                "alignment_residuals": loaded["alignment_residuals"],
                "duplicate_notes": loaded["duplicate_notes"],
                "uncovered_frames": loaded["uncovered_frames"],
                "header_version_set": header_set,
                "completed_shard_ids": completed_shards,
                "frame_ownership": loaded["frame_ownership"],
            }
            store.reset()
            store.commit_stage(
                "load_align",
                completed_shards,
                {"global_grid": grid, "load_artifacts": load_artifacts},
                {"global_version": 1},
            )
            resume_stage = "load_align"

        self._maybe_exit("load_align")

        # Reload rich shard objects needed by later stages (deterministic).
        rich = load_shards(self._paths("shard_dir"), manifest, self.config)
        loaded_shards = rich["loaded_shards"]
        self.context.update(
            {
                "sample_id": sample_id,
                "config_fingerprint": self.fingerprint,
                "header_version_set": header_set,
                "compat_records": load_artifacts["compat_records"],
                "alignment_residuals": load_artifacts["alignment_residuals"],
            }
        )

        # Stage 2: boundary decisions.
        if resume_stage and STAGE_ORDER.index(resume_stage) >= STAGE_ORDER.index("boundary_decision"):
            boundary_payload = store.load_json("boundary_artifacts")
            boundary_failures = boundary_payload["failures"]
            from .models import boundary_from_dict

            decisions = [boundary_from_dict(row) for row in boundary_payload["decisions"]]
        else:
            boundary_result = decide_boundaries(
                loaded_shards,
                grid,
                self.config,
                inject_failure_boundary=self.injection.get("fail_boundary_id"),
            )
            decisions = boundary_result["decisions"]
            boundary_failures = boundary_result["failures"]
            boundary_payload = {
                "decisions": [decision.to_dict() for decision in decisions],
                "failures": boundary_failures,
            }
            store.commit_stage(
                "boundary_decision",
                completed_shards,
                {"boundary_artifacts": boundary_payload},
                {"global_version": 1},
            )
        self.context["failures"].extend(boundary_failures)
        self._maybe_exit("boundary_decision")

        frame_decisions, frame_distribution = classify_frames(
            grid, float(self.config["thresholds"]["frame_posterior"])
        )

        # Stage 3: smoothing and trajectory reconstruction.
        if resume_stage and STAGE_ORDER.index(resume_stage) >= STAGE_ORDER.index("trajectory_build"):
            trajectory_payload = store.load_json("trajectory_artifacts")
            smooth_grid = store.load_array("smoothed_grid")
            from .models import trajectory_from_dict

            trajectories = [trajectory_from_dict(row) for row in trajectory_payload["trajectories"]]
            smoothing_records = trajectory_payload["smoothing_records"]
            trajectory_failures = trajectory_payload["failures"]
        else:
            smoothing_result = smooth_boundaries(
                grid,
                decisions,
                self.config,
                inject_failure_trajectory=self.injection.get("fail_smoothing_boundary_id"),
            )
            smooth_grid = smoothing_result["grid"]
            smoothing_records = smoothing_result["smoothing_records"]
            trajectory_failures = smoothing_result["failures"]
            smooth_frame_decisions, _ = classify_frames(
                smooth_grid, float(self.config["thresholds"]["frame_posterior"])
            )
            built = build_trajectories(
                smooth_frame_decisions,
                load_artifacts["frame_ownership"],
                loaded_shards,
                decisions,
                smoothing_records,
                self.config,
            )
            trajectories = built["trajectories"]
            trajectory_payload = {
                "trajectories": [trajectory.to_dict() for trajectory in trajectories],
                "smoothing_records": smoothing_records,
                "failures": trajectory_failures,
                "counts": built["counts"],
                "dropped_short": built["dropped_short"],
                "duplicates": built["duplicates"],
                "frame_distribution": frame_distribution,
                "frame_decisions": smooth_frame_decisions,
            }
            store.commit_stage(
                "trajectory_build",
                completed_shards,
                {
                    "smoothed_grid": smooth_grid,
                    "trajectory_artifacts": trajectory_payload,
                },
                {"global_version": 1},
            )
        self.context["failures"].extend(trajectory_failures)
        self._maybe_exit("trajectory_build")

        # Stage 4: evaluation.
        if resume_stage and STAGE_ORDER.index(resume_stage) >= STAGE_ORDER.index("match_evaluation"):
            evaluation = store.load_json("evaluation_artifacts")
        else:
            evaluation = evaluate(trajectories, truth_events, self.config)
            store.commit_stage(
                "match_evaluation",
                completed_shards,
                {"evaluation_artifacts": evaluation},
                {"global_version": 1},
            )

        self.context.update(
            {
                "boundary_decisions": decisions,
                "frame_distribution": trajectory_payload.get("frame_distribution", frame_distribution),
                "trajectories": trajectories,
                "trajectory_counts": trajectory_payload["counts"],
                "dropped_short": trajectory_payload["dropped_short"],
                "duplicates": trajectory_payload["duplicates"],
                "smoothing_parameters": {
                    "method": self.config["smoothing"]["method"],
                    "window_frames": self.config["smoothing"]["window_frames"],
                    "records": smoothing_records,
                },
                "evaluation": evaluation,
                "stage_status": {
                    "load_align": "complete",
                    "boundary_decision": "complete",
                    "trajectory_build": "complete",
                    "match_evaluation": "complete",
                },
            }
        )
        run_material = json.dumps(
            {
                "sample_id": sample_id,
                "fingerprint": self.fingerprint,
                "headers": header_set,
                "shards": completed_shards,
            },
            sort_keys=True,
        ).encode("utf-8")
        self.context["run_id"] = hashlib.sha256(run_material).hexdigest()[:12]
        return self.context

    def write_report(self) -> Path:
        report = build_report(self.context)
        path = self._paths("report")
        atomic_write_json(path, report)
        return path

"""Incremental recomputation engine with explicit invalidation closures.

A parameter change maps to its first affected stage (``PARAM_ENTRY_STAGE``);
the closure is that stage plus every downstream stage, applied to the
requested frame set, and cascades into every fixed interval that contains a
recomputed frame. Both the per-frame and interval routes call the same
``process_frame`` primitive and the engine asserts that the actual recomputed
set equals the predicted closure (E_CLOSURE_DIVERGE otherwise). Frames outside
the closure keep their committed 8-bit hash untouched.
"""

import os

import numpy as np

from .config import (
    PARAM_ENTRY_STAGE, STAGES, STAGE_INDEX, ProcessingState,
)
from .errors import ClosureDivergeError
from .hashing import hash_array
from .ledger import append_state, commit_frame_output, log_op
from .metrics import (
    compute_frame_metrics, compute_interval_metrics, default_intervals,
)
from .pipeline import InjectionContext, process_frame
from .synth import FRAME_HEIGHT, FRAME_WIDTH


class IncrementalEngine:
    def __init__(self, work_dir, ledger, manifest, state, *,
                 context=None, intervals=None, width=None, height=None):
        self.work_dir = work_dir
        self.ledger = ledger
        self.manifest = manifest
        self.state = state
        self.context = context or InjectionContext()
        self.width = width or manifest["generator"]["width"]
        self.height = height or manifest["generator"]["height"]
        self.regions = manifest["regions"]
        self.frame_count = manifest["generator"]["frame_count"]
        self.intervals = intervals or default_intervals(self.frame_count)
        self.frames_dir = os.path.join(work_dir, "frames")
        self._last_recompute = None  # {"frames": [...], "stages": [...]}

    # ------------------------------------------------------------------
    # Closure prediction
    # ------------------------------------------------------------------
    def closure_stages(self, old_state, new_state):
        changed = new_state.changed_parameters(old_state)
        if not changed:
            return [], changed
        entry = min(STAGE_INDEX[PARAM_ENTRY_STAGE[p]] for p in changed)
        return list(STAGES[entry:]), changed

    def affected_intervals(self, frames):
        frames = set(frames)
        out = {}
        for name, (lo, hi) in self.intervals.items():
            hit = sorted(f for f in frames if lo <= f < hi)
            if hit:
                out[name] = hit
        return out

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def transition(self, new_state, target_frames, *, route="interval"):
        """Apply a state change and recompute only the closure."""
        new_state.validate()
        stages, changed = self.closure_stages(self.state, new_state)
        target_frames = sorted(int(f) for f in target_frames)
        predicted = {
            "frames": target_frames,
            "stages": stages,
            "intervals": self.affected_intervals(target_frames),
            "changed_parameters": changed,
        }
        record = append_state(self.ledger, new_state)
        seq = record["seq"]

        actual_frames = []
        actual_stages = set()
        for frame_index in target_frames:
            uint8, stage_hashes, _ = process_frame(
                frame_index, self.frames_dir, new_state,
                context=self.context, width=self.width, height=self.height,
            )
            self._assert_closure_hashes(frame_index, stages, stage_hashes,
                                        new_state, seq)
            commit_frame_output(
                self.work_dir, self.ledger, frame_index, uint8, stage_hashes,
                seq, stages,
            )
            actual_frames.append(frame_index)
            actual_stages.update(stages)

        # Frames outside the closure must keep their previous output hash.
        untouched = sorted(set(range(self.frame_count)) - set(target_frames))
        for frame_index in untouched:
            rec = self.ledger["outputs"].get(str(frame_index))
            if rec is not None and rec["seq"] == seq:
                raise ClosureDivergeError(
                    "frame outside closure was rebuilt",
                    details={"frame": frame_index, "seq": seq},
                )

        actual = {
            "frames": actual_frames,
            "stages": sorted(actual_stages, key=STAGE_INDEX.get),
            "intervals": self.affected_intervals(actual_frames),
            "changed_parameters": changed,
        }
        if actual["frames"] != predicted["frames"] or \
                actual["stages"] != predicted["stages"] or \
                actual["intervals"] != predicted["intervals"]:
            raise ClosureDivergeError(
                "actual recompute set differs from closure",
                details={"predicted": predicted, "actual": actual},
            )
        self._last_recompute = actual

        self._recompute_metrics(target_frames, seq)
        log_op(
            self.ledger,
            {
                "seq": seq,
                "op": "transition",
                "route": route,
                "changed_parameters": changed,
                "closure": predicted,
                "actual": actual,
                "state_hash": record["state_hash"],
            },
        )
        self.state = new_state
        return actual

    def transition_per_frame(self, new_state, target_frames):
        """Per-frame route: one op-log entry per frame, same final outputs."""
        new_state.validate()
        stages, changed = self.closure_stages(self.state, new_state)
        target_frames = sorted(int(f) for f in target_frames)
        record = append_state(self.ledger, new_state)
        seq = record["seq"]

        actual_frames, actual_stages = [], set()
        for frame_index in target_frames:
            uint8, stage_hashes, _ = process_frame(
                frame_index, self.frames_dir, new_state,
                context=self.context, width=self.width, height=self.height,
            )
            self._assert_closure_hashes(frame_index, stages, stage_hashes,
                                        new_state, seq)
            commit_frame_output(
                self.work_dir, self.ledger, frame_index, uint8, stage_hashes,
                seq, stages,
            )
            actual_frames.append(frame_index)
            actual_stages.update(stages)
            log_op(
                self.ledger,
                {
                    "seq": seq,
                    "op": "transition_frame",
                    "frame": frame_index,
                    "changed_parameters": changed,
                    "stages": list(stages),
                    "state_hash": record["state_hash"],
                },
            )

        actual = {
            "frames": actual_frames,
            "stages": sorted(actual_stages, key=STAGE_INDEX.get),
            "intervals": self.affected_intervals(actual_frames),
            "changed_parameters": changed,
        }
        self._recompute_metrics(target_frames, seq)
        self._last_recompute = actual
        self.state = new_state
        return actual

    def _state_at_seq(self, seq):
        for record in self.ledger["states"]:
            if record["seq"] == seq:
                return ProcessingState.from_dict(record["state"])
        return self.state

    def _assert_closure_hashes(self, frame_index, stages, stage_hashes,
                               new_state, seq):
        """Stages unaffected since the frame's last commit must match.

        The comparison baseline is the state that produced the frame's prior
        committed output (not merely the previous engine state), so frames
        skipped by earlier transitions are checked against the full chain of
        parameter changes since their own last commit.
        """
        prior = self.ledger["outputs"].get(str(frame_index))
        if prior is None or prior["seq"] == seq:
            return
        baseline = self._state_at_seq(prior["seq"])
        safe_stages, _ = self.closure_stages(baseline, new_state)
        for stage in STAGES:
            if stage in safe_stages:
                break
            old_hash = prior["stage_hashes"].get(stage)
            if old_hash is not None and old_hash != stage_hashes[stage]:
                from .errors import HashMismatchError

                raise HashMismatchError(
                    "pre-closure stage hash changed unexpectedly",
                    details={"frame": frame_index, "stage": stage,
                             "expected": old_hash,
                             "observed": stage_hashes[stage]},
                )

    # ------------------------------------------------------------------
    # Initial full build (only allowed before any output exists)
    # ------------------------------------------------------------------
    def build_all(self):
        if self.ledger["outputs"]:
            raise RuntimeError("build_all is only for an empty ledger")
        record = append_state(self.ledger, self.state)
        seq = record["seq"]
        frames = list(range(self.frame_count))
        all_stages = list(STAGES)
        for frame_index in frames:
            uint8, stage_hashes, _ = process_frame(
                frame_index, self.frames_dir, self.state,
                context=self.context, width=self.width, height=self.height,
            )
            commit_frame_output(
                self.work_dir, self.ledger, frame_index, uint8, stage_hashes,
                seq, all_stages,
            )
        self._recompute_metrics(frames, seq)
        self._recompute_intervals(frames, seq)
        log_op(
            self.ledger,
            {
                "seq": seq,
                "op": "build_all",
                "frames": frames,
                "stages": all_stages,
                "state_hash": record["state_hash"],
            },
        )
        self._last_recompute = {"frames": frames, "stages": all_stages,
                                "intervals": self.affected_intervals(frames),
                                "changed_parameters": []}
        return self._last_recompute

    # ------------------------------------------------------------------
    # Metrics (independent recomputation from output files + truth)
    # ------------------------------------------------------------------
    def load_output(self, frame_index):
        rec = self.ledger["outputs"][str(frame_index)]
        path = os.path.join(self.work_dir, *rec["file"].split("/"))
        raw = np.fromfile(path, dtype=np.uint8)
        return raw.reshape(self.height, self.width, 3)

    def _recompute_metrics(self, frames, seq):
        for frame_index in frames:
            uint8 = self.load_output(frame_index)
            metrics = compute_frame_metrics(
                uint8, self.manifest["frames"][frame_index], self.state,
                self.regions,
            )
            metrics["seq"] = seq
            self.ledger["metrics"][str(frame_index)] = metrics
        self._recompute_intervals(frames, seq)

    def _recompute_intervals(self, frames, seq):
        frames = set(frames)
        for name, (lo, hi) in self.intervals.items():
            if not any(lo <= f < hi for f in frames):
                continue
            members = list(range(lo, hi))
            frame_metrics = [self.ledger["metrics"][str(f)] for f in members]
            pooled = compute_interval_metrics(frame_metrics)
            pooled["seq"] = seq
            self.ledger["interval_metrics"][name] = pooled

    def last_recompute(self):
        return self._last_recompute

    def verify_output_hashes(self):
        """Re-hash every committed output file and compare with the ledger."""
        from .errors import HashMismatchError

        problems = []
        for key, rec in self.ledger["outputs"].items():
            path = os.path.join(self.work_dir, *rec["file"].split("/"))
            arr = np.fromfile(path, dtype=np.uint8).reshape(
                self.height, self.width, 3
            )
            observed = hash_array(arr)
            if observed != rec["output_hash"]:
                problems.append(
                    {"frame": int(key), "expected": rec["output_hash"],
                     "observed": observed}
                )
        if problems:
            raise HashMismatchError(
                "committed output files disagree with ledger hashes",
                details={"problems": problems},
            )
        return True

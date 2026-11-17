"""Parallel reconstruction scheduler with two-layer budget control."""

import json
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import config_hash, camera_hash
from .errors import (
    E_POINT_BUDGET_EXHAUSTED,
    E_SLOT_BUDGET_EXHAUSTED,
    PipelineError,
)
from .features import extract_features
from .geometry import build_depth_map, match_and_triangulate_pair, merge_point_cloud
from .grouping import select_grouping
from .snapshot import SnapshotStore, atomic_write_json


def _pair_plan(group_view_ids, loop_pairs, group_index):
    """Consecutive intra-group pairs plus configured loop-closure pairs."""
    ordered = sorted(group_view_ids)
    pairs = [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]
    for pair in loop_pairs or []:
        a, b = int(pair[0]), int(pair[1])
        if a in ordered and b in ordered:
            pairs.append((a, b))
    unique = sorted({tuple(sorted(pair)) for pair in pairs})
    return [
        {"pair_id": "%d-%d" % pair, "views": pair, "stable_key": (group_index, pair)}
        for pair in unique
    ]


def _serialize_features(records):
    return [
        {
            "view_id": record["view_id"],
            "feature_id": record["feature_id"],
            "x": round(float(record["x"]), 9),
            "y": round(float(record["y"]), 9),
            "stable_key": [round(float(v), 9) for v in record["stable_key"]],
        }
        for record in records
    ]


class ReconstructionPipeline:
    """Runs grouped reconstruction and guarantees serial-equivalent outputs."""

    def __init__(self, views, config, cameras, injection_config=None, snapshot_root=None):
        self.views = sorted(views, key=lambda view: view["view_id"])
        self.views_by_id = {view["view_id"]: view for view in self.views}
        self.config = config
        self.cameras = cameras
        self.injection_config = injection_config or {}
        self.injections = self.injection_config.get("items", {})
        self.snapshot_root = snapshot_root
        self.store = None
        self.shutdown_requested = threading.Event()
        self.previous_signal = None
        self.failures = []
        self.budget_curve = []
        self.deduction_sequence = []
        self.pending_queue = []
        self.features_by_view = {}
        self.matches_by_pair = {}
        self.points_by_pair = {}
        self.depth_summaries = []
        self.depth_artifacts = {}
        self.point_cloud = []
        self.decision = None
        self.resume_point = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # signal handling
    # ------------------------------------------------------------------
    def _handle_signal(self, signum, _frame):
        self.shutdown_requested.set()

    def install_signal_handlers(self):
        self.previous_signal = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def restore_signal_handlers(self):
        if self.previous_signal is not None:
            signal.signal(signal.SIGTERM, self.previous_signal)
            signal.signal(signal.SIGINT, signal.default_int_handler)

    # ------------------------------------------------------------------
    # per-group pure computation (runs on worker threads)
    # ------------------------------------------------------------------
    def _compute_group(self, group):
        group_index = group["group_index"]
        group_features, group_matches, group_points, group_depth = {}, [], [], []
        local_failures = []
        for view_id in group["view_ids"]:
            view = self.views_by_id[view_id]
            try:
                group_features[view_id] = extract_features(view, self.config["feature_threshold"],
                                                           self.injections)
            except PipelineError as exc:
                local_failures.append(self._failure_record(view_id, "features", exc))
        pair_plans = _pair_plan(group["view_ids"], self.config.get("loop_group_pairs"), group_index)
        usable_views = set(group_features)
        for plan in pair_plans:
            a, b = plan["views"]
            if a not in usable_views or b not in usable_views:
                continue
            try:
                matches, points = match_and_triangulate_pair(
                    self.views_by_id[a], group_features[a],
                    self.views_by_id[b], group_features[b],
                    self.config["min_inlier_matches"],
                    self.config["min_parallax_deg"],
                    self.injections,
                )
                group_matches.extend(matches)
                group_points.extend(points)
            except PipelineError as exc:
                local_failures.append(
                    self._failure_record("%d/%d" % (a, b), "matching", exc)
                )
        group_points.sort(key=lambda item: item["stable_key"])
        group_matches.sort(key=lambda item: item["stable_key"])
        for view_id in group["view_ids"]:
            if view_id not in usable_views:
                continue
            try:
                depth, summary = build_depth_map(
                    self.views_by_id[view_id], group_points, self.config
                )
                group_depth.append((view_id, depth, summary))
            except PipelineError as exc:
                local_failures.append(self._failure_record(view_id, "depth", exc))
        group_depth.sort(key=lambda item: item[0])
        return {
            "group": group,
            "features": group_features,
            "matches": group_matches,
            "points": group_points,
            "depth": group_depth,
            "failures": local_failures,
            "slot_demand": group["slot_demand"],
            "point_demand": len(group_points),
            "stable_key": group["stable_key"],
        }

    @staticmethod
    def _failure_record(view_id, stage, exc):
        record = {
            "view_id": view_id,
            "stage": stage,
            "code": exc.code,
            "exception_type": type(exc).__name__,
            "message": exc.message,
        }
        record.update(exc.context)
        return record

    # ------------------------------------------------------------------
    # snapshot helpers
    # ------------------------------------------------------------------
    def _write_features_stage(self, target_dir):
        payload = {
            str(view_id): _serialize_features(records)
            for view_id, records in sorted(self.features_by_view.items())
        }
        atomic_write_json(os.path.join(target_dir, "features.json"), payload)

    def _write_matching_stage(self, target_dir):
        matches = []
        for pair_key in sorted(self.matches_by_pair):
            for match in self.matches_by_pair[pair_key]:
                matches.append({
                    "pair_id": match["pair_id"] if "pair_id" in match else
                    "%d-%d" % (match["view_a"], match["view_b"]),
                    "view_a": match["view_a"],
                    "view_b": match["view_b"],
                    "feature_a": match["feature_a"],
                    "feature_b": match["feature_b"],
                    "distance": round(float(match["distance"]), 9),
                    "stable_key": [
                        match["stable_key"][0], match["stable_key"][1],
                        [round(float(v), 9) for v in match["stable_key"][2]],
                        [round(float(v), 9) for v in match["stable_key"][3]],
                    ],
                })
        points = []
        for pair_key in sorted(self.points_by_pair):
            for point in self.points_by_pair[pair_key]:
                points.append(self._serialize_point(point))
        atomic_write_json(os.path.join(target_dir, "matches.json"), matches)
        atomic_write_json(os.path.join(target_dir, "points3d.json"), points)

    def _write_depth_stage(self, target_dir):
        atomic_write_json(
            os.path.join(target_dir, "depth_summaries.json"),
            sorted(self.depth_summaries, key=lambda item: item["stable_key"]),
        )
        for view_id, depth in sorted(self.depth_artifacts.items()):
            np.save(os.path.join(target_dir, "depth_%03d.npy" % view_id), depth)

    def _write_merge_stage(self, target_dir):
        atomic_write_json(
            os.path.join(target_dir, "point_cloud.json"),
            [self._serialize_point_cloud_entry(entry) for entry in self.point_cloud],
        )

    @staticmethod
    def _serialize_point(point):
        return {
            "xyz": [round(float(v), 9) for v in point["xyz"]],
            "source_views": list(point["source_views"]),
            "pair_id": point["pair_id"],
            "feature_a": point["feature_a"],
            "feature_b": point["feature_b"],
            "parallax_deg": round(float(point["parallax_deg"]), 9),
            "confidence": round(float(point["confidence"]), 9),
            "stable_key": [
                point["stable_key"][0], point["stable_key"][1],
                [round(float(v), 9) for v in point["stable_key"][2]],
                [round(float(v), 9) for v in point["stable_key"][3]],
            ],
        }

    @staticmethod
    def _serialize_point_cloud_entry(entry):
        return {
            "xyz": [round(float(v), 9) for v in entry["xyz"]],
            "voxel_key": list(entry["voxel_key"]),
            "source_views": list(entry["source_views"]),
            "pair_id": entry["pair_id"],
            "confidence": entry["confidence"],
            "member_count": entry["member_count"],
            "stable_key": list(entry["stable_key"]),
        }

    def _persist_manifest_state(self, status_by_group):
        def mutation(manifest):
            manifest["group_progress"] = {
                str(key): value for key, value in sorted(status_by_group.items())
            }
            manifest["consumed_slots"] = self._consumed_slots
            manifest["consumed_points"] = self._consumed_points
            manifest["deduction_sequence"] = list(self.deduction_sequence)
            manifest["pending_queue"] = [entry["group_index"] for entry in self.pending_queue]
            manifest["failures"] = list(self.failures)
        self.store.update_manifest(mutation)

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def run(self):
        if self.snapshot_root:
            self.store = SnapshotStore(
                self.snapshot_root, self.config,
                [{"view_id": v["view_id"], "K": v["K"].tolist(),
                  "R": v["R"].tolist(), "t": v["t"].tolist()} for v in self.views],
            )
            if self.store.manifest["global_version"] > 0:
                self.resume_point = self.store.verify_resumable()
                self._consumed_slots = self.store.manifest["consumed_slots"]
                self._consumed_points = self.store.manifest["consumed_points"]
                self.deduction_sequence = list(self.store.manifest["deduction_sequence"])
                self._committed_group_ids = set(
                    [int(k) for k, v in self.store.manifest["group_progress"].items()
                     if v.get("status") == "committed"]
                )
                # Prior suspension records describe a budget state that may no
                # longer hold; retain only real stage/load failures on resume.
                self.failures = [
                    record for record in self.store.manifest["failures"]
                    if record.get("code") not in (
                        E_SLOT_BUDGET_EXHAUSTED, E_POINT_BUDGET_EXHAUSTED,
                    )
                ]
            else:
                self._consumed_slots = 0
                self._consumed_points = 0
                self._committed_group_ids = set()
        else:
            self._consumed_slots = 0
            self._consumed_points = 0
            self._committed_group_ids = set()

        # Stage 1: feature extraction (per view, in serial order for stable state)
        for view in self.views:
            try:
                records = extract_features(
                    view, self.config["feature_threshold"], self.injections
                )
                self.features_by_view[view["view_id"]] = records
            except PipelineError as exc:
                self.failures.append(self._failure_record(view["view_id"], "features", exc))
        if self.store:
            self.store.commit_stage("features", self._write_features_stage)

        valid_views = [view for view in self.views if view["view_id"] in self.features_by_view]
        self.decision = select_grouping(self.config, valid_views, self.features_by_view)
        groups = sorted(self.decision["groups"], key=lambda group: group["stable_key"])
        groups = [group for group in groups if group["group_index"] not in self._committed_group_ids]

        if self.config["slot_budget"] == 0:
            for group in groups:
                self.pending_queue.append(group)
                self.failures.append({
                    "view_id": "group_%d" % group["group_index"],
                    "stage": "scheduling",
                    "code": E_SLOT_BUDGET_EXHAUSTED,
                    "exception_type": "PipelineError",
                    "message": "slot budget is zero; group retained at stable position",
                    "group_index": group["group_index"],
                })
            groups = []
        elif self.config["point_budget"] == 0:
            for group in groups:
                self.pending_queue.append(group)
                self.failures.append({
                    "view_id": "group_%d" % group["group_index"],
                    "stage": "scheduling",
                    "code": E_POINT_BUDGET_EXHAUSTED,
                    "exception_type": "PipelineError",
                    "message": "point budget is zero; group retained at stable position",
                    "group_index": group["group_index"],
                })
            groups = []

        slot_semaphore = threading.Semaphore(max(1, self.config["slot_budget"]))
        acquired = []
        status_by_group = {}
        if self.store:
            status_by_group = {
                int(key): dict(value)
                for key, value in self.store.manifest["group_progress"].items()
            }
        stop_dispatch = False
        stop_reason = None
        interval = self.config["snapshot_commit_interval"]
        commits_since_checkpoint = 0

        try:
            with ThreadPoolExecutor(max_workers=self.config["thread_count"]) as pool:
                in_flight = {}
                cursor = 0
                # Workers run concurrently, but every merge/commit happens below in
                # strict group stable-key order, guaranteeing serial equivalence.
                while cursor < len(groups) or in_flight:
                    while (
                        not stop_dispatch
                        and cursor < len(groups)
                        and len(in_flight) < self.config["thread_count"]
                    ):
                        group = groups[cursor]
                        if not slot_semaphore.acquire(timeout=0.05):
                            break
                        if self.shutdown_requested.is_set():
                            slot_semaphore.release()
                            stop_dispatch = True
                            stop_reason = "shutdown_signal"
                            break
                        acquired.append(group["group_index"])
                        in_flight[group["group_index"]] = pool.submit(
                            self._compute_group, group
                        )
                        cursor += 1
                    if not in_flight:
                        break
                    next_index = min(in_flight)
                    if stop_dispatch:
                        # Drain in stable order without committing anything: the
                        # budget/shutdown boundary freezes all in-flight groups.
                        ordered = sorted(in_flight)
                        for group_index in ordered:
                            future = in_flight.pop(group_index)
                            future.result()
                            slot_semaphore.release()
                            acquired.remove(group_index)
                        break
                    future = in_flight.pop(next_index)
                    result = future.result()
                    group = result["group"]
                    slot_semaphore.release()
                    acquired.remove(group["group_index"])
                    if self._consumed_points + result["point_demand"] > self.config["point_budget"]:
                        self.pending_queue.append(group)
                        status_by_group[group["group_index"]] = {
                            "status": "suspended",
                            "reason": E_POINT_BUDGET_EXHAUSTED,
                            "stable_key": list(group["stable_key"]),
                        }
                        self.failures.append({
                            "view_id": "group_%d" % group["group_index"],
                            "stage": "scheduling",
                            "code": E_POINT_BUDGET_EXHAUSTED,
                            "exception_type": "PipelineError",
                            "message": "global point budget exhausted",
                            "group_index": group["group_index"],
                        })
                        stop_dispatch = True
                        stop_reason = E_POINT_BUDGET_EXHAUSTED
                        continue
                    self._commit_result(result, status_by_group)
                    commits_since_checkpoint += 1
                    if self.store and commits_since_checkpoint >= interval:
                        self._checkpoint_stages()
                        self._persist_manifest_state(status_by_group)
                        commits_since_checkpoint = 0

                if stop_dispatch:
                    for group in groups[cursor:]:
                        self.pending_queue.append(group)
                    self.pending_queue.sort(key=lambda group: group["stable_key"])
                    for group in self.pending_queue:
                        status_by_group.setdefault(group["group_index"], {
                            "status": "suspended",
                            "reason": stop_reason or "shutdown_signal",
                            "stable_key": list(group["stable_key"]),
                        })
        finally:
            if self.store:
                self._checkpoint_stages()
                self._persist_manifest_state(status_by_group)

        self.pending_queue.sort(key=lambda group: group["stable_key"])

        self._finalize_merge()
        if self.store:
            self.store.commit_stage("merge", self._write_merge_stage)
            self._persist_manifest_state(status_by_group)
            self.resume_point = self.store.verify_resumable()
        self.failures.sort(key=lambda item: (str(item["view_id"]), item["stage"], item["code"]))
        return self._build_report(status_by_group, stopped=stop_dispatch)

    def _commit_result(self, result, status_by_group):
        group = result["group"]
        self.failures.extend(result["failures"])
        for view_id, records in sorted(result["features"].items()):
            self.features_by_view[view_id] = records
        for match in result["matches"]:
            pair_key = (min(match["view_a"], match["view_b"]),
                        max(match["view_a"], match["view_b"]))
            self.matches_by_pair.setdefault(pair_key, [])
            if not any(
                existing["feature_a"] == match["feature_a"]
                and existing["feature_b"] == match["feature_b"]
                and existing["view_a"] == match["view_a"]
                for existing in self.matches_by_pair[pair_key]
            ):
                self.matches_by_pair[pair_key].append(match)
        for point in result["points"]:
            pair_key = (min(point["source_views"]), max(point["source_views"]))
            self.points_by_pair.setdefault(pair_key, []).append(point)
        for pair_key in self.matches_by_pair:
            self.matches_by_pair[pair_key].sort(key=lambda item: item["stable_key"])
        for pair_key in self.points_by_pair:
            self.points_by_pair[pair_key].sort(key=lambda item: item["stable_key"])
        for view_id, depth, summary in result["depth"]:
            self.depth_artifacts[view_id] = depth
            if not any(item["view_id"] == view_id for item in self.depth_summaries):
                self.depth_summaries.append(summary)
        self.depth_summaries.sort(key=lambda item: item["stable_key"])
        self._consumed_slots += result["slot_demand"]
        self._consumed_points += result["point_demand"]
        self.deduction_sequence.append({
            "group_index": group["group_index"],
            "slots": result["slot_demand"],
            "points": result["point_demand"],
            "consumed_slots": self._consumed_slots,
            "consumed_points": self._consumed_points,
            "stable_key": list(group["stable_key"]),
        })
        self.budget_curve.append({
            "group_index": group["group_index"],
            "consumed_slots": self._consumed_slots,
            "consumed_points": self._consumed_points,
        })
        status_by_group[group["group_index"]] = {
            "status": "committed",
            "slot_demand": result["slot_demand"],
            "point_demand": result["point_demand"],
            "stable_key": list(group["stable_key"]),
        }

    def _checkpoint_stages(self):
        self.store.commit_stage("features", self._write_features_stage)
        self.store.commit_stage("matching", self._write_matching_stage)
        self.store.commit_stage("depth", self._write_depth_stage)

    def _finalize_merge(self):
        all_points = []
        for pair_key in sorted(self.points_by_pair):
            all_points.extend(self.points_by_pair[pair_key])
        all_points.sort(key=lambda item: item["stable_key"])
        if not all_points:
            self.point_cloud = []
            return
        self.point_cloud = merge_point_cloud(all_points, self.config["voxel_size"])

    def _build_report(self, status_by_group, stopped):
        return {
            "config_hash": config_hash(self.config),
            "camera_hash": camera_hash([
                {"view_id": v["view_id"], "K": v["K"].tolist(),
                 "R": v["R"].tolist(), "t": v["t"].tolist()}
                for v in self.views
            ]),
            "grouping_decision": {
                "metrics": self.decision["metrics"],
                "candidate_scores": self.decision["candidate_scores"],
                "selected": self.decision["selected"],
                "reason": self.decision["reason"],
            },
            "group_progress": {
                str(key): value for key, value in sorted(status_by_group.items())
            },
            "budget_curve": list(self.budget_curve),
            "deduction_sequence": list(self.deduction_sequence),
            "pending_queue": [entry["group_index"] for entry in self.pending_queue],
            "failures": list(self.failures),
            "resume_point": self.resume_point,
            "shutdown_observed": self.shutdown_requested.is_set(),
            "point_cloud_size": len(self.point_cloud),
        }

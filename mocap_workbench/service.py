"""Application service shared by CLI and the PySide6 desktop shell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import export_snapshot
from .detection import compute_evidence, thresholds_from_mapping
from .generator import GeneratedBundle, generate_bundle, load_bundle
from .errors import BusinessError, ErrorCode
from .model import Snapshot, Thresholds
from .persistence import CommitInfo, RecoveryInfo, StateStore
from .pipeline import (
    apply_incremental,
    build_with_evidence,
    check_consistency,
    rebuild_full_for_consistency,
)
from .preview import PerformanceRecorder, PreviewEngine, PreviewResult, SnapshotManager
from .repair import add_keyframe, remove_keyframe, validate_suppressed_auto


@dataclass
class WorkbenchService:
    workspace: Path
    bundle: GeneratedBundle
    store: StateStore
    recorder: PerformanceRecorder
    manager: SnapshotManager
    recovery: RecoveryInfo
    restored: bool
    restore_stale: dict[str, Any] | None = None

    @classmethod
    def open(
        cls,
        workspace: Path,
        frame_count: int,
        seed: int,
        namespace: str | None = None,
        regenerate: bool = False,
    ) -> "WorkbenchService":
        workspace = Path(workspace).resolve()
        data_dir = workspace / "data" / f"{frame_count}_{seed}"
        if regenerate or not (data_dir / "record.json").exists():
            bundle = generate_bundle(data_dir, frame_count, seed)
        else:
            bundle = load_bundle(data_dir)
        store = StateStore(workspace, namespace or f"frames-{frame_count}-seed-{seed}")
        recovery = store.begin_session()
        restore_stale = None
        try:
            restored_snapshot = store.restore_committed_snapshot(bundle.record, bundle.truth, bundle.defects)
        except BusinessError as exc:
            if exc.code != ErrorCode.E_SNAPSHOT_STALE:
                raise
            restored_snapshot = None
            restore_stale = exc.to_dict()
        if restored_snapshot is not None:
            evidence = restored_snapshot.evidence
            snapshot = restored_snapshot
            restored = True
        else:
            evidence = compute_evidence(bundle.record)
            snapshot = build_with_evidence(
                bundle.record,
                bundle.truth,
                bundle.defects,
                evidence,
                thresholds_from_mapping(),
                frozenset(),
                0,
                committed=True,
            )
            store.commit(snapshot, bundle.defects)
            restored = False
            if restore_stale is not None:
                store.record_event(
                    "stale_commit_replaced",
                    {
                        "reason": restore_stale,
                        "replacement_version": snapshot.version.to_dict(),
                        "record_sha256": bundle.record.content_sha256,
                    },
                    prefix="recovery",
                )
                store.recover_orphans()
        recorder = PerformanceRecorder(workspace, store.namespace)
        engine = PreviewEngine(recorder)
        manager = SnapshotManager(snapshot, engine)
        return cls(
            workspace,
            bundle,
            store,
            recorder,
            manager,
            recovery,
            restored,
            restore_stale=restore_stale,
        )

    @property
    def snapshot(self) -> Snapshot:
        return self.manager.current()

    def preview(
        self,
        thresholds: Thresholds | None = None,
        keyframes: frozenset[int] | set[int] | None = None,
        suppressed_auto: frozenset[int] | set[int] | None = None,
    ) -> PreviewResult:
        return self.manager.engine.generate(
            self.snapshot,
            self.bundle.defects,
            thresholds=thresholds,
            keyframes=keyframes,
            suppressed_auto=suppressed_auto,
        )

    def commit(
        self,
        thresholds: Thresholds | None = None,
        keyframes: frozenset[int] | set[int] | None = None,
        suppressed_auto: frozenset[int] | set[int] | None = None,
        verify_full: bool = True,
    ) -> tuple[Snapshot, CommitInfo, Snapshot | None]:
        base = self.snapshot
        candidate = apply_incremental(
            base,
            self.bundle.defects,
            thresholds=thresholds,
            keyframes=keyframes,
            suppressed_auto=suppressed_auto,
            committed=True,
        )
        full = rebuild_full_for_consistency(candidate, self.bundle.defects) if verify_full else None
        if full is not None:
            candidate = Snapshot(
                record=candidate.record,
                truth=candidate.truth,
                thresholds=candidate.thresholds,
                keyframes=candidate.keyframes,
                evidence=candidate.evidence,
                detection=candidate.detection,
                anchors=candidate.anchors,
                repair=candidate.repair,
                evaluation=candidate.evaluation,
                version=candidate.version,
                suppressed_auto=candidate.suppressed_auto,
                closure=candidate.closure,
                consistency=check_consistency(candidate, full),
                committed=True,
            )
        commit_info = self.store.commit(candidate, self.bundle.defects)
        self.manager.swap_committed(candidate)
        return candidate, commit_info, full

    def export(self, output_dir: Path | None = None) -> dict[str, str]:
        snapshot = self.snapshot
        output_dir = output_dir or self.workspace / "outputs" / f"{snapshot.frame_count}_{snapshot.version.label}"
        return export_snapshot(snapshot, output_dir)

    def add_keyframe(self, frame: int) -> frozenset[int]:
        snapshot = self.snapshot
        return add_keyframe(snapshot.keyframes, frame, snapshot.record, snapshot.detection)

    def remove_user_keyframe(self, frame: int) -> frozenset[int]:
        return remove_keyframe(self.snapshot.keyframes, frame)

    def active_anchor_frames(self) -> set[int]:
        snapshot = self.snapshot
        frames: set[int] = set(snapshot.keyframes)
        for anchors in snapshot.anchors.joint_auto:
            frames.update(anchors)
        frames.update(snapshot.anchors.root_auto)
        frames.difference_update(snapshot.suppressed_auto)
        return frames

    def delete_anchor(self, frame: int) -> tuple[frozenset[int], frozenset[int]]:
        snapshot = self.snapshot
        if frame not in self.active_anchor_frames():
            raise BusinessError(ErrorCode.E_KEYFRAME_INVALID, "anchor does not exist", {"frame": frame})
        keyframes = set(snapshot.keyframes)
        suppressed = set(snapshot.suppressed_auto)
        keyframes.discard(frame)
        auto_frames: set[int] = set(snapshot.anchors.root_auto)
        for anchors in snapshot.anchors.joint_auto:
            auto_frames.update(anchors)
        if frame in auto_frames:
            suppressed.add(frame)
        return frozenset(keyframes), frozenset(suppressed)

    def move_anchor(
        self, old_frame: int, new_frame: int
    ) -> tuple[frozenset[int], frozenset[int]]:
        snapshot = self.snapshot
        active = self.active_anchor_frames()
        if old_frame not in active:
            raise BusinessError(ErrorCode.E_KEYFRAME_INVALID, "anchor does not exist", {"frame": old_frame})
        if new_frame in active:
            raise BusinessError(
                ErrorCode.E_KEYFRAME_INVALID,
                "target frame already has an anchor",
                {"frame": new_frame},
            )
        if snapshot.detection.joint_bad[new_frame, :].any() or snapshot.detection.root_bad[new_frame]:
            raise BusinessError(
                ErrorCode.E_KEYFRAME_INVALID,
                "target frame is defective and cannot be an anchor",
                {"frame": new_frame},
            )
        keyframes = set(snapshot.keyframes)
        suppressed = set(snapshot.suppressed_auto)
        keyframes.discard(old_frame)
        auto_frames: set[int] = set(snapshot.anchors.root_auto)
        for anchors in snapshot.anchors.joint_auto:
            auto_frames.update(anchors)
        if old_frame in auto_frames:
            suppressed.add(old_frame)
        keyframes.add(new_frame)
        return frozenset(keyframes), frozenset(suppressed)

    def close(self) -> Path:
        self.manager.shutdown()
        return self.store.close_cleanly()
